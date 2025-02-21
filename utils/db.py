# Copyright 2024 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
import os
import shutil
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from json import JSONEncoder
from typing import Any, AsyncGenerator, Generator, Optional, overload

import duckdb
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from utils.resources import LLMDeployment

from .datarobot_storage import AsyncDataRobotStorage
from .schema import AnalystChatMessage


class ChatJSONEncoder(JSONEncoder):
    """Custom JSON encoder to handle special types."""

    def default(self, obj: Any) -> Any:
        try:
            if isinstance(obj, pd.Period):
                return str(obj)
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            if hasattr(obj, "dtype"):
                return obj.item()
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            if hasattr(obj, "to_dict"):
                return obj.to_dict()
            if isinstance(obj, datetime):
                return obj.isoformat()
            return super().default(obj)
        except TypeError:
            return str(obj)  # Fallback to string representation


class ChatHistory(BaseModel):
    user_id: str
    chat_history: dict[str, list[AnalystChatMessage]] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat(),
        }
    )


class ChatPersistence:
    @overload
    def __init__(self, *, user_id: str, db_path: None = None) -> None: ...

    @overload
    def __init__(self, *, user_id: None = None, db_path: str) -> None: ...

    def __init__(
        self, *, user_id: str | None = None, db_path: str | None = None
    ) -> None:
        """Initialize database path and create tables."""
        self.db_path = self.get_db_path(user_id=user_id, db_path=db_path)
        # Initialize tables on first creation
        with self._get_connection() as conn:
            self._init_tables(conn)

    @contextmanager
    def _get_connection(self) -> Generator[duckdb.DuckDBPyConnection, Any, None]:
        """Context manager for database connections."""
        conn = duckdb.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    @asynccontextmanager
    async def _get_async_connection(
        self,
    ) -> AsyncGenerator[duckdb.DuckDBPyConnection, Any]:
        """Async context manager for database connections."""
        loop = asyncio.get_running_loop()
        conn = await loop.run_in_executor(None, duckdb.connect, self.db_path)
        try:
            yield conn
        finally:
            await loop.run_in_executor(None, conn.close)

    def _init_tables(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Create the chat_history table if it doesn't exist."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                user_id VARCHAR,
                chat_history JSON,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                PRIMARY KEY (user_id)
            )
        """)

    def _serialize_messages(
        self, chat_history: dict[str, list[AnalystChatMessage]]
    ) -> str:
        """Serialize messages list to JSON string."""
        try:
            return json.dumps(
                {
                    k: [msg.model_dump() for msg in messages]
                    for k, messages in chat_history.items()
                },
                cls=ChatJSONEncoder,
            )
        except Exception:
            raise

    def _deserialize_messages(
        self, messages_json: str
    ) -> dict[str, list[AnalystChatMessage]]:
        """Deserialize JSON string back to list of AnalystChatMessage objects."""
        try:
            messages_data = json.loads(messages_json)
            return {
                chat_name: [AnalystChatMessage.model_validate(msg) for msg in messages]
                for chat_name, messages in messages_data.items()
            }
        except Exception:
            raise

    async def save_chat_history(self, history: ChatHistory) -> None:
        """Save or update chat history for a user."""
        try:
            messages_json = self._serialize_messages(history.chat_history)
            async with self._get_async_connection() as conn:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None,
                    conn.execute,
                    """
                    INSERT OR REPLACE INTO chat_history
                    (user_id, chat_history, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    [
                        history.user_id,
                        messages_json,
                        history.created_at,
                        datetime.now().isoformat(),
                    ],
                )
        except Exception:
            raise

    async def get_chat_history(self, user_id: str) -> Optional[ChatHistory]:
        """Retrieve chat history for a user."""
        async with self._get_async_connection() as conn:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: conn.execute(
                    """
                    SELECT user_id, chat_history, created_at, updated_at
                    FROM chat_history
                    WHERE user_id = ?
                    """,
                    [user_id],
                ).fetchone(),
            )

            if result:
                try:
                    messages = self._deserialize_messages(result[1])
                    return ChatHistory(
                        user_id=result[0],
                        chat_history=messages,
                        created_at=result[2],
                        updated_at=result[3],
                    )
                except Exception:
                    raise
        return None

    async def append_message(
        self, user_id: str, chat_name: str, message: AnalystChatMessage
    ) -> None:
        """Append a new message to a user's chat history."""
        history = await self.get_chat_history(user_id)
        if history:
            if chat_name in history.chat_history:
                history.chat_history[chat_name].append(message)
                history.updated_at = datetime.now()
                await self.save_chat_history(history)
            else:
                history.chat_history[chat_name] = [message]
                history.updated_at = datetime.now()
                await self.save_chat_history(history)
        else:
            history = ChatHistory(
                user_id=user_id,
                chat_history={chat_name: [message]},
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            await self.save_chat_history(history)

    async def get_all_users(self) -> list[dict[str, Any]]:
        """Get list of all users and their last activity."""
        async with self._get_async_connection() as conn:
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(
                None,
                lambda: conn.execute("""
                    SELECT user_id, updated_at
                    FROM chat_history
                    ORDER BY updated_at DESC
                """).fetchall(),
            )
            return [{"user_id": row[0], "last_active": row[1]} for row in results]

    async def delete_chat_history(self, user_id: str) -> None:
        """Delete chat history for a specific user."""
        async with self._get_async_connection() as conn:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                conn.execute,
                "DELETE FROM chat_history WHERE user_id = ?",
                [user_id],
            )

    async def persist_data(self) -> None:
        """Asynchronously persist chat history to DataRobot storage."""
        deployment_id = LLMDeployment().id

        backup_path = f"{self.db_path}.bak"
        try:
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(
                    None, shutil.copyfile, self.db_path, backup_path
                )
            except Exception:
                raise  # Re-raise to trigger the finally block

            async with AsyncDataRobotStorage(
                deployment_id=deployment_id
            ) as datarobot_storage:
                try:
                    await datarobot_storage.delete_artifact(self.db_path)
                except Exception:
                    pass
                try:
                    await datarobot_storage.store_file(self.db_path, backup_path)
                except Exception:
                    raise
        except Exception:
            raise
        finally:
            if os.path.exists(backup_path):
                await loop.run_in_executor(None, os.remove, backup_path)

    @classmethod
    async def load_data(
        cls, user_id: str | None = None, db_path: str | None = None
    ) -> None:
        """Asynchronously load chat history from DataRobot storage."""
        db_path = cls.get_db_path(user_id, db_path)
        deployment_id = LLMDeployment().id
        async with AsyncDataRobotStorage(
            deployment_id=deployment_id
        ) as datarobot_storage:
            # Download from datarobot storage asynchronously
            await datarobot_storage.retrieve_file(db_path, db_path)

    @staticmethod
    def get_db_path(user_id: str | None = None, db_path: str | None = None) -> str:
        """Return the database path for a given user."""
        if user_id is not None and db_path is None:
            db_path = f"{user_id}_chat_history.db"
        elif user_id is None and db_path is not None:
            user_id = None
            db_path = "chat_history.db"
        else:
            raise ValueError(
                "Either user_id or db_path must be provided, but not both."
                f"Received: user_id={user_id}, db_path={db_path}"
            )
        return db_path
