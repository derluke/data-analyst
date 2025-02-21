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
import base64
import json
import pickle
from typing import Any, Awaitable, Optional, TypeVar, cast

import aiofiles
import aiohttp
import datarobot as dr
import pandas as pd
from requests_toolbelt import MultipartEncoder


class AsyncDataRobotStorage:
    def __init__(
        self,
        deployment_id: str,
    ):
        self.client = dr.client.get_client()
        self.deployment_id = deployment_id
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Token {self.client.token}",
                    "Content-Type": "application/json",
                }
            )
        return self._session

    async def _close_session(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self) -> "AsyncDataRobotStorage":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self._close_session()

    async def _get_kv_pair(self, name: str) -> dr.KeyValue:
        """Get a key-value pair asynchronously."""
        loop = asyncio.get_running_loop()
        all_kv_pairs = await loop.run_in_executor(
            None,
            lambda: dr.KeyValue.list(
                self.deployment_id,
                dr.KeyValueEntityType.DEPLOYMENT,
            ),
        )
        kv_pairs = [kv for kv in all_kv_pairs if kv.name == name]
        if not kv_pairs:
            raise ValueError(f"No KeyValue pair found with name: {name}")
        return kv_pairs[0]

    async def _create_or_update_kv(
        self,
        name: str,
        value: Any,
        value_type: dr.KeyValueType,
    ) -> dr.KeyValue:
        """Create or update a key-value pair asynchronously."""
        try:
            kv_pair = await self._get_kv_pair(name)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: kv_pair.update(
                    name=name,
                    category=dr.KeyValueCategory.ARTIFACT,
                    entity_id=self.deployment_id,
                    entity_type=dr.KeyValueEntityType.DEPLOYMENT,
                    value_type=value_type,
                    value=value,
                ),
            )
        except ValueError:
            loop = asyncio.get_running_loop()
            kv_pair = await loop.run_in_executor(
                None,
                lambda: dr.KeyValue.create(
                    name=name,
                    category=dr.KeyValueCategory.ARTIFACT,
                    entity_id=self.deployment_id,
                    entity_type=dr.KeyValueEntityType.DEPLOYMENT,
                    value_type=value_type,
                    value=value,
                ),
            )
        return kv_pair

    async def store_dict(self, name: str, data: dict[str, Any]) -> dr.KeyValue:
        """Store a dictionary asynchronously."""
        return await self._create_or_update_kv(
            name, json.dumps(data), dr.KeyValueType.JSON
        )

    async def retrieve_dict(self, name: str) -> dict[str, Any]:
        """Retrieve a dictionary asynchronously."""
        kv_pair = await self._get_kv_pair(name)
        loop = asyncio.get_running_loop()
        value = await loop.run_in_executor(None, kv_pair.get_value)
        if not isinstance(value, str):
            raise ValueError(f"Expected value to be a string, but got {type(value)}")
        return cast(dict[str, Any], json.loads(value))

    async def store_dataframe(self, name: str, df: pd.DataFrame) -> dr.KeyValue:
        """Store a DataFrame asynchronously."""
        loop = asyncio.get_running_loop()
        serialized_df = await loop.run_in_executor(None, pickle.dumps, df)
        encoded_df = base64.b64encode(serialized_df).decode("ascii")
        return await self._create_or_update_kv(name, encoded_df, dr.KeyValueType.STRING)

    async def retrieve_dataframe(self, name: str) -> pd.DataFrame:
        """Retrieve a DataFrame asynchronously."""
        kv_pair = await self._get_kv_pair(name)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: cast(pd.DataFrame, pickle.loads(base64.b64decode(kv_pair.value))),
        )

    async def store_file(self, name: str, file_path: str) -> str:
        """Store a file in DataRobot storage asynchronously."""
        # Read file asynchronously
        async with aiofiles.open(file_path, "rb") as f:
            file_data = await f.read()

            files = {
                "file": (name, file_data),
                "entityId": self.deployment_id,
                "entityType": dr.KeyValueEntityType.DEPLOYMENT.value,
                "category": dr.KeyValueCategory.ARTIFACT.value,
                "valueType": dr.KeyValueType.BINARY.value,
                "name": name,
            }

            try:
                m = MultipartEncoder(fields=files)
                data = m.to_string()

                session = await self._get_session()
                url = f"{self.client.endpoint}/keyValues/fromFile/"

                async with session.post(
                    url,
                    data=data,
                    headers={
                        "Content-Type": m.content_type,
                        "Accept": "application/json",
                    },
                ) as response:
                    if not response.ok:
                        error_text = await response.text()
                        print(f"\nError response: {error_text}")
                        print(f"Status: {response.status}")
                        print(f"Headers: {response.headers}")
                        raise aiohttp.ClientResponseError(
                            response.request_info,
                            response.history,
                            status=response.status,
                            message=f"{response.reason}: {error_text}",
                        )

                    result = await response.json()
                    return str(result["id"])
            except Exception:
                raise

    async def retrieve_file(
        self, name: str, save_path: str | None = None
    ) -> bytes | None:
        """Retrieve a file from DataRobot storage asynchronously."""
        kv_pair = await self._get_kv_pair(name)
        session = await self._get_session()
        url = f"{self.client.endpoint}/keyValues/{kv_pair.id}/file"

        async with session.get(url) as response:
            response.raise_for_status()
            content = await response.read()

            if save_path:
                async with aiofiles.open(save_path, "wb") as f:
                    await f.write(content)
                return None
            return content

    async def list_artifacts(self) -> list[dr.KeyValue]:
        """List artifacts asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: dr.KeyValue.list(
                self.deployment_id, dr.KeyValueEntityType.DEPLOYMENT
            ),
        )

    async def delete_artifact(self, name: str) -> None:
        """Delete an artifact asynchronously."""
        kv_pair = await self._get_kv_pair(name)
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, kv_pair.delete)
        except dr.errors.ClientError as e:
            if e.error_code == 404:
                pass
            else:
                raise

    async def clear_all(self) -> None:
        """Clear all artifacts asynchronously."""
        artifacts = await self.list_artifacts()
        tasks = [self.delete_artifact(artifact.name) for artifact in artifacts]
        await asyncio.gather(*tasks)


# Backward compatibility wrapper
class DataRobotStorage:
    """Synchronous wrapper for AsyncDataRobotStorage."""

    def __init__(self, deployment_id: str):
        self._async_storage = AsyncDataRobotStorage(deployment_id)
        self.deployment_id = deployment_id
        self.client = dr.client.get_client()

    T = TypeVar("T")

    def _run_async(self, coro: Awaitable[T]) -> T:
        """Run coroutine in the event loop."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)

    def store_dict(self, name: str, data: dict[str, Any]) -> dr.KeyValue:
        return self._run_async(self._async_storage.store_dict(name, data))

    def retrieve_dict(self, name: str) -> dict[str, Any]:
        return self._run_async(self._async_storage.retrieve_dict(name))

    def store_dataframe(self, name: str, df: pd.DataFrame) -> dr.KeyValue:
        return self._run_async(self._async_storage.store_dataframe(name, df))

    def retrieve_dataframe(self, name: str) -> pd.DataFrame:
        return self._run_async(self._async_storage.retrieve_dataframe(name))

    def store_file(self, name: str, file_path: str) -> str:
        return self._run_async(self._async_storage.store_file(name, file_path))

    def retrieve_file(self, name: str, save_path: str | None = None) -> bytes | None:
        return self._run_async(self._async_storage.retrieve_file(name, save_path))

    def list_artifacts(self) -> list[dr.KeyValue]:
        return self._run_async(self._async_storage.list_artifacts())

    def delete_artifact(self, name: str) -> None:
        return self._run_async(self._async_storage.delete_artifact(name))

    def clear_all(self) -> None:
        return self._run_async(self._async_storage.clear_all())
