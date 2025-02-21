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

import os
import tempfile
from typing import Any

import pandas as pd
import pytest

from utils.datarobot_storage import AsyncDataRobotStorage
from utils.resources import LLMDeployment


@pytest.fixture(scope="session")
def deployment_id() -> str:
    return LLMDeployment().id


@pytest.fixture(scope="session")
def sample_dict() -> dict[str, Any]:
    return {
        "text": "sample text",
        "number": 42,
        "nested": {"a": 1, "b": 2},
        "list": [1, 2, 3],
    }


@pytest.fixture(scope="session")
def sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "str_col": ["a", "b", "c"],
            "date_col": pd.date_range("2024-01-01", periods=3),
            "period_col": [
                pd.Period("2024-01"),
                pd.Period("2024-02"),
                pd.Period("2024-03"),
            ],
            "bool_col": [True, False, True],
            "category_col": pd.Categorical(["x", "y", "z"]),
            "nullable_int": pd.Series([1, None, 3], dtype="Int64"),
        }
    )


@pytest.fixture(scope="session")
def test_file_path() -> str:
    # Don't create the file here, just return a path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tf:
        return tf.name


@pytest.mark.asyncio
class TestAsyncDataRobotStorage:
    @pytest.fixture(scope="class")
    def storage(self, deployment_id: str) -> AsyncDataRobotStorage:
        return AsyncDataRobotStorage(deployment_id)

    async def test_store_and_retrieve_dict(
        self, storage: AsyncDataRobotStorage, sample_dict: dict[str, Any]
    ) -> None:
        async with storage as storage_ctx:
            await storage_ctx.store_dict("test_dict", sample_dict)
            retrieved = await storage_ctx.retrieve_dict("test_dict")
            assert retrieved == sample_dict

    async def test_store_and_retrieve_dataframe(
        self, storage: AsyncDataRobotStorage, sample_dataframe: pd.DataFrame
    ) -> None:
        async with storage as storage_ctx:
            await storage_ctx.store_dataframe("test_df", sample_dataframe)
            retrieved = await storage_ctx.retrieve_dataframe("test_df")
            pd.testing.assert_frame_equal(retrieved, sample_dataframe)

    async def test_store_and_retrieve_file(
        self, storage: AsyncDataRobotStorage, test_file_path: str
    ) -> None:
        async with storage as storage_ctx:
            # Create test file with content
            test_content = (
                b"This is test content\nWith multiple lines\nAnd some numbers 123!"
            )
            with open(test_file_path, "wb") as f:
                f.write(test_content)
            try:
                await storage_ctx.delete_artifact("test_file.txt")
            except Exception:
                pass
            try:
                # Store file using the file path
                await storage_ctx.store_file("test_file.txt", test_file_path)

                # Retrieve without saving
                content = await storage_ctx.retrieve_file("test_file.txt")
                assert content == test_content

                # Test retrieving with saving
                save_path = test_file_path + ".retrieved"
                try:
                    await storage_ctx.retrieve_file("test_file.txt", save_path)
                    with open(save_path, "rb") as f:
                        saved_content = f.read()
                    assert saved_content == test_content
                finally:
                    if os.path.exists(save_path):
                        os.unlink(save_path)
            finally:
                if os.path.exists(test_file_path):
                    os.unlink(test_file_path)

    async def test_list_and_delete_artifacts(
        self, storage: AsyncDataRobotStorage
    ) -> None:
        async with storage as storage_ctx:
            # Store some test artifacts
            await storage_ctx.store_dict("list_test_dict1", {"test": 1})
            await storage_ctx.store_dict("list_test_dict2", {"test": 2})

            # List artifacts
            artifacts = await storage_ctx.list_artifacts()
            assert len(artifacts) > 0

            # Verify our test artifacts are present
            artifact_names = [a.name for a in artifacts]
            assert "list_test_dict1" in artifact_names
            assert "list_test_dict2" in artifact_names

            # Delete one artifact
            await storage_ctx.delete_artifact("list_test_dict1")
            updated_artifacts = await storage_ctx.list_artifacts()
            updated_names = [a.name for a in updated_artifacts]
            assert "list_test_dict1" not in updated_names
            assert "list_test_dict2" in updated_names
