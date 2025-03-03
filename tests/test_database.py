# test database# Copyright 2024 DataRobot, Inc.
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

from pathlib import Path

import pandas as pd
import pytest

from utils.analyst_db import AnalystDB
from utils.schema import (
    AnalystDataset,
)


async def get_analyst_db(db_version: int) -> AnalystDB:
    analyst_db = await AnalystDB.create(
        user_id="test_user_123",
        db_path=Path("."),
        dataset_db_name="datasets",
        chat_db_name="chats",
        db_version=db_version,
    )
    return analyst_db


@pytest.mark.asyncio
async def test_drop_tables(url_diabetes: str, dataset_loaded: AnalystDataset) -> None:
    assert dataset_loaded.columns is not None

    df = pd.read_csv(url_diabetes)
    # Replace non-JSON compliant values
    df = df.replace([float("inf"), -float("inf")], None)  # Replace infinity with None
    df = df.where(pd.notnull(df), None)  # Replace NaN with None

    # Create dataset dictionary
    dataset = AnalystDataset(
        name="new_dataset_name",
        data=df,
    )
    new_db = await get_analyst_db(2)
    await new_db.register_dataset(dataset)

    assert len(await new_db.list_analyst_datasets()) == 1
