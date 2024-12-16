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
import os

import pandas as pd
import pytest
from datarobotx.idp.datasets import get_or_create_dataset_from_df
from fastapi.testclient import TestClient

# Import FastAPI functions directly
from utils.api import app, cleanse_dataframes, get_dictionary
from utils.schema import CleanseRequest, DatasetInput

client = TestClient(app)


@pytest.fixture(scope="module")
def diabetes_dataset_url():
    return "https://s3.amazonaws.com/datarobot_public_datasets/10k_diabetes_20.csv"


@pytest.fixture(scope="module")
def dataset_loaded(diabetes_dataset_url):
    df = pd.read_csv(diabetes_dataset_url)
    # Replace non-JSON compliant values
    df = df.replace([float("inf"), -float("inf")], None)  # Replace infinity with None
    df = df.where(pd.notnull(df), None)  # Replace NaN with None

    # Create dataset dictionary
    dataset = {
        "name": os.path.splitext(os.path.basename(diabetes_dataset_url))[0],
        "data": df.to_dict("records"),
    }
    return dataset


@pytest.fixture(scope="module")
def diabetes_dataset_id(diabetes_dataset_url, dr_client):
    df = pd.read_csv(
        diabetes_dataset_url,
    )
    return get_or_create_dataset_from_df(
        dr_client.endpoint,
        dr_client.token,
        "10k_diabetes_20.csv",
        df,
    )


@pytest.fixture(scope="module")
def dataset_cleaned(dataset_loaded):
    request = CleanseRequest(datasets=[DatasetInput(**dataset_loaded)])
    result = asyncio.run(cleanse_dataframes(request))
    return result.model_dump()


def test_dataset_is_cleansed(dataset_cleaned):
    assert dataset_cleaned["metadata"]["total_datasets"] == 1


@pytest.fixture(scope="module")
def data_dictionary(dataset_loaded):
    dict_request = CleanseRequest(datasets=[DatasetInput(**dataset_loaded)])
    dictionary_result = await get_dictionary(dict_request)
    return dictionary_result.model_dump()
