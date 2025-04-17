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

from typing import Any

import plotly.graph_objects as go
import pytest
import pytest_asyncio

from utils.analyst_db import AnalystDB, DataSourceType
from utils.schema import (
    AnalystDataset,
    CleansedDataset,
    DataDictionary,
    DataDictionaryColumn,
    GetBusinessAnalysisRequest,
    GetBusinessAnalysisResult,
    RunAnalysisRequest,
    RunAnalysisResult,
    RunChartsRequest,
    RunChartsResult,
)


@pytest_asyncio.fixture(scope="module")
async def dataset_cleansed(
    pulumi_up: Any, dataset_loaded: AnalystDataset, analyst_db: AnalystDB
) -> CleansedDataset:
    from utils.api import (
        cleanse_dataframe,
    )

    result = await cleanse_dataframe(dataset_loaded)
    await analyst_db.register_dataset(result, data_source=DataSourceType.FILE)
    return result


def test_dataset_is_cleansed(dataset_cleansed: CleansedDataset) -> None:
    assert dataset_cleansed.cleaning_report is not None


@pytest_asyncio.fixture(scope="module")
async def cleansed_dataset_from_api(
    pulumi_up: Any,
    dataset_loaded: AnalystDataset,
    analyst_db: AnalystDB,
    dataset_cleansed: CleansedDataset,
) -> CleansedDataset:
    from utils.rest_api import get_cleansed_dataset

    # We need to register the dataset first to ensure it exists in the database
    cleansed_dataset = await get_cleansed_dataset(
        name=dataset_loaded.name, skip=0, limit=10000, analyst_db=analyst_db
    )
    return cleansed_dataset


@pytest_asyncio.fixture(scope="module")
async def cleansed_dataset_with_pagination(
    pulumi_up: Any,
    dataset_loaded: AnalystDataset,
    analyst_db: AnalystDB,
    dataset_cleansed: CleansedDataset,
) -> tuple[CleansedDataset, CleansedDataset, CleansedDataset]:
    from utils.rest_api import get_cleansed_dataset

    # Get with skip=0, limit=2
    dataset1 = await get_cleansed_dataset(
        name=dataset_loaded.name, skip=0, limit=2, analyst_db=analyst_db
    )

    # Get with skip=2, limit=2
    dataset2 = await get_cleansed_dataset(
        name=dataset_loaded.name, skip=2, limit=2, analyst_db=analyst_db
    )

    # Get with skip exceeding dataset size
    dataset3 = await get_cleansed_dataset(
        name=dataset_loaded.name, skip=10000, limit=2, analyst_db=analyst_db
    )

    return dataset1, dataset2, dataset3


def test_get_cleansed_dataset_by_name_api(
    dataset_cleansed: CleansedDataset, cleansed_dataset_from_api: CleansedDataset
) -> None:
    # Verify we can retrieve the cleansed dataset by name
    assert cleansed_dataset_from_api is not None
    assert cleansed_dataset_from_api.name == dataset_cleansed.name
    assert len(cleansed_dataset_from_api.cleaning_report) == len(
        dataset_cleansed.cleaning_report
    )

    # Check that the dataframes have the same shape
    assert cleansed_dataset_from_api.to_df().shape == dataset_cleansed.to_df().shape


def test_get_cleansed_dataset_with_pagination(
    dataset_cleansed: CleansedDataset,
    cleansed_dataset_with_pagination: tuple[
        CleansedDataset, CleansedDataset, CleansedDataset
    ],
) -> None:
    dataset1, dataset2, dataset3 = cleansed_dataset_with_pagination

    # First dataset should have at most 2 rows
    assert dataset1.dataset.to_df().shape[0] <= 2

    # Second dataset should have skip=2, so it should start from the 3rd row of the original dataset
    if dataset_cleansed.dataset.to_df().shape[0] > 2:
        # Only verify if the original dataset has enough rows
        first_row_of_second_batch = dataset2.dataset.to_df().row(0, named=True)
        third_row_of_original = dataset_cleansed.dataset.to_df().row(2, named=True)

        # Compare a few columns to verify they match
        for col in dataset2.dataset.to_df().columns[:3]:  # Check first 3 columns
            if col in third_row_of_original and col in first_row_of_second_batch:
                assert first_row_of_second_batch[col] == third_row_of_original[col]

    # Third dataset should be empty (skip > dataset size)
    assert dataset3.dataset.to_df().shape[0] == 0


@pytest_asyncio.fixture(scope="module")
async def data_dictionary(
    pulumi_up: Any,
    dataset_loaded: AnalystDataset,
    analyst_db: AnalystDB,
) -> DataDictionary:
    from utils.api import (
        get_dictionary,
    )

    dictionary_result = await get_dictionary(dataset_loaded)
    await analyst_db.register_data_dictionary(dictionary_result)

    return dictionary_result


@pytest.fixture
def question() -> str:
    return "What are some interesting insights about the medication?"


@pytest.fixture
def run_analysis_request(
    pulumi_up: Any,
    dataset_cleansed: CleansedDataset,
    data_dictionary: DataDictionary,
    question: str,
    analyst_db: AnalystDB,
) -> RunAnalysisRequest:
    analysis_request = RunAnalysisRequest(
        dataset_names=[dataset_cleansed.name],
        question=question,
    )
    return analysis_request


@pytest.fixture
def run_analysis_result_canned() -> RunAnalysisResult:
    with open("tests/models/run_analysis_result.json") as f:
        return RunAnalysisResult.model_validate_json(f.read())


@pytest.fixture
def run_charts_result_canned() -> RunChartsResult:
    with open("tests/models/run_charts_result.json") as f:
        return RunChartsResult.model_validate_json(f.read())


@pytest.fixture
def run_business_result_canned() -> GetBusinessAnalysisResult:
    with open("tests/models/run_business_result.json") as f:
        return GetBusinessAnalysisResult.model_validate_json(f.read())


@pytest.fixture
def chart_request(
    pulumi_up: Any, run_analysis_result_canned: RunAnalysisResult, question: str
) -> RunChartsRequest:
    # Prepare requests
    chart_request = RunChartsRequest(
        dataset=run_analysis_result_canned.dataset,
        question=question,
    )
    return chart_request


@pytest.fixture
def business_request(
    pulumi_up: Any, run_analysis_result_canned: RunAnalysisResult, question: str
) -> GetBusinessAnalysisRequest:
    assert run_analysis_result_canned.dataset is not None
    business_request = GetBusinessAnalysisRequest(
        dataset=run_analysis_result_canned.dataset,
        dictionary=DataDictionary(
            name="analysis_result",
            column_descriptions=[
                DataDictionaryColumn(
                    column=col,
                    description="Analysis result column",
                    data_type=str(
                        run_analysis_result_canned.dataset.to_df()[col].dtype
                    ),
                )
                for col in run_analysis_result_canned.dataset.to_df().columns
            ],
        ),
        question=question,
    )
    return business_request


@pytest.mark.asyncio
async def test_run_analysis(
    pulumi_up: Any,
    run_analysis_request: RunAnalysisRequest,
    dataset_loaded: AnalystDataset,
    analyst_db: AnalystDB,
) -> None:
    from utils.api import (
        run_analysis,
    )

    run_analysis_result = await run_analysis(run_analysis_request, analyst_db)

    assert run_analysis_result.code is not None
    assert len(run_analysis_result.code) > 1
    assert run_analysis_result.dataset is not None
    df = run_analysis_result.dataset.to_df()
    assert df.shape[0] > 0
    assert run_analysis_result.status == "success"


@pytest.mark.asyncio
async def test_run_charts_analysis(
    pulumi_up: Any, chart_request: RunChartsRequest
) -> None:
    from utils.api import (
        run_charts,
    )

    run_charts_result = await run_charts(chart_request)
    assert isinstance(run_charts_result.fig1, go.Figure)
    assert isinstance(run_charts_result.fig2, go.Figure)
    assert run_charts_result.code is not None
    assert len(run_charts_result.code) > 1


@pytest.mark.asyncio
async def test_run_business_analysis(
    pulumi_up: Any,
    business_request: GetBusinessAnalysisRequest,
) -> None:
    from utils.api import (
        get_business_analysis,
    )

    run_business_result = await get_business_analysis(business_request)
    assert len(run_business_result.bottom_line) > 1
    assert len(run_business_result.additional_insights) > 1
    assert len(run_business_result.follow_up_questions) > 0


# TODO: add tests of reflection in run_analysis once test_api refactored/cleaned up
