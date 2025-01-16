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
import plotly.graph_objects as go
import pytest

from utils.api import (
    cleanse_dataframes,
    get_business_analysis,
    get_dictionary,
    run_analysis,
    run_charts,
)
from utils.schema import (
    BusinessAnalysisRequest,
    BusinessAnalysisResult,
    CleanseResult,
    DataDictionariesAndMetadata,
    DatasetInput,
    RunAnalysisRequest,
    RunAnalysisResult,
    RunChartsRequest,
    RunChartsResult,
)


@pytest.fixture(scope="module")
def dataset_loaded(url_diabetes: str) -> DatasetInput:
    df = pd.read_csv(url_diabetes)
    # Replace non-JSON compliant values
    df = df.replace([float("inf"), -float("inf")], None)  # Replace infinity with None
    df = df.where(pd.notnull(df), None)  # Replace NaN with None

    # Create dataset dictionary
    dataset = DatasetInput(
        name=os.path.splitext(os.path.basename(url_diabetes))[0],
        data=df.to_dict("records"),
    )
    return dataset


@pytest.fixture(scope="module")
def dataset_cleansed(dataset_loaded: DatasetInput) -> CleanseResult:
    result = asyncio.run(cleanse_dataframes([dataset_loaded]))
    return result


def test_dataset_is_cleansed(dataset_cleansed: CleanseResult) -> None:
    assert dataset_cleansed.metadata.total_datasets == 1


@pytest.fixture(scope="module")
def data_dictionary(dataset_loaded: DatasetInput) -> DataDictionariesAndMetadata:
    # TODO change fixtures to pytest standard async runs
    dictionary_result = asyncio.run(get_dictionary([dataset_loaded]))
    return dictionary_result


@pytest.fixture
def question() -> str:
    return "What are some interesting insights about the medication?"


@pytest.fixture
def run_analysis_request(
    dataset_cleansed: CleanseResult,
    data_dictionary: DataDictionariesAndMetadata,
    question: str,
) -> RunAnalysisRequest:
    analysis_request = RunAnalysisRequest(
        data={ds.name: ds.data for ds in dataset_cleansed.datasets},
        dictionary={
            d["name"]: d["dictionary"]
            for d in data_dictionary.model_dump()["dictionaries"]
        },
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
def run_business_result_canned() -> BusinessAnalysisResult:
    with open("tests/models/run_business_result.json") as f:
        return BusinessAnalysisResult.model_validate_json(f.read())


@pytest.fixture
def run_analysis_result(run_analysis_request: RunAnalysisRequest) -> RunAnalysisResult:
    result = asyncio.run(run_analysis(run_analysis_request))
    return result


@pytest.fixture
def chart_request(
    run_analysis_result_canned: RunAnalysisResult, question: str
) -> RunChartsRequest:
    # Prepare requests
    chart_request = RunChartsRequest(
        data=run_analysis_result_canned.data,
        question=question,
    )
    return chart_request


@pytest.fixture
def business_request(
    run_analysis_result_canned: RunAnalysisResult, question: str
) -> BusinessAnalysisRequest:
    df = pd.DataFrame.from_records(run_analysis_result_canned.data)
    business_request = BusinessAnalysisRequest(
        data=run_analysis_result_canned.data,
        dictionary=[
            {
                "column": col,
                "description": "Analysis result column",
                "data_type": str(df[col].dtype),
            }
            for col in df.columns
        ],
        question=question,
    )
    return business_request


@pytest.fixture
def run_charts_result(chart_request: RunChartsRequest) -> RunChartsResult:
    run_charts_result = asyncio.run(run_charts(chart_request))
    return run_charts_result


@pytest.fixture
def run_business_result(
    business_request: BusinessAnalysisRequest,
) -> BusinessAnalysisResult:
    run_business_result = asyncio.run(get_business_analysis(business_request))
    return run_business_result


def test_run_analysis(run_analysis_result: RunAnalysisResult) -> None:
    df = pd.DataFrame.from_records(run_analysis_result.data)
    assert run_analysis_result.code is not None
    assert len(run_analysis_result.code) > 1
    assert run_analysis_result.data
    assert df.shape[0] > 0
    assert run_analysis_result.status == "success"


def test_run_charts_analysis(run_charts_result: RunChartsResult) -> None:
    assert isinstance(run_charts_result.fig1, go.Figure)
    assert isinstance(run_charts_result.fig2, go.Figure)
    assert len(run_charts_result.code) > 1


def test_run_business_analysis(
    run_business_result: BusinessAnalysisResult,
) -> None:
    assert len(run_business_result.bottom_line) > 1
    assert len(run_business_result.additional_insights) > 1
    assert len(run_business_result.follow_up_questions) > 0
