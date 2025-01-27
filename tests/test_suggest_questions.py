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

import pandas as pd
import pytest

from utils.api import _validate_question_feasibility, suggest_questions
from utils.schema import AnalystDataset


@pytest.fixture
def sample_datasets() -> list[AnalystDataset]:
    df1 = pd.DataFrame({"sales": [100], "revenue": [200]})
    df2 = pd.DataFrame({"profit": [50], "costs": [150]})

    ds1 = AnalystDataset(name="sales_data", data=df1)
    ds2 = AnalystDataset(name="financial_data", data=df2)
    return [ds1, ds2]


def test_validate_question_feasibility_valid() -> None:
    columns = ["total_sales", "revenue_2023", "profit_margin"]
    question = "What were the total sales in 2023?"

    result = _validate_question_feasibility(question, columns)

    assert result.is_valid
    assert "total_sales" in result.available_columns
    assert "can be answered" in result.validation_message


def test_validate_question_feasibility_invalid() -> None:
    columns = ["revenue", "costs"]
    question = "What is the customer satisfaction score?"

    result = _validate_question_feasibility(question, columns)

    assert not result.is_valid
    assert not result.available_columns
    assert "unavailable data" in result.validation_message


@pytest.mark.asyncio
async def test_suggest_questions_basic(mocker: Any) -> None:
    datasets = [
        AnalystDataset(name="test", data=pd.DataFrame({"col1": [], "col2": []}))
    ]

    mock_completion = mocker.Mock()
    mock_completion.questions = ["How many records in col1?"]
    mocker.patch("instructor.client.Instructor.create", return_value=mock_completion)
    result = await suggest_questions(datasets)

    assert len(result) == 1
    assert result[0].question == "How many records in col1?"
    assert result[0].is_valid


def test_validate_question_case_insensitive() -> None:
    columns = ["Total_Sales", "REVENUE"]
    question = "what were the total sales?"

    result = _validate_question_feasibility(question, columns)

    assert result.is_valid
    assert "total_sales" in [c.lower() for c in result.available_columns]


@pytest.mark.asyncio
async def test_suggest_questions_empty_dataset() -> None:
    datasets = [AnalystDataset(name="test", data=pd.DataFrame())]

    with pytest.raises(ValueError, match="Dictionary DataFrame cannot be empty"):
        await suggest_questions(datasets)
