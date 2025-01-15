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

from __future__ import annotations

from typing import Literal

from utils.schema import AppInfra

from .common.globals import (
    GlobalLLM,
    GlobalPredictionEnvironmentPlatforms,
)
from .common.schema import (
    PredictionEnvironmentArgs,
    UseCaseArgs,
)
from .common.stack import get_stack

project_name = get_stack()

prediction_environment_resource_name = (
    f"Data Analyst Prediction Environment [{project_name}]"
)

prediction_environment_args = PredictionEnvironmentArgs(
    resource_name=f"Data Analyst Prediction Environment [{project_name}]",
    platform=GlobalPredictionEnvironmentPlatforms.DATAROBOT_SERVERLESS,
).model_dump(mode="json", exclude_none=True)

use_case_args = UseCaseArgs(
    resource_name=f"Data Analyst Use Case [{project_name}]",
    description="Use case for Data Analyst application",
).model_dump(exclude_none=True)

LLM = GlobalLLM.AZURE_OPENAI_GPT_4_O

DATABASE_CONNECTION_TYPE: Literal["bigquery", "snowflake"] = "snowflake"

with open("frontend/app_infra.json", "w") as infra_selection:
    infra_selection.write(
        AppInfra(database=DATABASE_CONNECTION_TYPE, llm=LLM.name).model_dump_json()
    )
