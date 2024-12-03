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

import os

from .common.globals import GlobalPredictionEnvironmentPlatforms
from .common.schema import (
    CoreSettings,
    PredictionEnvironmentArgs,
    UseCaseArgs,
)
from .common.stack import get_stack


# Core settings are overridable by environment variables; env values take precedence
core = CoreSettings(
    genai_deployment_type="diy",
    genai_buzok_deployment_type="azure",
)

project_name = get_stack()

default_prediction_server_id = os.getenv("DATAROBOT_PREDICTION_ENVIRONMENT_ID", None)
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
