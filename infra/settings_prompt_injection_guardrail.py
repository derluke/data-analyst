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

import pulumi_datarobot as datarobot
from pydantic import BaseModel

from .common.globals import GlobalGuardrailTemplateName, GlobalRegisteredModelName
from .common.schema import (
    Condition,
    CustomModelGuardConfigurationArgs,
    DeploymentArgs,
    GuardConditionComparator,
    Intervention,
    ModerationAction,
    Stage,
)
from .settings_main import default_prediction_server_id, project_name


class GlobalGuardrail(BaseModel):
    deployment_args: DeploymentArgs
    registered_model_name: GlobalRegisteredModelName
    custom_model_guard_configuration_args: CustomModelGuardConfigurationArgs


prompt_injection_guardrail = GlobalGuardrail(
    deployment_args=DeploymentArgs(
        resource_name=f"Prompt Injection Guard Deployment [{project_name}]",
        label=f"Prompt Injection Guard [{project_name}]",
        predictions_settings=(
            None
            if default_prediction_server_id
            else datarobot.DeploymentPredictionsSettingsArgs(
                min_computes=0, max_computes=1
            )
        ),
    ),
    registered_model_name=GlobalRegisteredModelName.PROMPT_INJECTION,
    custom_model_guard_configuration_args=CustomModelGuardConfigurationArgs(
        name=f"Prompt Injection Guard Configuration [{project_name}]",
        template_name=GlobalGuardrailTemplateName.PROMPT_INJECTION,
        stages=[Stage.PROMPT],
        intervention=Intervention(
            action=ModerationAction.BLOCK,
            condition=Condition(
                comparand=0.7,
                comparator=GuardConditionComparator.GREATER_THAN,
            ).model_dump_json(),
            message="This query triggered our prompt injection guardrail. Please rephrase your question.",
        ),
    ),
)
