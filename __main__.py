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
import pathlib

import datarobot as dr
import pulumi
import pulumi_datarobot as datarobot

from infra import (
    settings_app_infra,
    settings_chat_agent,
    settings_credentials,
    settings_generative,
    settings_main,
)
from infra.common.feature_flags import check_feature_flags
from infra.common.globals import GlobalRuntimeEnvironment
from infra.common.urls import get_deployment_url
from infra.components.custom_model_deployment import CustomModelDeployment
from infra.components.dr_credential import (
    get_credential_runtime_parameter_values,
    get_llm_credentials,
)
from infra.components.playground_custom_model import PlaygroundCustomModel
from utils.resources import (
    app_env_name,
    llm_deployment_env_name,
)

check_feature_flags(pathlib.Path("infra/feature_flag_requirements.yaml"))

# Set usecase
if "DATAROBOT_DEFAULT_USE_CASE" in os.environ:
    use_case_id = os.environ["DATAROBOT_DEFAULT_USE_CASE"]
    pulumi.info(f"Using existing use case '{use_case_id}'")
    use_case = datarobot.UseCase.get(
        id=use_case_id,
        resource_name="Data Analyst Use Case [PRE-EXISTING]",
    )
else:
    use_case = datarobot.UseCase(**settings_main.use_case_args)

prediction_environment = datarobot.PredictionEnvironment(
    resource_name=settings_main.prediction_environment_resource_name,
    platform=dr.enums.PredictionEnvironmentPlatform.DATAROBOT_SERVERLESS,
)


llm_credential = get_llm_credentials(settings_generative.LLM)
db_credential = settings_credentials.db_credential

llm_runtime_parameter_values = get_credential_runtime_parameter_values(llm_credential)
db_runtime_parameter_values = get_credential_runtime_parameter_values(db_credential)  # type: ignore[arg-type]


llm_custom_model = PlaygroundCustomModel(
    resource_name=f"Chat Agent Buzok Deployment [{settings_main.project_name}]",
    use_case=use_case,
    playground_args=settings_generative.playground_args,
    llm_blueprint_args=settings_generative.llm_blueprint_args,
    runtime_parameter_values=llm_runtime_parameter_values,
    custom_model_args=settings_generative.custom_model_args,
)


llm_deployment = CustomModelDeployment(
    resource_name=f"Chat Agent Deployment [{settings_main.project_name}]",
    use_case_ids=[use_case.id],
    custom_model_version_id=llm_custom_model.version_id,
    registered_model_args=settings_chat_agent.registered_model_args,
    prediction_environment=prediction_environment,
    deployment_args=settings_chat_agent.deployment_args,
)


app_runtime_parameters = [
    datarobot.ApplicationSourceRuntimeParameterValueArgs(
        key=llm_deployment_env_name,
        type="deployment",
        value=llm_deployment.id,
    ),
] + db_runtime_parameter_values

app_source = datarobot.ApplicationSource(
    files=settings_app_infra.get_app_files(),
    runtime_parameter_values=app_runtime_parameters,  # type: ignore[arg-type]
    base_environment_id=GlobalRuntimeEnvironment.PYTHON_39_STREAMLIT.value.id,
    **settings_app_infra.app_source_args,
)

app_source_version_id = pulumi.Output.all(app_source.id, app_source.version_id).apply(
    lambda args: settings_app_infra.ensure_app_source_settings(*args)
)


app = datarobot.CustomApplication(
    resource_name=settings_app_infra.app_resource_name,
    source_version_id=app_source.version_id,
    use_case_ids=[use_case.id],
)

app.id.apply(settings_app_infra.ensure_app_settings)

# Chat Agent output
pulumi.export(llm_deployment_env_name, llm_deployment.id)
pulumi.export(
    settings_chat_agent.deployment_args.resource_name,
    llm_deployment.id.apply(get_deployment_url),
)

# App output
pulumi.export(app_env_name, app.id)
pulumi.export(
    settings_app_infra.app_resource_name,
    app.application_url,
)
pulumi.export(llm_deployment_env_name, llm_deployment.id)
