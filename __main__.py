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
import infra.settings_credentials as credentials
import pulumi
import pulumi_datarobot as datarobot

from application.resources import app_env_name, chat_agent_deployment_env_name

from infra import (
    settings_app_infra,
    settings_chat_agent,
    settings_main,
)
from infra.common.feature_flags import check_feature_flags
from infra.common.globals import GlobalRuntimeEnvironment
from infra.common.urls import get_deployment_url
from infra.components.custom_model_deployment import CustomModelDeployment
from infra.components.dr_credential import DRCredential
from infra.components.rag_custom_model import PlaygroundCustomModel

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

# Set prediction server
if settings_main.default_prediction_server_id is not None:
    prediction_environment = datarobot.PredictionEnvironment.get(
        resource_name=settings_main.prediction_environment_resource_name,
        id=settings_main.default_prediction_server_id,
    )
else:
    prediction_environment = datarobot.PredictionEnvironment(
        resource_name=settings_main.prediction_environment_resource_name,
        platform=dr.enums.PredictionEnvironmentPlatform.DATAROBOT_SERVERLESS,
    )

# Make a credential
credential_resource_provider = settings_main.core.genai_deployment_provider.title()
llm_credential = DRCredential(
    resource_name=f"{credential_resource_provider} LLM Credentials [{settings_main.project_name}]",
    credential=credentials.llm_credential,
    credential_args=credentials.llm_credential_args,
)

if settings_main.core.genai_deployment_type == "diy":
    # Custom model
    chat_agent_custom_model = datarobot.CustomModel(
        files=settings_chat_agent.get_files(
            runtime_parameter_values=llm_credential.runtime_parameter_values,
        ),
        runtime_parameter_values=llm_credential.runtime_parameter_values,
        **settings_chat_agent.custom_model_args.model_dump(
            mode="json", exclude_none=True
        ),
    )

    chat_agent_deployment = CustomModelDeployment(
        resource_name=f"Chat Agent Custom Model Deployment [{settings_main.project_name}]",
        use_case=use_case,
        custom_model_version_id=chat_agent_custom_model.version_id,
        registered_model_args=settings_chat_agent.registered_model_args,
        prediction_environment=prediction_environment,
        deployment_args=settings_chat_agent.deployment_args,
    )
elif settings_main.core.genai_deployment_type == "dr":
    chat_agent_deployment = PlaygroundCustomModel(
        resource_name=f"Chat Agent Buzok Deployment [{settings_main.project_name}]",
        use_case=use_case,
        playground_args=settings_chat_agent.playground_args,
        llm_blueprint_args=settings_chat_agent.llm_blueprint_args,
        runtime_parameter_values=llm_credential.runtime_parameter_values,
        guard_configurations=settings_chat_agent,
        custom_model_args=settings_chat_agent.custom_model_args,
    )
else:
    raise ValueError("GenAI Deployment type must be one of DIY and DR")


app_runtime_parameters = [
    datarobot.ApplicationSourceRuntimeParameterValueArgs(
        key=chat_agent_deployment_env_name,
        type="deployment",
        value=chat_agent_deployment.id,
    )
]

app_source = datarobot.ApplicationSource(
    files=settings_app_infra.get_app_files(),
    runtime_parameter_values=app_runtime_parameters,
    base_environment_id=GlobalRuntimeEnvironment.PYTHON_39_STREAMLIT.value.id,
    **settings_app_infra.app_source_args,
)

app = datarobot.CustomApplication(
    resource_name=settings_app_infra.app_resource_name,
    source_version_id=app_source.version_id,
    use_case_ids=[use_case.id],
)

app.id.apply(settings_app_infra.ensure_app_settings)

# Chat Agent output
pulumi.export(chat_agent_deployment_env_name, chat_agent_deployment.id)
pulumi.export(
    settings_chat_agent.deployment_args.resource_name,
    chat_agent_deployment.id.apply(get_deployment_url),
)

# App output
pulumi.export(app_env_name, app.id)
pulumi.export(
    settings_app_infra.app_resource_name,
    app.application_url,
)
