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

import pathlib

import datarobot as dr
import infra.settings_credentials as credentials
import pulumi
import pulumi_datarobot as datarobot
from dataanalyst.resources import (
    app_env_name,
    generator_deployment_env_name,
    sandbox_deployment_env_name,
)
from infra import (
    settings_app_infra,
    settings_generator,
    settings_main,
    settings_sandbox,
)
from infra.common.feature_flags import check_feature_flags
from infra.common.globals import GlobalRuntimeEnvironment
from infra.common.urls import get_deployment_url
from infra.components.custom_model_deployment import CustomModelDeployment
from infra.components.dr_credential import DRCredential

check_feature_flags(pathlib.Path("infra/feature_flag_requirements.yaml"))


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


llm_credential = DRCredential(
    resource_name=f"Azure LLM Credentials [{settings_main.project_name}]",
    credential=credentials.llm_credential,
    credential_args=credentials.llm_credential_args,
)

db_credential = DRCredential(
    resource_name=f"Database Credentials [{settings_main.project_name}]",
    credential=credentials.db_credential,
    credential_args=credentials.db_credential_args,
)


generator_custom_model = datarobot.CustomModel(
    files=settings_generator.get_files(
        runtime_parameter_values=llm_credential.runtime_parameter_values,
    ),
    runtime_parameter_values=llm_credential.runtime_parameter_values,
    **settings_generator.custom_model_args.model_dump(mode="json", exclude_none=True),
)


generator_deployment = CustomModelDeployment(
    resource_name=f"Generator Custom Model Deployment [{settings_main.project_name}]",
    custom_model_version_id=generator_custom_model.version_id,
    registered_model_args=settings_generator.registered_model_args,
    prediction_environment=prediction_environment,
    deployment_args=settings_generator.deployment_args,
)


sandbox_custom_model = datarobot.CustomModel(
    files=settings_sandbox.get_files(),
    **settings_sandbox.custom_model_args.model_dump(mode="json", exclude_none=True),
)


sandbox_deployment = CustomModelDeployment(
    resource_name=f"Sandbox Custom Model Deployment [{settings_main.project_name}]",
    custom_model_version_id=sandbox_custom_model.version_id,
    registered_model_args=settings_sandbox.registered_model_args,
    prediction_environment=prediction_environment,
    deployment_args=settings_sandbox.deployment_args,
)


app_runtime_parameters = [
    datarobot.ApplicationSourceRuntimeParameterValueArgs(
        key=generator_deployment_env_name,
        type="deployment",
        value=generator_deployment.id,
    ),
    datarobot.ApplicationSourceRuntimeParameterValueArgs(
        key=sandbox_deployment_env_name,
        type="deployment",
        value=sandbox_deployment.id,
    ),
] + db_credential.app_runtime_parameter_values

app_source = datarobot.ApplicationSource(
    files=settings_app_infra.get_app_files(
        runtime_parameter_values=app_runtime_parameters
    ),
    runtime_parameter_values=app_runtime_parameters,
    base_environment_id=GlobalRuntimeEnvironment.PYTHON_39_STREAMLIT.value.id,
    **settings_app_infra.app_source_args,
)

app = datarobot.CustomApplication(
    resource_name=settings_app_infra.app_resource_name,
    source_version_id=app_source.version_id,
)

app.id.apply(settings_app_infra.ensure_app_settings)

# Generator output
pulumi.export(generator_deployment_env_name, generator_deployment.id)
pulumi.export(
    settings_generator.deployment_args.resource_name,
    generator_deployment.id.apply(get_deployment_url),
)

# Sandbox output
pulumi.export(sandbox_deployment_env_name, sandbox_deployment.id)
pulumi.export(
    settings_sandbox.deployment_args.resource_name,
    sandbox_deployment.id.apply(get_deployment_url),
)

# App output
pulumi.export(app_env_name, app.id)
pulumi.export(
    settings_app_infra.app_resource_name,
    app.application_url,
)
