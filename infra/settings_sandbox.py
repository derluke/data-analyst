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
import pulumi_datarobot as datarobot
from jinja2 import BaseLoader, Environment

from dataanalyst.schema import SandboxDeploymentSettings
from infra.common.globals import GlobalRuntimeEnvironment
from infra.common.schema import (
    CustomModelArgs,
    DeploymentArgs,
    RegisteredModelArgs,
)

from .settings_main import (
    default_prediction_server_id,
    project_name,
)

sandbox_deployment_path = pathlib.Path("deployment_sandbox/")

custom_model_args = CustomModelArgs(
    resource_name=f"Sandbox Custom Model [{project_name}]",
    name=f"Sandbox Custom Model [{project_name}]",
    base_environment_id=GlobalRuntimeEnvironment.PYTHON_39_SCIKIT_LEARN.value.id,
    target_name=SandboxDeploymentSettings().target_feature_name,
    target_type=dr.enums.TARGET_TYPE.TEXT_GENERATION,
    replicas=2,
    network_access="NONE",
)


registered_model_args = RegisteredModelArgs(
    resource_name=f"Sandbox Registered Model [{project_name}]",
)

deployment_args = DeploymentArgs(
    resource_name=f"Sandbox Deployment [{project_name}]",
    label=f"Sandbox Deployment [{project_name}]",
    predictions_settings=(
        None
        if default_prediction_server_id
        else datarobot.DeploymentPredictionsSettingsArgs(min_computes=0, max_computes=1)
    ),
    predictions_data_collection_settings=datarobot.DeploymentPredictionsDataCollectionSettingsArgs(
        enabled=True,
    ),
)


def get_files() -> list[tuple[str, str]]:
    with open(sandbox_deployment_path / "model-metadata.yaml.jinja") as f:
        template = Environment(loader=BaseLoader()).from_string(f.read())
    with open(sandbox_deployment_path / "model-metadata.yaml", "w") as f:
        runtime_parameters = template.render(
            custom_model_name=custom_model_args.name,
            target_type=custom_model_args.target_type,
        )
        f.write(runtime_parameters)

    files = [
        (str(f), str(f.relative_to(sandbox_deployment_path)))
        for f in sandbox_deployment_path.glob("**/*")
        if f.is_file() and f.name != "model-metadata.yaml.jinja"
    ] + [
        (
            "dataanalyst/__init__.py",
            "dataanalyst/__init__.py",
        ),
        ("dataanalyst/schema.py", "dataanalyst/schema.py"),
        ("dataanalyst/data_model.py", "dataanalyst/data_model.py"),
    ]
    return files
