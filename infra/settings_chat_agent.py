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
import textwrap

import datarobot as dr
import pulumi_datarobot as datarobot
from jinja2 import BaseLoader, Environment

from application.schema import ChatAgentDeploymentSettings
from .common.globals import GlobalRuntimeEnvironment, GlobalLLM

from .common.schema import (
    CustomModelArgs,
    DeploymentArgs,
    LLMBlueprintArgs,
    LLMSettings,
    PlaygroundArgs,
    RegisteredModelArgs,
)
from infra.settings_main import core

from .settings_main import (
    default_prediction_server_id,
    project_name,
)


custom_model_args = CustomModelArgs(
    resource_name=f"Chat Agent Custom Model [{project_name}]",
    name=f"Chat Agent Custom Model [{project_name}]",
    base_environment_id=GlobalRuntimeEnvironment.PYTHON_311_MODERATIONS.value.id,
    target_name=ChatAgentDeploymentSettings().target_feature_name,
    target_type=dr.enums.TARGET_TYPE.TEXT_GENERATION,
    replicas=2,
)


registered_model_args = RegisteredModelArgs(
    resource_name=f"Chat Agent Registered Model [{project_name}]",
)

deployment_args = DeploymentArgs(
    resource_name=f"Chat Agent Deployment [{project_name}]",
    label=f"Chat Agent Deployment [{project_name}]",
    predictions_settings=(
        None
        if default_prediction_server_id
        else datarobot.DeploymentPredictionsSettingsArgs(min_computes=0, max_computes=1)
    ),
    predictions_data_collection_settings=datarobot.DeploymentPredictionsDataCollectionSettingsArgs(
        enabled=True,
    ),
)


if core.genai_deployment_type == "dr":
    # if providing DIY RAG logic, these settings are N/A
    playground_args = PlaygroundArgs(
        resource_name=f"Data Analyst Playground [{project_name}]",
    )

    llm_blueprint_args = LLMBlueprintArgs(
        resource_name=f"Data Analyst LLM Blueprint [{project_name}]",
        llm_id=core.genai_deployment_name_buzok,
        llm_settings=LLMSettings(
            max_completion_length=512,
            system_prompt=textwrap.dedent(
                """\
                Use the following pieces of context to answer the user's question.
                If you don't know the answer, just say that you don't know, don't try to make up an answer.
                ----------------
                {context}"""
            ),
        ),
    )

elif core.genai_deployment_type == "diy":
    chat_agent_deployment_path = pathlib.Path("custom_deployment_chat_agent/")

    def get_files(
        runtime_parameter_values: list[datarobot.CustomModelRuntimeParameterValueArgs],
    ) -> list[tuple[str, str]]:
        llm_runtime_parameter_specs = "\n".join(
            [
                textwrap.dedent(
                    f"""\
                - fieldName: {param.key}
                  type: {param.type}"""
                )
                for param in runtime_parameter_values
            ]
        )

        with open(chat_agent_deployment_path / "model-metadata.yaml.jinja") as f:
            template = Environment(loader=BaseLoader()).from_string(f.read())
        with open(chat_agent_deployment_path / "model-metadata.yaml", "w") as f:
            runtime_parameters = template.render(
                custom_model_name=custom_model_args.name,
                target_type=custom_model_args.target_type,
                runtime_parameters=llm_runtime_parameter_specs,
            )
            f.write(runtime_parameters)

        files = [
            (str(f), str(f.relative_to(chat_agent_deployment_path)))
            for f in chat_agent_deployment_path.glob("**/*")
            if f.is_file() and f.name not in ("README.md", "model-metadata.yaml.jinja")
        ] + [
            (
                "application/__init__.py",
                "application/__init__.py",
            ),
            ("application/schema.py", "application/schema.py"),
            ("application/credentials.py", "application/credentials.py"),
        ]
        return files
