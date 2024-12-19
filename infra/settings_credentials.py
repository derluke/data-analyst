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

from pydantic import BaseModel, ValidationError

from utils.credentials import (
    AzureOpenAICredentials,
    GoogleLLMCredentials,
    OpenAICredentials,
    SnowflakeCredentials,
)

from .common.schema import CredentialArgs
from .settings_main import core, project_name


def set_credential(credential_type: BaseModel) -> BaseModel:
    try:
        credential = credential_type()
        credential.test()
    except ValidationError as e:
        raise ValueError(
            "Unable to load credentials. "
            "Verify you have setup your environment variables as described in README.md."
        ) from e
    return credential


llm_credential_args = CredentialArgs(
    resource_name=f"Data Analyst LLM Credential [{project_name}]",
)

if core.genai_deployment_provider == "azure":
    llm_credential = set_credential(AzureOpenAICredentials)
elif core.genai_deployment_provider == "google":
    llm_credential = set_credential(GoogleLLMCredentials)
elif core.genai_deployment_provider == "openai":
    llm_credential = set_credential(OpenAICredentials)
else:
    raise NotImplementedError(
        "Only Azure and Google LLM credentials are currently supported."
    )

db_credential = SnowflakeCredentials()
db_credential_args = CredentialArgs(
    resource_name=f"Data Analyst Snowflake Credential [{project_name}]",
)
