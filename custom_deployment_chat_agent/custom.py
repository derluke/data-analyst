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

import sys

sys.path.append("../")

from collections.abc import Iterator

import pandas as pd
from openai import NotFoundError
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    CompletionCreateParams,
)

from utils.schema import ChatAgentDeploymentSettings


def choose_llm_type(chat_agent_deployment_settings):
    # Try Azure
    try:
        from openai import AzureOpenAI

        from utils.credentials import AzureOpenAICredentials

        credentials = AzureOpenAICredentials()
        client = AzureOpenAI(
            api_version=credentials.api_version,
            azure_endpoint=credentials.azure_endpoint,
            api_key=credentials.api_key,
            max_retries=int(chat_agent_deployment_settings.max_retries),
            timeout=chat_agent_deployment_settings.request_timeout,
        )
        default_model_name = credentials.azure_deployment
        print("running on azure model")
    except ValueError:
        # print(e)
        print("azure failed, running on openai model")
        from openai import OpenAI

        from utils.credentials import OpenAICredentials

        credentials = OpenAICredentials()
        client = OpenAI(
            api_key=credentials.api_key,
            max_retries=int(chat_agent_deployment_settings.max_retries),
            timeout=chat_agent_deployment_settings.request_timeout,
        )
        default_model_name = credentials.deployment

    return client, default_model_name


def load_model(*args, **kwargs) -> tuple[ChatAgentDeploymentSettings, str]:
    chat_agent_deployment_settings = ChatAgentDeploymentSettings()

    client, default_model_name = choose_llm_type(chat_agent_deployment_settings)

    return client, chat_agent_deployment_settings, default_model_name


def score(data: pd.DataFrame, model, **kwargs) -> pd.DataFrame:
    """This is the legacy score hook for
    datarobot version prior to 10.2

    Parameters
    ----------
    data : pd.DataFrame
        Input data. the prompt will be taken from
        the column "promptText" which must exists
    model : OpenAI
        The model object from openai.

    Returns
    -------
    pd.DataFrame
        DataFrame of responses from model
    """
    client, model_settings, default_model_name = model
    prompts = data["promptText"].tolist()
    responses = []

    for prompt in prompts:
        response = client.chat.completions.create(
            messages=[
                {"role": "user", "content": f"{prompt}"},
            ],
            temperature=model_settings.temperature,
            model=default_model_name,
        )
        responses.append(response.choices[0].message.content)

    return pd.DataFrame({"resultText": responses})


def chat(
    completion_create_params: CompletionCreateParams,
    model,
) -> ChatCompletion | Iterator[ChatCompletionChunk]:
    """Chat Hook compatibale with ChatCompletion
    OpenAI Specification

    Parameters
    ----------
    completion_create_params : CompletionCreateParams
        object that holds all the parameters needed to create the chat completion.
    model : OpenAI
        The model object from openai.  this is injected output from load_model

    Returns
    -------
    ChatCompletion
        the completion object with generated choices.
    """
    client, default_model_name = model[0], model[2]
    print(completion_create_params)
    try:
        return client.chat.completions.create(**completion_create_params)
    except (NotFoundError, TypeError):
        print(
            f"{completion_create_params.get('model')} not found, deferring to default"
        )
        completion_create_params["model"] = default_model_name
        return client.chat.completions.create(**completion_create_params)


if __name__ == "__main__":
    model = load_model()
    print(score(pd.DataFrame({"promptText": ["hello"]}), model))
    print(
        chat(
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                ],
                "model": "gpt-4o",
            },
            model=load_model(),
        )
    )
