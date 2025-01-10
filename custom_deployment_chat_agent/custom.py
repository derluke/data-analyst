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

import json
import os
import sys
from typing import Any

sys.path.append("../")

from collections.abc import Iterator

import pandas as pd
from openai import NotFoundError, OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    CompletionCreateParams,
)

from utils.resources import LLMDeployment
from utils.schema import ChatAgentDeploymentSettings


# TODO: Change to any LLM for custom deployment
def choose_llm_type(chat_agent_deployment_settings: Any) -> tuple[OpenAI, str]:
    try:
        deployment_id = json.loads(LLMDeployment().id)["payload"]
        DATAROBOT_ENDPOINT = os.environ.get("DATAROBOT_ENDPOINT")
        DATAROBOT_API_TOKEN = os.environ.get("DATAROBOT_API_TOKEN")

        client = OpenAI(
            api_key=DATAROBOT_API_TOKEN,
            base_url=f"{DATAROBOT_ENDPOINT}/deployments/{deployment_id}",
        )
        default_model_name = "llm-blueprint"
        print("running on azure model")
    except Exception as e:
        raise ValueError(
            f"Failed to generate deployment with base_url {DATAROBOT_ENDPOINT}/deployments/{deployment_id}: {e}"
        ) from e
    return client, default_model_name


def load_model(
    *args: Any, **kwargs: Any
) -> tuple[OpenAI, ChatAgentDeploymentSettings, str]:
    chat_agent_deployment_settings = ChatAgentDeploymentSettings()

    client, default_model_name = choose_llm_type(chat_agent_deployment_settings)

    return client, chat_agent_deployment_settings, default_model_name


def score(
    data: pd.DataFrame,
    model: tuple[OpenAI, ChatAgentDeploymentSettings, str],
    **kwargs: Any,
) -> pd.DataFrame:
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
    model: str,
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
