import sys

sys.path.append("../")

from openai import AzureOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    CompletionCreateParams,
)
import pandas as pd
from collections.abc import Iterator

from application.credentials import AzureOpenAICredentials
from application.schema import ChatAgentDeploymentSettings


# TODO: Custom should work with multiple model types
def load_model(*args, **kwargs) -> tuple[AzureOpenAI, ChatAgentDeploymentSettings, str]:
    from openai import AzureOpenAI

    credentials = AzureOpenAICredentials()
    chat_agent_deployment_settings = ChatAgentDeploymentSettings()

    client = AzureOpenAI(
        api_version=credentials.api_version,
        azure_endpoint=credentials.azure_endpoint,
        api_key=credentials.api_key,
        max_retries=int(chat_agent_deployment_settings.max_retries),
        timeout=chat_agent_deployment_settings.request_timeout,
    )
    default_model_name = credentials.azure_deployment
    return client, chat_agent_deployment_settings, default_model_name


def score(
    data: pd.DataFrame, model: tuple[AzureOpenAI, ChatAgentDeploymentSettings], **kwargs
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
    model: tuple[AzureOpenAI, ChatAgentDeploymentSettings],
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
    client = model[0]
    if "model" not in completion_create_params:
        completion_create_params["model"] = model[2]
    return client.chat.completions.create(**completion_create_params)


if __name__ == "__main__":

    model = load_model()
    print(score(pd.DataFrame({"promptText": ["hello"]}), model))
    print(
        chat(
            {
                "model": AzureOpenAICredentials().azure_deployment,
                "messages": [
                    {"role": "user", "content": "hello"},
                ],
                "temperature": ChatAgentDeploymentSettings().temperature,
            },
            model,
        )
    )
