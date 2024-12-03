import sys
import time
import traceback
from functools import partial
from typing import Callable, List, Tuple, Type, Union

from openai.types.chat import ChatCompletion
from pandas import DataFrame
from pydantic import BaseModel, field_validator

sys.path.append("..")  # for ease of local development

from dataanalyst.credentials import AzureOpenAICredentials
from dataanalyst.data_model import (
    AnalysisStrategy,
    AnalysisSummary,
    CodeSnippet,
    ColumnSelection,
    DataDictionary,
    GenerationRequest,
    GenerationResponse,
    SQLSnippet,
    StructuredGenerationRequest,
)
from dataanalyst.schema import GeneratorDeploymentSettings


def get_model(credentials, **kwargs):
    import instructor
    from openai import AzureOpenAI

    client = AzureOpenAI(
        api_version=credentials.api_version,
        azure_endpoint=credentials.azure_endpoint,
        api_key=credentials.api_key,
        timeout=kwargs["request_timeout"],
        max_retries=int(kwargs["max_retries"]),
    )
    instructor_client = instructor.from_openai(client)
    create = partial(
        client.chat.completions.create,
        temperature=kwargs["temperature"],
    )
    create_with_completion = partial(
        instructor_client.chat.completions.create_with_completion,
        temperature=kwargs["temperature"],
    )
    return create, create_with_completion


def load_model(input_dir):
    """Prepare code generation chain."""

    generator_deployment_settings = GeneratorDeploymentSettings()
    credentials = AzureOpenAICredentials()
    params = {}
    params["max_retries"] = generator_deployment_settings.max_retries
    params["request_timeout"] = generator_deployment_settings.request_timeout
    params["temperature"] = generator_deployment_settings.temperature
    default_model_name = credentials.azure_deployment
    return get_model(credentials, **params) + (default_model_name,)


def get_response_model(structure: StructuredGenerationRequest) -> Type:
    if structure.type == "code":
        return CodeSnippet
    elif structure.type == "sql":
        return SQLSnippet
    elif structure.type == "data_dict":
        return DataDictionary
    elif structure.type == "analysis_strategy":
        return AnalysisStrategy
    elif structure.type == "analysis_summary":
        return AnalysisSummary
    elif structure.type == "col_selection":
        valid_cols = structure.metadata["valid_columns"]

        class ValidColumnSelection(ColumnSelection):
            @field_validator("analysis_columns", "join_columns")
            @classmethod
            def validate_cols(cls, value: List[str]):
                for col in value:
                    if col not in valid_cols:
                        raise ValueError(f'"{col}" is not a valid column name')
                return value

        return ValidColumnSelection


def update_response(
    resp: GenerationResponse,
    result: Union[ChatCompletion, Tuple[BaseModel, ChatCompletion]],
) -> None:
    if isinstance(result, Tuple):
        structured_response, completion = result
    else:
        completion = result
        structured_response = None

    resp.content = (
        completion.choices[0].message.content or ""
    )  # DR requires a non-None target value
    if resp.completions is None:
        resp.completions = []
    resp.completions.append(completion)
    resp.completion_tokens += completion.usage.completion_tokens
    resp.total_tokens += completion.usage.total_tokens
    resp.prompt_tokens += completion.usage.prompt_tokens
    resp.structured_content = structured_response


def score(data, model, **kwargs):
    """Orchestrate completions, optionally enforcing structured output."""
    create: Callable[..., ChatCompletion]
    create_with_completion: Callable[..., Tuple[BaseModel, ChatCompletion]]

    create, create_with_completion, default_model = model

    responses = []
    for _, row in data.iterrows():
        resp = GenerationResponse()
        now = time.time()
        try:
            req = GenerationRequest(**row.to_dict())
            model = req.model if req.model is not None else default_model
            if req.structure is None or req.structure.two_stages:
                update_response(resp, create(messages=req.messages, model=model))
            if req.structure is not None:
                update_response(
                    resp,
                    create_with_completion(
                        messages=(
                            req.messages
                            if not req.structure.two_stages
                            else req.messages
                            + [{"role": "assistant", "content": resp.content}]
                        ),
                        model=model,
                        response_model=get_response_model(req.structure),
                    ),
                )
        except Exception:
            resp.content = (
                "Encountered an error while generating a response:\n\n"
                + traceback.format_exc()
            )
        resp.elapsed_secs = time.time() - now
        responses.append(resp.model_dump())
    return DataFrame(responses)
