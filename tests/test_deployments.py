import base64
import json

import datarobot as dr
import pandas as pd
import plotly.express as px
import pytest
from dataanalyst.data_model import (
    GenerationRequest,
    GenerationResponse,
    PySandboxRequest,
    PySandboxResponse,
    StructuredGenerationRequest,
)


@pytest.fixture
def generator_deployment(generator_deployment_id):
    return dr.Deployment.get(generator_deployment_id)


@pytest.fixture
def base64_png():
    fig = px.bar(x=["a", "b", "c"], y=[1, 3, 2])
    png_bytes = fig.to_image(format="png", width=300)
    return base64.b64encode(png_bytes).decode("utf-8")


@pytest.fixture(scope="class")
def diabetes_dataset_url():
    return "https://s3.amazonaws.com/datarobot_public_datasets/10k_diabetes_20.csv"


class TestGenerator:
    def test_code_generation(self, generator_deployment, get_response):
        messages = json.dumps([{"role": "user", "content": "print hello world"}])
        resp = GenerationResponse(
            **get_response(
                GenerationRequest(
                    messages=messages,
                    structure=StructuredGenerationRequest(
                        type="code"
                    ).model_dump_json(),
                ),
                generator_deployment,
            )
        )

        assert "print" in resp.structured_content.python_code
        assert "hello" in resp.structured_content.python_code.lower()
        assert resp.completions is not None
        assert (
            resp.total_tokens > 0
            and resp.prompt_tokens > 0
            and resp.completion_tokens > 0
        )

    def test_image_prompt(self, generator_deployment, get_response, base64_png):
        messages = json.dumps(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_png}"
                            },
                        },
                    ],
                }
            ]
        )
        resp = GenerationResponse(
            **get_response(
                GenerationRequest(messages=messages, model="gpt-4o"),
                generator_deployment,
            )
        )

        assert resp.content
        assert resp.completions is not None
        assert (
            resp.total_tokens > 0
            and resp.prompt_tokens > 0
            and resp.completion_tokens > 0
        )

    def test_unstructured_generation(self, generator_deployment, get_response):
        for model in [None, "gpt-35-turbo-16k"]:
            messages = json.dumps(
                [{"role": "user", "content": "what is the capital of Germany"}]
            )
            resp = GenerationResponse(
                **get_response(
                    GenerationRequest(
                        messages=messages,
                        model=model,
                    ),
                    generator_deployment,
                )
            )

            assert "berlin" in resp.content.lower()
            assert resp.completions is not None
            assert (
                resp.total_tokens > 0
                and resp.prompt_tokens > 0
                and resp.completion_tokens > 0
            )

    def test_analysis_strategy_generation(self, generator_deployment, get_response):
        messages = json.dumps(
            [
                {
                    "role": "user",
                    "content": "what is the relationship between the variables 'x' and 'y'?",
                }
            ]
        )
        for two_stages in [True, False]:
            resp = GenerationResponse(
                **get_response(
                    GenerationRequest(
                        messages=messages,
                        structure=StructuredGenerationRequest(
                            type="analysis_strategy",
                            two_stages=two_stages,
                        ).model_dump_json(),
                    ),
                    generator_deployment,
                )
            )

            assert len(resp.structured_content.analyses)
            assert (
                resp.total_tokens > 0
                and resp.prompt_tokens > 0
                and resp.completion_tokens > 0
            )

    def test_data_dict_generation(self, generator_deployment, get_response):
        messages = json.dumps(
            [
                {
                    "role": "user",
                    "content": """\
|    |   age | location   |
|---:|------:|:-----------|
|  0 |    25 | MN         |
|  1 |    30 | CA         |

Write a data dictionary for the columns in the dataset.
""",
                }
            ]
        )
        resp = GenerationResponse(
            **get_response(
                GenerationRequest(
                    messages=messages,
                    structure=StructuredGenerationRequest(
                        type="data_dict",
                    ).model_dump_json(),
                ),
                generator_deployment,
            )
        )

        assert len(resp.structured_content.dictionary)
        assert (
            resp.total_tokens > 0
            and resp.prompt_tokens > 0
            and resp.completion_tokens > 0
        )

    def test_col_selection_generation(
        self, generator_deployment, get_response, diabetes_dataset_url
    ):
        df = pd.read_csv(diabetes_dataset_url)
        cols = "Columns:\n" + "\n".join(df.columns.to_list())
        messages = json.dumps(
            [
                {
                    "role": "user",
                    "content": f"What columns are required to count total readmissions?\n{cols}",
                }
            ]
        )
        resp = GenerationResponse(
            **get_response(
                GenerationRequest(
                    messages=messages,
                    structure=StructuredGenerationRequest(
                        type="col_selection",
                        metadata={"valid_columns": df.columns.to_list()},
                    ).model_dump_json(),
                ),
                generator_deployment,
            )
        )

        assert len(resp.structured_content.analysis_columns)
        assert (
            resp.total_tokens > 0
            and resp.prompt_tokens > 0
            and resp.completion_tokens > 0
        )

    def test_summary_generation(self, generator_deployment, get_response):
        messages = json.dumps(
            [
                {
                    "role": "user",
                    "content": "Older patients on average have higher readmission likelihood.",
                }
            ]
        )
        resp = GenerationResponse(
            **get_response(
                GenerationRequest(
                    messages=messages,
                    structure=StructuredGenerationRequest(
                        type="analysis_summary",
                    ).model_dump_json(),
                ),
                generator_deployment,
            )
        )

        assert len(resp.structured_content.summary)
        assert len(resp.structured_content.next_questions)
        assert (
            resp.total_tokens > 0
            and resp.prompt_tokens > 0
            and resp.completion_tokens > 0
        )
