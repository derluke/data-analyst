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

from __future__ import annotations

import ast
import asyncio
import base64
import functools
import io
import json
import logging
import re
import sys
import tempfile
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from types import FunctionType
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Sequence,
    Tuple,
    TypeVar,
    cast,
)

import datarobot as dr
import instructor
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import psutil
from joblib import Memory
from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)
from plotly.subplots import make_subplots
from pydantic import ValidationError

sys.path.append("..")

from utils import prompts
from utils.database_helpers import Database
from utils.datetime_helpers import convert_datetime_series, is_date_column
from utils.resources import LLMDeployment
from utils.schema import (
    AiCatalogDataset,
    AnalysisError,
    AnalystDataset,
    BusinessAnalysisGeneration,
    BusinessAnalysisMetadata,
    BusinessAnalysisRequest,
    BusinessAnalysisResult,
    ChartCodeHistory,
    ChartExecutionError,
    ChartGenerationMetadata,
    ChartGenerationResult,
    ChartPerformance,
    ChartValidationError,
    ChatRequest,
    CleansedDataset,
    CleansingReport,
    CodeGeneration,
    CodeValidator,
    DatabaseAnalysisCodeGeneration,
    DatabaseAnalysisMetadata,
    DatabaseAnalysisRequest,
    DatabaseAnalysisResult,
    DataDictionary,
    DataDictionaryColumn,
    DictionaryGeneration,
    EnhancedQuestionGeneration,
    MemoryUsage,
    QuestionListGeneration,
    RunAnalysisRequest,
    RunAnalysisResult,
    RunAnanlysisResultMetadata,
    RunChartsRequest,
    RunChartsResult,
    RunChartsResultMetadata,
    ValidatedQuestion,
    ValidationMessage,
)

logger = logging.getLogger("DataAnalystFrontend")

try:
    dr_client = dr.Client()  # type: ignore[attr-defined]
    chat_agent_deployment_id = LLMDeployment().id
    deployment_chat_base_url = (
        dr_client.endpoint + f"/deployments/{chat_agent_deployment_id}/"
    )

    openai_client = OpenAI(
        api_key=dr_client.token,
        base_url=deployment_chat_base_url,
        timeout=90,
        max_retries=2,
    )
    client = instructor.from_openai(openai_client, mode=instructor.Mode.MD_JSON)


except ValidationError as e:
    raise ValueError(
        "Unable to load Deployment ID."
        "If running locally, verify you have selected the correct "
        "stack and that it is active using `pulumi stack output`. "
        "If running in DataRobot, verify your runtime parameters have been set correctly."
    ) from e

ALTERNATIVE_LLM_BIG = "gpt-4o"
ALTERNATIVE_LLM_SMALL = "gpt-4o-mini"
DICTIONARY_BATCH_SIZE = 10
MAX_AI_CATALOG_DATASET_SIZE = 400e6  # aligns to 400MB set in streamlit config.toml
DISK_CACHE_LIMIT_BYTES = 512e6

_memory = Memory(tempfile.gettempdir(), verbose=0)
_memory.clear(warn=False)  # clear cache on startup

T = TypeVar("T")


def cache(f: T) -> T:
    """Cache function and coroutine results to disk using joblib."""
    cached_f = _memory.cache(f)

    if asyncio.iscoroutinefunction(f):

        async def awrapper(*args: Any, **kwargs: Any) -> Any:
            in_cache = cached_f.check_call_in_cache(*args, **kwargs)
            result = await cached_f(*args, **kwargs)
            if not in_cache:
                _memory.reduce_size(DISK_CACHE_LIMIT_BYTES)
            else:
                logger.info(
                    f"Using previously cached result for function `{f.__name__}`"
                )
            return result

        return cast(T, awrapper)
    else:

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            in_cache = cached_f.check_call_in_cache(*args, **kwargs)
            result = cached_f(*args, **kwargs)
            if not in_cache:
                _memory.reduce_size(DISK_CACHE_LIMIT_BYTES)
            else:
                logger.info(
                    f"Using previously cached result for function `{f.__name__}`"  # type: ignore
                )
            return result

        return cast(T, wrapper)


class InvalidGeneratedCode(Exception):
    """Raised when LLM generated code is found to be invalid."""

    def __init__(
        self, *args: Any, code: str | None = None, exception: Exception | None = None
    ):
        super().__init__(*args)
        self.code = code
        self.exception = exception


class MaxReflectionAttempts(Exception):
    """Raised after final attempt to self-correct LLM code generation"""

    def __init__(
        self, *args: Any, exception_history: list[InvalidGeneratedCode] | None = None
    ):
        super().__init__(*args)
        self.exception_history = exception_history


U = TypeVar("U")


def reflect_code_generation_errors(
    max_attempts: int,
) -> Callable[[Callable[..., Awaitable[U]]], Callable[..., Awaitable[U]]]:
    """Reflect LLM code generation errors for self-correction

    Exceptions raised by invalid code will be injected back into the
    decorated function via the `exception_history` keyword argument.

    `exception_history` contains a list of InvalidGeneratedCode
    exceptions.
    """

    def _outer_wrapper(
        f: Callable[..., Awaitable[U]],
    ) -> Callable[..., Awaitable[U]]:
        @functools.wraps(f)
        async def _inner_wrapper(*args: Any, **kwargs: Any) -> U:
            attempts = 1
            exception_history: list[InvalidGeneratedCode] = []
            kwargs["exception_history"] = exception_history
            while attempts <= max_attempts:
                try:
                    return await f(*args, **kwargs)
                except InvalidGeneratedCode as e:
                    msg = type(e.exception).__name__ + f": {str(e.exception)}"
                    logger.info(
                        f"LLM generated code raised {msg}\nGenerated code:\n{e.code}"
                    )
                    exception_history.append(e)
                attempts += 1

            msg = f"{f.__name__} failed to generate valid code after {max_attempts} attempts"
            logger.error(msg)
            raise MaxReflectionAttempts(msg, exception_history=exception_history)

        return _inner_wrapper

    return _outer_wrapper


# This can be large as we are not storing the actual datasets in memory, just metadata
@functools.lru_cache(maxsize=32)
def list_catalog_datasets(limit: int = 100) -> List[AiCatalogDataset]:
    """
    Fetch datasets from AI Catalog with specified limit

    Args:
        limit: int
        Datasets to retrieve. Max value: 100
    """

    url = f"datasets?limit={limit}"

    # Get all datasets and manually limit the results
    datasets = dr.client.get_client().get(url).json()["data"]

    return [
        AiCatalogDataset(
            id=ds["datasetId"],
            name=ds["name"],
            created=(
                ds["creationDate"][:10] if "creationDate" in ds else "N/A"  # %Y-%m-%d
            ),
            size=(
                f"{ds['datasetSize'] / (1024 * 1024):.1f} MB"
                if "datasetSize" in ds
                else "N/A"
            ),
        )
        for ds in datasets
    ]


@cache
def download_catalog_datasets(*args: Any) -> list[AnalystDataset]:
    """Load selected datasets as pandas DataFrames

    Args:
        *args: list of dataset IDs to download

    Returns:
        list[DatasetInput]: Dictionary of dataset names and data
    """
    dataset_ids = list(args)
    datasets = [dr.Dataset.get(id_) for id_ in dataset_ids]  # type: ignore
    if (
        sum([ds.size for ds in datasets if ds.size is not None])
        > MAX_AI_CATALOG_DATASET_SIZE
    ):
        raise ValueError(
            f"The requested AI Catalog datasets must total <= {int(MAX_AI_CATALOG_DATASET_SIZE)} bytes"
        )

    result_datasets: list[AnalystDataset] = []
    for dataset in datasets:
        try:
            df_records = cast(
                list[dict[str, Any]],
                dataset.get_as_dataframe().to_dict(orient="records"),
            )
            result_datasets.append(AnalystDataset(name=dataset.name, data=df_records))
            logger.info(f"Successfully downloaded {dataset.name}")
        except Exception as e:
            logger.error(f"Failed to read dataset {dataset.name}: {str(e)}")
            continue
    return result_datasets


async def _get_dictionary_batch(
    columns: list[str], df: pd.DataFrame, batch_size: int = 5
) -> list[DataDictionaryColumn]:
    """Process a batch of columns to get their descriptions"""

    # Get sample data and stats for just these columns
    # Convert timestamps to ISO format strings for JSON serialization
    sample_data = {}
    for col in columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            # Convert timestamps to ISO format strings
            sample_data[col] = (
                df[col]
                .head(10)
                .apply(lambda x: x.isoformat() if pd.notnull(x) else None)
                .to_dict()
            )
        else:
            sample_data[col] = df[col].head(10).to_dict()

    # Handle numeric summary
    numeric_summary = {}
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            desc = df[col].describe()
            numeric_summary[col] = {
                k: float(v) if pd.notnull(v) else None
                for k, v in desc.to_dict().items()
            }

    # Get categories for non-numeric columns
    categories = []
    for column in columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            try:
                value_counts = df[column].value_counts().head(10)
                # Convert any timestamp values to strings
                if pd.api.types.is_datetime64_any_dtype(df[column]):
                    value_counts.index = value_counts.index.map(
                        lambda x: x.isoformat() if pd.notnull(x) else None
                    )
                categories.append({column: list(value_counts.keys())})
            except Exception:
                continue

    # Create messages for OpenAI
    messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(
            role="system", content=prompts.SYSTEM_PROMPT_GET_DICTIONARY
        ),
        ChatCompletionUserMessageParam(
            role="user", content=f"Data: {json.dumps(sample_data)}"
        ),
        ChatCompletionUserMessageParam(
            role="user", content=f"Statistical Summary: {json.dumps(numeric_summary)}"
        ),
    ]

    if categories:
        messages.append(
            ChatCompletionUserMessageParam(
                role="user", content=f"Categorical Values: {json.dumps(categories)}"
            )
        )

    # Get descriptions from OpenAI
    completion: DictionaryGeneration = client.chat.completions.create(
        response_model=DictionaryGeneration,
        model=ALTERNATIVE_LLM_SMALL,
        messages=messages,
    )

    try:
        # Convert to dictionary format
        descriptions = completion.to_dict()

        # Only return descriptions for requested columns
        return [
            DataDictionaryColumn(
                column=col,
                description=descriptions.get(col, "No description available"),
                data_type=str(df[col].dtype),
            )
            for col in columns
        ]

    except ValueError as e:
        logger.error(f"Invalid dictionary response: {str(e)}")
        return [
            DataDictionaryColumn(
                column=col,
                description="No valid description available",
                data_type=str(df[col].dtype),
            )
            for col in columns
        ]


async def _get_dictionary(dataset: AnalystDataset) -> DataDictionary:
    """Process a single dataset with parallel column batch processing"""
    try:
        # Convert JSON to DataFrame
        df = dataset.to_df()

        # Add debug logging
        logger.info(f"Processing dataset {dataset.name} with shape {df.shape}")

        # Handle empty dataset
        if df.empty:
            logger.warning(f"Dataset {dataset.name} is empty")
            return DataDictionary(
                name=dataset.name,
                dictionary=[],
            )

        # Split columns into batches
        column_batches = [
            list(df.columns[i : i + DICTIONARY_BATCH_SIZE])
            for i in range(0, len(df.columns), DICTIONARY_BATCH_SIZE)
        ]
        logger.info(
            f"Created {len(column_batches)} batches for {len(df.columns)} columns"
        )

        tasks = [
            _get_dictionary_batch(batch, df, DICTIONARY_BATCH_SIZE)
            for batch in column_batches
        ]

        results = await asyncio.gather(*tasks)
        dictionary = sum(results, [])

        logger.info(
            f"Created dictionary with {len(dictionary)} entries for dataset {dataset.name}"
        )

        return DataDictionary(
            name=dataset.name,
            dictionary=dictionary,
        )

    except Exception as e:
        raise Exception(f"Error processing dataset {dataset.name}: {str(e)}")


# Add memory management helper
def _get_memory_usage() -> MemoryUsage:
    """Get current memory usage statistics"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return MemoryUsage(
        rss=memory_info.rss / 1024 / 1024,  # RSS in MB
        vms=memory_info.vms / 1024 / 1024,  # VMS in MB
        percent=process.memory_percent(),
    )


def _validate_question_feasibility(
    question: str, available_columns: List[str]
) -> ValidatedQuestion:
    """Validate if a question can be answered with available data

    Checks if common data elements mentioned in the question exist in columns
    """
    # Convert question and columns to lowercase for matching
    question_lower = question.lower()
    columns_lower = [col.lower() for col in available_columns]

    # Extract potential column references from question
    words = set(re.findall(r"\b\w+\b", question_lower))

    # Find matches and missing terms
    found_columns = [col for col in columns_lower if any(word in col for word in words)]
    missing_columns = [
        word for word in words if any(word in col for col in columns_lower)
    ]

    is_valid = len(found_columns) > 0
    message = (
        "Question can be answered with available data"
        if is_valid
        else "Question may require unavailable data"
    )

    return ValidatedQuestion(
        question=question,
        is_valid=is_valid,
        available_columns=found_columns,
        missing_columns=missing_columns,
        validation_message=message,
    )


async def suggest_questions(
    datasets: list[AnalystDataset], max_columns: int = 40
) -> list[ValidatedQuestion]:
    """Generate and validate suggested analysis questions

    Args:
        dictionary: DataFrame containing data dictionary
        max_columns: Maximum number of columns to include in prompt

    Returns:
        Dict containing:
            - questions: List of validated question objects
            - metadata: Dictionary of processing information
    """
    # Validate input
    dictionary = sum(
        [
            DataDictionary.from_df(
                ds.to_df(),
                column_descriptions=f"Column from dataset {ds.name}",
            ).dictionary
            for ds in datasets
        ],
        [],
    )

    if len(dictionary) < 1:
        raise ValueError("Dictionary DataFrame cannot be empty")

    # Limit columns for OpenAI prompt
    total_columns = len(dictionary)
    if total_columns > max_columns:
        # Take first and last 20 columns
        half_max = max_columns // 2
        first_half = dictionary[:half_max]
        last_half = dictionary[-half_max:]

        # Remove any duplicates
        dictionary = first_half + last_half

        # deduplicate
        dictionary = list({item.column: item for item in dictionary}.values())

    # Convert dictionary to format expected by OpenAI
    dict_data = {
        "columns": [d.column for d in dictionary],
        "descriptions": [d.description for d in dictionary],
        "data_types": [d.data_type for d in dictionary],
    }

    # Create OpenAI messages
    messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(
            role="system", content=prompts.SYSTEM_PROMPT_SUGGEST_A_QUESTION
        ),
        ChatCompletionUserMessageParam(
            role="user", content=f"Data Dictionary:\n{json.dumps(dict_data)}"
        ),
    ]

    completion: QuestionListGeneration = client.chat.completions.create(
        response_model=QuestionListGeneration,
        model=ALTERNATIVE_LLM_SMALL,
        messages=messages,
    )

    available_columns = dict_data["columns"]
    validated_questions: list[ValidatedQuestion] = []

    for question in completion.questions:
        validated_questions.append(
            _validate_question_feasibility(question, available_columns)
        )

    return validated_questions


# TODO: duplicated in schema
def _validate_chart_code(code: str) -> Tuple[bool, str]:
    """Validate chart generation code for safety and correctness"""
    try:
        tree = ast.parse(code)
        imports: list[str] = []

        # Check imports
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    imports.extend(n.name.split(".")[0] for n in node.names)
                elif node.module is not None:
                    imports.append(node.module.split(".")[0])

        allowed_modules = {"pandas", "numpy", "plotly", "scipy"}
        illegal_imports = set(imports) - allowed_modules
        if illegal_imports:
            return False, f"Illegal imports detected: {illegal_imports}"

        # Verify create_charts function exists
        has_create_charts = any(
            isinstance(node, ast.FunctionDef) and node.name == "create_charts"
            for node in ast.walk(tree)
        )
        if not has_create_charts:
            return False, "Missing create_charts function"

        return True, "Validation passed"

    except SyntaxError as e:
        return False, f"Syntax error in code: {str(e)}"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def _figure_to_base64(fig: go.Figure) -> str | None:
    """Convert Plotly figure to base64 encoded PNG"""
    try:
        if not isinstance(fig, go.Figure):
            raise ValueError(f"Expected plotly.graph_objects.Figure, got {type(fig)}")
        img_bytes = fig.to_image(format="png")
        return base64.b64encode(img_bytes).decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to convert figure to base64: {str(e)}")
        return None


async def _create_charts(
    df: pd.DataFrame,
    question: str,
    metadata: Dict[str, Any],
    error_message: str | None = None,
    failed_code: str | None = None,
    max_attempts: int = 3,
) -> ChartGenerationResult:
    """Generate and validate chart code with retry logic"""
    attempts = 0
    validation_errors: list[ChartValidationError] = []
    execution_errors: list[ChartExecutionError] = []
    code_history: list[ChartCodeHistory] = []

    while attempts < max_attempts:
        attempts += 1

        try:
            # Create messages for OpenAI
            messages: list[ChatCompletionMessageParam] = [
                ChatCompletionSystemMessageParam(
                    role="system",
                    content=prompts.SYSTEM_PROMPT_PLOTLY_CHART,
                ),
                ChatCompletionUserMessageParam(
                    role="user", content=f"Question: {question}"
                ),
                ChatCompletionUserMessageParam(
                    role="user", content=f"Data Metadata:\n{json.dumps(metadata)}"
                ),
                ChatCompletionUserMessageParam(
                    role="user", content=f"Data top 25 rows:\n{df.head(25).to_string()}"
                ),
            ]

            # Add error context if available
            if error_message and failed_code:
                messages.extend(
                    [
                        {"role": "user", "content": f"Previous error: {error_message}"},
                        {"role": "user", "content": f"Failed code:\n{failed_code}"},
                    ]
                )

            # Get response based on model mode
            response: CodeGeneration = client.chat.completions.create(
                response_model=CodeGeneration,
                model=ALTERNATIVE_LLM_BIG,
                temperature=0,
                messages=messages,
            )

            code = response.code

            # Track code history
            code_history.append(
                ChartCodeHistory(
                    attempt=attempts,
                    code=code,
                    timestamp=datetime.now().isoformat(),
                )
            )

            # Validate the generated code
            is_valid, validation_message = _validate_chart_code(code)

            if not is_valid:
                validation_errors.append(
                    ChartValidationError(
                        attempt=attempts,
                        error=validation_message,
                        code=code,
                        timestamp=datetime.now().isoformat(),
                    )
                )
                continue

            try:
                # Create namespace for execution with single dataframe
                namespace = {
                    "pd": pd,
                    "np": np,
                    "df": df,  # Pass single dataframe instead of dictionary
                    "go": go,
                    "make_subplots": make_subplots,
                }

                # Execute the code with stdout/stderr capture
                stdout = io.StringIO()
                stderr = io.StringIO()

                with redirect_stdout(stdout), redirect_stderr(stderr):
                    exec(code, namespace)
                    fig1, fig2 = namespace["create_charts"](df)  # Pass single dataframe
                return ChartGenerationResult(
                    fig1=fig1,
                    fig2=fig2,
                    code=code,
                    validation=ValidationMessage(
                        is_valid=True, message=validation_message
                    ),
                    metadata=ChartGenerationMetadata(
                        timestamp=datetime.now().isoformat(),
                        question=question,
                        stdout=stdout.getvalue(),
                        stderr=stderr.getvalue(),
                    ),
                    attempts=attempts,
                    validation_errors=validation_errors,
                    execution_errors=execution_errors,
                    code_history=code_history,
                )

            except Exception as exec_error:
                logger.error(f"Execution error: {str(exec_error)}")
                execution_errors.append(
                    ChartExecutionError(
                        attempt=attempts,
                        error_type=type(exec_error).__name__,
                        error_message=str(exec_error),
                        code=code,
                        stdout=stdout.getvalue() if "stdout" in locals() else "",
                        stderr=stderr.getvalue() if "stderr" in locals() else "",
                        timestamp=datetime.now().isoformat(),
                    )
                )

                if attempts == max_attempts:
                    raise ValueError(
                        f"Failed to execute charts after {max_attempts} attempts. Last error: {str(exec_error)}"
                    )

        except Exception as e:
            execution_errors.append(
                ChartExecutionError(
                    attempt=attempts,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    code=code if "code" in locals() else None,
                    timestamp=datetime.now().isoformat(),
                )
            )

            if attempts == max_attempts:
                raise ValueError(
                    f"Failed to generate valid charts after {max_attempts} attempts: {str(e)}"
                )

    raise ValueError(f"Failed to generate valid charts after {max_attempts} attempts")


async def _generate_python_analysis_code(
    request: RunAnalysisRequest, validation_error: InvalidGeneratedCode | None = None
) -> str:
    """
    Generate Python analysis code based on JSON data and question.

    Parameters:
    - request: RunAnalysisRequest containing data and question
    - validation_errors: Past validation errors to include in prompt

    Returns:
    - Generated code
    """
    # Convert dictionary data structure to list of columns for all datasets
    all_columns = []
    all_descriptions = []
    all_data_types = []

    for dictionary in request.dictionary:
        for entry in dictionary.dictionary:
            all_columns.append(f"{dictionary.name}.{entry.column}")
            all_descriptions.append(entry.description)
            all_data_types.append(entry.data_type)

    # Create dictionary format for prompt
    dictionary_data = {
        "columns": all_columns,
        "descriptions": all_descriptions,
        "data_types": all_data_types,
    }

    # Get sample data and shape info for all datasets
    all_samples = []
    all_shapes = []

    for dataset in request.data:
        df = dataset.to_df()
        all_shapes.append(f"{dataset.name}: {df.shape[0]} rows x {df.shape[1]} columns")
        # Limit sample to 10 rows
        sample_df = df.head(10)
        all_samples.append(f"{dataset.name}:\n{sample_df.to_string()}")

    shape_info = "\n".join(all_shapes)
    sample_data = "\n\n".join(all_samples)

    # Create messages for OpenAI
    messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(
            role="system", content=prompts.SYSTEM_PROMPT_PYTHON_ANALYST
        ),
        ChatCompletionUserMessageParam(
            role="user", content=f"Business Question: {request.question}"
        ),
        ChatCompletionUserMessageParam(
            role="user", content=f"Data Shapes:\n{shape_info}"
        ),
        ChatCompletionUserMessageParam(
            role="user", content=f"Sample Data:\n{sample_data}"
        ),
        ChatCompletionUserMessageParam(
            role="user",
            content=f"Data Dictionary:\n{json.dumps(dictionary_data)}",
        ),
    ]

    # Add error context if available
    if validation_error:
        msg = type(validation_error).__name__ + f": {str(validation_error)}"
        messages.extend(
            [
                ChatCompletionUserMessageParam(
                    role="user",
                    content=f"Previous attempt failed with error: {msg}",
                ),
                ChatCompletionUserMessageParam(
                    role="user",
                    content=f"Failed code: {validation_error.code}",
                ),
                ChatCompletionUserMessageParam(
                    role="user",
                    content="Please generate new code that avoids this error.",
                ),
            ]
        )

    completion: CodeGeneration = client.chat.completions.create(
        response_model=CodeGeneration,
        model=ALTERNATIVE_LLM_BIG,
        temperature=0.1,
        messages=messages,
    )

    return completion.code


@cache
async def cleanse_dataframes(
    datasets: list[AnalystDataset],
) -> list[CleansedDataset]:
    """Clean and standardize multiple pandas DataFrames."""
    cleaned_datasets = []

    for dataset in datasets:
        df = dataset.to_df()
        if df.empty:
            raise ValueError(f"Dataset {dataset.name} is empty")

        report = CleansingReport(columns_cleaned=[], errors=[], warnings=[])

        # Clean column names
        original_cols = df.columns.tolist()
        df.columns = [re.sub(r"\s+", " ", col.strip()) for col in df.columns]  # type: ignore[assignment]
        cleaned_cols = df.columns.tolist()

        # Track column name changes
        for orig, cleaned in zip(original_cols, cleaned_cols):
            if orig != cleaned:
                report.columns_cleaned.append(orig)
                report.warnings.append(f"Column '{orig}' renamed to '{cleaned}'")

        # Process each column
        for col in df.columns:
            try:
                original = df[col].copy()

                # Handle numeric columns
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    if not df[col].equals(original):
                        report.columns_cleaned.append(col)

                # Handle potential numeric strings (with currency/percentage)
                elif (
                    df[col].dtype == "object"
                    and df[col].notna().all()
                    and df[col]
                    .str.replace(r"[$%,\s]", "", regex=True)
                    .str.match(r"^-?\d*\.?\d*$")
                    .all()
                ):
                    df[col] = pd.to_numeric(
                        df[col].astype(str).str.replace(r"[$%,\s]", "", regex=True),
                        errors="coerce",
                    )
                    report.columns_cleaned.append(col)

                # Handle dates
                elif is_date_column(df[col]):
                    df[col] = convert_datetime_series(df[col])
                    if not df[col].equals(original):
                        report.columns_cleaned.append(col)

                # Handle categorical
                elif df[col].dtype == "object":
                    mask = df[col].notna()
                    if mask.any():
                        temp = df.loc[mask, col]
                        if not pd.api.types.is_string_dtype(temp):
                            temp = temp.astype(str)
                        df.loc[mask, col] = temp.str.strip()
                        if not df[col].equals(original):
                            report.columns_cleaned.append(col)

            except Exception as e:
                report.errors.append(f"Error processing column {col}: {str(e)}")

        cleaned_datasets.append(
            CleansedDataset(
                name=dataset.name,
                data=df.replace({pd.NaT: None}).to_dict("records"),
                cleaning_report=report,
            )
        )
    return cleaned_datasets


async def get_dictionary(
    datasets: Sequence[AnalystDataset],
) -> list[DataDictionary]:
    """
    Generate data dictionary for multiple datasets.

    Parameters:
    - datasets: list[DatasetInput] containing datasets

    Returns:
    - Dictionary containing column descriptions and metadata
    """
    try:
        # Add debug logging
        logger.info(f"Received dictionary request with {len(datasets)} datasets")

        tasks = [_get_dictionary(dataset) for dataset in datasets]

        results = await asyncio.gather(*tasks)
        # Process datasets using ThreadPoolExecutor instead of ProcessPoolExecutor

        logger.info(f"Returning dictionary response with {len(results)} results")
        return results

    except Exception as e:
        msg = type(e).__name__ + f": {str(e)}"
        logger.error(f"Error in get_dictionary: {msg}")
        raise


async def rephrase_message(messages: ChatRequest) -> dict[str, Any]:
    """Process chat messages history and return a new question

    Args:
        messages: List of message dictionaries with 'role' and 'content' fields

    Returns:
        Dict[str, str]: Dictionary containing response content
    """
    # Convert messages to string format for prompt
    messages_str = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in messages.messages]
    )

    prompt_messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(
            content=prompts.SYSTEM_PROMPT_REPHRASE_MESSAGE,
            role="system",
        ),
        ChatCompletionUserMessageParam(
            content=f"Message History:\n{messages_str}",
            role="user",
        ),
    ]

    completion: EnhancedQuestionGeneration = client.chat.completions.create(
        response_model=EnhancedQuestionGeneration,
        model=ALTERNATIVE_LLM_BIG,
        messages=prompt_messages,
    )

    return completion.model_dump()


async def run_charts(request: RunChartsRequest) -> RunChartsResult:
    """
    Generate and execute chart code with validation.
    """
    # TODO: this needs a refactor, does duplicative transformations, loop appears broken, etc.
    # Convert JSON to DataFrame
    df = request.data.to_df()
    if df.empty:
        raise ValueError("Input DataFrame cannot be empty")

    dataframe_metadata = {
        "metadata_shape": list(df.shape),
        "metadata_describe": json.loads(df.describe(include="all").to_json()),
        "metadata_dtypes": json.loads(df.dtypes.astype(str).to_json()),
    }
    dataframe_metadata_clone = dataframe_metadata.copy()
    max_attempts = 3
    attempt = 0
    last_error = None
    last_failed_code = None

    while True:  # Changed to while True with explicit breaks
        try:
            # Generate charts with retry logic
            result = await _create_charts(
                df=df,
                question=request.question,
                metadata=dataframe_metadata_clone,
                error_message=last_error,
                failed_code=last_failed_code,
            )
            fig1_base64 = _figure_to_base64(result.fig1) if result.fig1 else None
            fig2_base64 = _figure_to_base64(result.fig2) if result.fig2 else None

            # Explicit return here
            return RunChartsResult(
                fig1=result.fig1,
                fig2=result.fig2,
                fig1_base_64=fig1_base64,
                fig2_base_64=fig2_base64,
                code=result.code,
                metadata=RunChartsResultMetadata(
                    timestamp=result.metadata.timestamp,
                    question=result.metadata.question,
                    stdout=result.metadata.stdout,
                    stderr=result.metadata.stderr,
                    dataframe_metadata=dataframe_metadata,
                    validation=result.validation,
                    attempts=result.attempts,
                    validation_errors=result.validation_errors,
                    execution_errors=result.execution_errors,
                    code_history=result.code_history,
                    performance=ChartPerformance(
                        memory_usage=_get_memory_usage(),
                        total_time=(
                            datetime.fromisoformat(result.metadata.timestamp)
                            - datetime.fromisoformat(result.code_history[0].timestamp)
                        ).total_seconds(),
                    ),
                ),
            )

        except Exception as e:
            attempt += 1
            last_error = str(e)
            last_failed_code = result.code if "result" in locals() else None

            if attempt >= max_attempts:
                error_context = {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "validation_errors": (
                        result.validation_errors if "result" in locals() else []
                    ),
                    "execution_errors": (
                        result.execution_errors if "result" in locals() else []
                    ),
                    "code_history": result.code_history if "result" in locals() else [],
                    "attempts": attempt,
                    "timestamp": datetime.now().isoformat(),
                }
                raise RuntimeError(str(error_context))

            # Always raise the exception
            raise e


async def get_business_analysis(
    request: BusinessAnalysisRequest,
) -> BusinessAnalysisResult:
    """
    Generate business analysis based on data and question.

    Parameters:
    - request: BusinessAnalysisRequest containing data and question

    Returns:
    - Dictionary containing analysis components
    """
    try:
        # Convert JSON data to DataFrame for analysis
        df = request.data.to_df()

        # Get first 1000 rows as CSV with quoted values for context
        df_csv = df.head(750).to_csv(index=False, quoting=1)

        # Create messages for OpenAI
        messages: list[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(
                role="system", content=prompts.SYSTEM_PROMPT_BUSINESS_ANALYSIS
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=f"Business Question: {request.question}",
            ),
            ChatCompletionUserMessageParam(
                role="user", content=f"Analyzed Data:\n{df_csv}"
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=f"Data Dictionary:\n{request.dictionary.model_dump_json()}",
            ),
        ]

        completion: BusinessAnalysisGeneration = client.chat.completions.create(
            response_model=BusinessAnalysisGeneration,
            model=ALTERNATIVE_LLM_BIG,
            temperature=0.1,
            messages=messages,
        )

        # Ensure all response fields are present
        metadata = BusinessAnalysisMetadata(
            timestamp=datetime.now().isoformat(),
            question=request.question,
            rows_analyzed=len(df),
            columns_analyzed=len(df.columns),
        )
        return BusinessAnalysisResult(
            **completion.model_dump(),
            metadata=metadata,
        )

    except Exception as e:
        msg = type(e).__name__ + f": {str(e)}"
        logger.error(f"Error in get_business_analysis: {msg}")
        raise


@reflect_code_generation_errors(max_attempts=3)
async def _run_analysis(
    request: RunAnalysisRequest,
    exception_history: list[InvalidGeneratedCode] | None = None,
) -> RunAnalysisResult:
    if not request.data:
        raise ValueError("Input data cannot be empty")

    if exception_history is None:
        exception_history = []

    code = await _generate_python_analysis_code(
        request, next(iter(exception_history), None)
    )

    dataframes: dict[str, pd.DataFrame] = {}
    for dataset in request.data:
        if dataset.data:
            df = dataset.to_df()
            dataframes[dataset.name] = df
        else:
            dataframes[dataset.name] = pd.DataFrame()

    # Create namespace for execution
    namespace = {"pd": pd, "np": np, "dfs": dataframes}
    # Capture stdout and stderr
    stdout = io.StringIO()
    stderr = io.StringIO()

    # Execute the code
    try:
        CodeValidator.validate_imports(code)
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exec(code, namespace)

            if "analyze_data" not in namespace:
                raise NameError(
                    "Generated code did not define a valid `analyze_data` function"
                )

            if not isinstance(namespace["analyze_data"], FunctionType):
                raise TypeError("`analyze_data` must be a function")

            result = namespace["analyze_data"](dataframes)

            if not isinstance(result, (pd.DataFrame, list, dict)):
                result = pd.DataFrame(result)
    except Exception as e:
        raise InvalidGeneratedCode(code=code, exception=e)
    return RunAnalysisResult(
        status="success",
        code=code,
        data=AnalystDataset(name="analysis_result", data=result),
        metadata=RunAnanlysisResultMetadata(
            timestamp=datetime.now().isoformat(),
            attempts=len(exception_history) + 1,
            error_history=[],  # TODO: fix if needed in final interface
            stdout=stdout.getvalue(),
            stderr=stderr.getvalue(),
            datasets_analyzed=len(dataframes),
            total_rows_analyzed=sum(
                len(df) for df in dataframes.values() if not df.empty
            ),
            total_columns_analyzed=sum(
                len(df.columns) for df in dataframes.values() if not df.empty
            ),
        ),
    )


async def run_analysis(
    request: RunAnalysisRequest,
) -> RunAnalysisResult:
    """Execute analysis workflow on datasets."""
    try:
        return await _run_analysis(request)
    except MaxReflectionAttempts:
        return RunAnalysisResult(
            status="failed",
            suggestions="Consider reformulating the question or checking data quality",
            metadata=RunAnanlysisResultMetadata(
                timestamp=datetime.now().isoformat(),
                attempts=0,  # TODO: fix if needed in final interface
                error_history=[],  # TODO: fix if needed in final interface
            ),
        )


async def _get_database_analysis_code(
    request: DatabaseAnalysisRequest,
) -> DatabaseAnalysisCodeGeneration:
    """
    Generate Snowflake SQL analysis code based on data samples and question.

    Parameters:
    - request: DatabaseAnalysisRequest containing data samples and question

    Returns:
    - Dictionary containing generated code and description
    """
    try:
        # Convert dictionary data structure to list of columns for all tables
        all_tables_info = [d.model_dump(mode="json") for d in request.dictionary]

        # Get sample data for all tables
        all_samples = []
        for table in request.data:
            df = table.to_df()
            sample_str = f"Table: {table.name}\n{df.head(10).to_string()}"
            all_samples.append(sample_str)

        # Create messages for OpenAI
        messages: list[ChatCompletionMessageParam] = [
            Database.get_system_prompt(),
            ChatCompletionUserMessageParam(
                content=f"Business Question: {request.question}",
                role="user",
            ),
            ChatCompletionUserMessageParam(
                content=f"Sample Data:\n{chr(10).join(all_samples)}", role="user"
            ),
            ChatCompletionUserMessageParam(
                content=f"Data Dictionary:\n{json.dumps(all_tables_info)}", role="user"
            ),
        ]

        # Add error context if available
        if request.error_message and request.failed_code:
            messages.extend(
                [
                    {"role": "user", "content": "Previous attempt failed with error:"},
                    {"role": "user", "content": request.error_message},
                    {"role": "user", "content": "Failed code:"},
                    {"role": "user", "content": request.failed_code},
                    {
                        "role": "user",
                        "content": "Please generate new code that avoids this error.",
                    },
                ]
            )

        # Get response from OpenAI
        completion = client.chat.completions.create(
            response_model=DatabaseAnalysisCodeGeneration,
            model=ALTERNATIVE_LLM_BIG,
            temperature=0.1,
            messages=messages,
        )

        return completion

    except Exception as e:
        msg = type(e).__name__ + f": {str(e)}"
        logger.error(f"Error in _get_snowflake_analysis_code: {msg}")
        raise


async def run_database_analysis(
    request: DatabaseAnalysisRequest, max_attempts: int = 3, timeout: int = 300
) -> DatabaseAnalysisResult:
    """Execute Snowflake analysis with retry logic and error handling"""
    attempts = 0
    error_history: list[AnalysisError] = []
    conn = None
    last_generated_code = None
    start_time = time.time()

    if not request.data:
        raise ValueError("Input data cannot be empty")
    try:
        while True:  # Changed from while attempts < max_attempts
            attempts += 1

            try:
                # Update request with error context if available
                if error_history:
                    request.error_message = error_history[-1].error
                    request.failed_code = error_history[-1].code

                # Generate SQL code
                code_result = await _get_database_analysis_code(request)
                sql_code = code_result.code
                last_generated_code = sql_code

                results, query_metadata = Database.execute_query(
                    query=sql_code, timeout=timeout
                )
                results = cast(list[dict[str, Any]], results)
                return DatabaseAnalysisResult(
                    status="success",
                    code=sql_code,
                    description=code_result.description,
                    data=AnalystDataset(name="analysis_result", data=results),
                    metadata=DatabaseAnalysisMetadata(
                        attempts=attempts,
                        execution_time=time.time() - start_time,
                        error_history=error_history,
                        memory_usage=_get_memory_usage(),
                        query_metadata=query_metadata,
                        tables_analyzed=len(request.data),
                        total_sample_rows=sum(
                            len(samples.to_df()) for samples in request.data
                        ),
                    ),
                )

            except Exception as e:
                error_history.append(
                    AnalysisError(
                        attempt=attempts,
                        error=str(e),
                        error_type=type(e).__name__,
                        code=sql_code if "sql_code" in locals() else None,
                        timestamp=datetime.now().isoformat(),
                        memory_usage=_get_memory_usage(),
                    )
                )

                if attempts >= max_attempts:
                    # Explicit return for max attempts reached
                    return DatabaseAnalysisResult(
                        status="failed",
                        last_generated_code=last_generated_code,
                        metadata=DatabaseAnalysisMetadata(
                            attempts=attempts,
                            error_history=error_history,
                            execution_time=time.time() - start_time,
                            memory_usage=_get_memory_usage(),
                        ),
                        suggestions="Consider reformulating the question or checking data access permissions",
                    )

                # Exponential backoff between attempts
                time.sleep(min(2**attempts, 10))
                continue  # Explicit continue

    except Exception as e:
        error_history.append(
            AnalysisError(
                attempt=attempts,
                error=str(e),
                error_type=type(e).__name__,
                code=sql_code if "sql_code" in locals() else None,
                timestamp=datetime.now().isoformat(),
                memory_usage=_get_memory_usage(),
            )
        )
        logger.error(f"Error in run_database_analysis: {str(e)}")
        return DatabaseAnalysisResult(
            status="failed",
            last_generated_code=last_generated_code,
            metadata=DatabaseAnalysisMetadata(
                attempts=attempts,
                error_history=error_history,
                execution_time=time.time() - start_time,
                memory_usage=_get_memory_usage(),
            ),
        )
