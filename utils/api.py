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
import base64
import functools
import io
import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from types import FunctionType
from typing import Any, Dict, List, Tuple

import datarobot as dr
import instructor
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import psutil
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
from utils.credentials import SnowflakeCredentials
from utils.datetime_helpers import convert_datetime_series, is_date_column
from utils.resources import LLMDeployment
from utils.schema import (
    AiCatalogDataset,
    AnalysisError,
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
    CleanseResult,
    CleansingReport,
    Code,
    CodeGenerationResult,
    CodeValidator,
    DataDictionariesAndMetadata,
    DataDictionary,
    DataDictionaryColumn,
    DataDictionaryMetadata,
    DatasetInput,
    DatasetOutput,
    DictionaryResult,
    EnhancedUserMessageForChat,
    MemoryUsage,
    QuestionList,
    QuestionSuggestionMetadata,
    QuestionSuggestions,
    QuestionValidationResult,
    RunAnalysisRequest,
    RunAnalysisResult,
    RunAnanlysisResultMetadata,
    RunChartsRequest,
    RunChartsResult,
    RunChartsResultMetadata,
    SnowflakeAnalysisCode,
    SnowflakeAnalysisMetadata,
    SnowflakeAnalysisRequest,
    SnowflakeAnalysisResult,
    ValidationMessage,
)
from utils.snowflake_helpers import create_snowflake_connection, execute_snowflake_query

logger = logging.getLogger("DataAnalystFrontend")

SNOWFLAKE_CREDENTIALS = SnowflakeCredentials()

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

MODEL_MODE = "openai"
DICTIONARY_BATCH_SIZE = 10


@functools.lru_cache(maxsize=2)
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
                f"{ds['datasetSize'] / (1024*1024):.1f} MB"
                if "datasetSize" in ds
                else "N/A"
            ),
        )
        for ds in datasets
    ]


@functools.lru_cache(maxsize=8)
def download_catalog_datasets(*args: Any) -> dict[str, list[dict[str, Any]]]:
    """Load selected datasets as pandas DataFrames

    Args:
        *args: list of dataset IDs to download

    Returns:
        dict[str, list[dict[str, Any]]]: Dictionary of dataset names and data
    """
    dataframes: dict[str, list[dict[str, Any]]] = {}
    dataset_ids = list(args)
    for id in dataset_ids:
        dataset = dr.Dataset.get(id)  # type: ignore[attr-defined]
        try:
            df_records = dataset.get_as_dataframe().to_dict(orient="records")

            df_records_converted = [
                {str(k): v for k, v in record.items()} for record in df_records
            ]
            dataframes[dataset.name] = df_records_converted
            logger.info(f"Successfully downloaded {dataset.name}")
        except Exception as e:
            logger.error(f"Failed to read dataset {dataset.name}: {str(e)}")
            continue
    return dataframes


def _process_column_batch(
    columns: List[str], df: pd.DataFrame, batch_size: int = 5
) -> Dict[str, str]:
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
    completion: DictionaryResult = client.chat.completions.create(
        response_model=DictionaryResult,
        model="gpt-4o-mini",
        messages=messages,
    )
    # response = json.loads(completion.choices[0].message.content)

    try:
        # Convert to dictionary format
        descriptions = completion.to_dict()

        # Only return descriptions for requested columns
        return {
            col: descriptions.get(col, "No description available") for col in columns
        }

    except ValueError as e:
        logger.error(f"Invalid dictionary response: {str(e)}")
        return {col: "No valid description available" for col in columns}


def _process_dataset(dataset: DatasetInput) -> DataDictionary:
    """Process a single dataset with parallel column batch processing"""
    try:
        batch_start = datetime.now()

        # Convert JSON to DataFrame
        df = pd.DataFrame(dataset.data)

        # Add debug logging
        logger.info(f"Processing dataset {dataset.name} with shape {df.shape}")

        # Handle empty dataset
        if df.empty:
            logger.warning(f"Dataset {dataset.name} is empty")
            return DataDictionary(
                name=dataset.name,
                dictionary=[],
                cache_hit=False,
                batch_time=0,
            )

        # Split columns into batches
        column_batches = [
            list(df.columns[i : i + DICTIONARY_BATCH_SIZE])
            for i in range(0, len(df.columns), DICTIONARY_BATCH_SIZE)
        ]
        logger.info(
            f"Created {len(column_batches)} batches for {len(df.columns)} columns"
        )

        # Process column batches using ThreadPoolExecutor
        batch_results = {}  # Change to dictionary to maintain column-description mapping
        with ThreadPoolExecutor() as executor:
            batch_futures = {
                executor.submit(
                    _process_column_batch, batch, df, DICTIONARY_BATCH_SIZE
                ): batch
                for batch in column_batches
            }

            # Collect results as they complete
            for future in as_completed(batch_futures):
                try:
                    result = future.result()
                    # Assuming process_column_batch returns a dictionary mapping columns to descriptions
                    batch_results.update(
                        result
                    )  # Merge results maintaining column mapping
                except Exception as e:
                    logger.error(f"Error processing batch: {str(e)}")
                    continue

        # Combine results
        dictionary = [
            DataDictionaryColumn(
                data_type=str(df[col].dtype),
                column=col,
                description=batch_results.get(col, "No description available"),
            )
            for col in df.columns
        ]

        logger.info(
            f"Created dictionary with {len(dictionary)} entries for dataset {dataset.name}"
        )

        batch_time = (datetime.now() - batch_start).total_seconds()

        return DataDictionary(
            name=dataset.name,
            dictionary=dictionary,
            cache_hit=False,
            batch_time=batch_time,
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
) -> QuestionValidationResult:
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

    return QuestionValidationResult(
        question=question,
        is_valid=is_valid,
        available_columns=found_columns,
        missing_columns=missing_columns,
        validation_message=message,
    )


async def _generate_question_suggestions(
    dictionary: pd.DataFrame, max_columns: int = 40
) -> QuestionSuggestions:
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
    if dictionary.empty:
        raise ValueError("Dictionary DataFrame cannot be empty")

    required_cols = ["column", "description", "data_type"]
    if not all(col in dictionary.columns for col in required_cols):
        raise ValueError(f"Dictionary must contain columns: {required_cols}")

    # Limit columns for OpenAI prompt
    total_columns = len(dictionary)
    if total_columns > max_columns:
        # Take first and last 20 columns
        half_max = max_columns // 2
        first_half = dictionary.head(half_max)
        last_half = dictionary.tail(half_max)

        # Remove any duplicates
        dictionary = pd.concat([first_half, last_half]).drop_duplicates()

    # Convert dictionary to format expected by OpenAI
    dict_data = {
        "columns": dictionary["column"].tolist(),
        "descriptions": dictionary["description"].tolist(),
        "data_types": dictionary["data_type"].tolist(),
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

    completion: QuestionList = client.chat.completions.create(
        response_model=QuestionList,
        model="gpt-4o-mini",
        messages=messages,
    )

    available_columns = dictionary["column"].tolist()
    validated_questions: list[QuestionValidationResult] = []

    for question in completion.questions:
        validated_questions.append(
            _validate_question_feasibility(question, available_columns)
        )

    metadata = QuestionSuggestionMetadata(
        total_columns=total_columns,
        columns_used=len(dictionary),
        timestamp=datetime.now().isoformat(),
        questions_generated=len(validated_questions),
        valid_questions=sum(1 for q in validated_questions if q.is_valid),
    )

    return QuestionSuggestions(questions=validated_questions, metadata=metadata)


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
            response: Code = client.chat.completions.create(
                response_model=Code,
                model="gpt-4o",
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
    request: RunAnalysisRequest, validation_error: dict[str, str] = {}
) -> dict[str, str]:
    """
    Generate Python analysis code based on JSON data and question.

    Parameters:
    - request: RunAnalysisRequest containing data and question
    - validation_errors: Past validation errors to include in prompt

    Returns:
    - Dictionary containing generated code and description
    """
    # Convert dictionary data structure to list of columns for all datasets
    all_columns = []
    all_descriptions = []
    all_data_types = []

    for dataset_name, dictionary_list in request.dictionary.items():
        for entry in dictionary_list:
            if isinstance(entry, dict) and "column" in entry:
                all_columns.append(f"{dataset_name}.{entry['column']}")
                all_descriptions.append(entry.get("description", ""))
                all_data_types.append(entry.get("data_type", ""))

    # Create dictionary format for prompt
    dictionary_data = {
        "columns": all_columns,
        "descriptions": all_descriptions,
        "data_types": all_data_types,
    }

    # Get sample data and shape info for all datasets
    all_samples = []
    all_shapes = []

    for dataset_name, dataset in request.data.items():
        df = pd.DataFrame(dataset)
        all_shapes.append(f"{dataset_name}: {df.shape[0]} rows x {df.shape[1]} columns")
        # Limit sample to 10 rows
        sample_df = df.head(10)
        all_samples.append(f"{dataset_name}:\n{sample_df.to_string()}")

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
        messages.extend(
            [
                ChatCompletionUserMessageParam(
                    role="user",
                    content=f"Previous attempt failed with error: {validation_error['error_message']}",
                ),
                ChatCompletionUserMessageParam(
                    role="user",
                    content=f"Failed code: {validation_error['failed_code']}",
                ),
                ChatCompletionUserMessageParam(
                    role="user",
                    content="Please generate new code that avoids this error.",
                ),
            ]
        )

    completion: Code = client.chat.completions.create(
        response_model=Code,
        model="gpt-4o",
        temperature=0.1,
        messages=messages,
    )

    return completion.model_dump()


async def _generate_analysis_code(
    request: RunAnalysisRequest, max_attempts: int = 10
) -> CodeGenerationResult:
    """Generate and validate analysis code with retry logic

    Args:
        request: RunAnalysisRequest containing data and question
        max_attempts: Maximum number of retry attempts for validation failures

    Returns:
        CodeGenerationResult containing generated code and metadata
    """
    attempts = 0
    validation_errors: list[str] = []
    validation_error: dict[str, str] = {}

    while attempts < max_attempts:
        attempts += 1
        try:
            # Get code from OpenAI
            code_response = await _generate_python_analysis_code(
                request, validation_error
            )

            # Validate the generated code
            is_valid, validation_message = CodeValidator.validate_imports(
                code_response["code"]
            )

            if is_valid:
                return CodeGenerationResult(
                    code=code_response["code"],
                    description=code_response["description"],
                    validation=ValidationMessage(
                        is_valid=True, message=validation_message
                    ),
                    metadata={
                        "timestamp": datetime.now().isoformat(),
                        "question": request.question,
                        "attempts": attempts,
                        "validation_history": validation_errors,
                    },
                    attempts=attempts,
                    validation_errors=validation_errors,
                )

            # If validation failed, add error to history and retry
            validation_errors.append(validation_message)
            validation_error["error_message"] = validation_message
            validation_error["failed_code"] = code_response["code"]

        except Exception as e:
            msg = type(e).__name__ + f": {str(e)}"
            validation_errors.append(msg)

    # If we get here, we've exhausted our attempts
    raise RuntimeError(f"Failed to generate valid code after {max_attempts} attempts")


async def cleanse_dataframes(
    datasets: list[DatasetInput],
) -> CleanseResult:
    """
    Clean and standardize multiple pandas DataFrames.

    Parameters:
    - datasets: list[DatasetInput] containing datasets to clean

    Returns:
    - CleanseResult containing cleaned datasets and metadata
    """
    try:
        logger.info("Starting cleanse_dataframes")
        cleaned_datasets = []
        total_datasets = len(datasets)

        for idx, dataset in enumerate(datasets):
            try:
                logger.info(f"Processing dataset: {dataset.name}")

                # Convert JSON to DataFrame
                df = pd.DataFrame(dataset.data)
                logger.debug(f"Created DataFrame with shape: {df.shape}")

                if df.empty:
                    raise ValueError("Input DataFrame is empty")

                # Initialize cleaning report
                cleaning_report = CleansingReport(
                    columns_cleaned=[], value_counts={}, errors=[], warnings=[]
                )

                # Clean column names - only remove leading/trailing whitespace and consecutive spaces
                original_columns = df.columns.tolist()
                df.columns = pd.Index(
                    [re.sub(r"\s+", " ", col.strip()) for col in df.columns], dtype=str
                )
                cleaned_columns = df.columns.tolist()

                # Track column name changes
                for orig, cleaned in zip(original_columns, cleaned_columns):
                    if orig != cleaned:
                        cleaning_report.columns_cleaned.append(orig)
                        cleaning_report.warnings.append(
                            f"Column '{orig}' renamed to '{cleaned}'"
                        )

                # Process each column
                for column in df.columns:
                    try:
                        # Store original value counts for reporting
                        original_counts = df[column].value_counts().to_dict()

                        # Clean numeric columns - more careful detection
                        if pd.api.types.is_numeric_dtype(df[column]):
                            try:
                                # Handle already numeric columns
                                df[column] = pd.to_numeric(df[column], errors="coerce")

                            except Exception as e:
                                cleaning_report.errors.append(
                                    f"Error cleaning numeric column {column}: {str(e)}"
                                )
                                continue
                        # Handle columns that might be numeric strings with currency/percentage
                        elif (
                            df[column].dtype == "object"
                            and df[column].notna().all()  # Only check non-null values
                            and df[column]
                            .str.replace(r"[$%,\s]", "", regex=True)
                            .str.match(r"^-?\d*\.?\d*$")
                            .all()
                        ):
                            try:
                                # Remove currency symbols, commas, and percentages
                                df[column] = pd.to_numeric(
                                    df[column]
                                    .astype(str)
                                    .str.replace(r"[$%,\s]", "", regex=True),
                                    errors="coerce",
                                )

                            except Exception as e:
                                cleaning_report.errors.append(
                                    f"Error cleaning numeric column {column}: {str(e)}"
                                )
                                continue

                        # Clean date columns
                        elif is_date_column(df[column]):
                            try:
                                original_values = df[column].copy()
                                # Convert to datetime strings using vectorized operation
                                df[column] = convert_datetime_series(df[column])

                                # Compare before and after
                                if not df[column].equals(original_values):
                                    cleaning_report.columns_cleaned.append(column)
                                    cleaning_report.value_counts[column] = {
                                        "before": {
                                            str(k): str(v)
                                            for k, v in original_counts.items()
                                        },
                                        "after": df[column]
                                        .value_counts()
                                        .to_dict(),  # Already strings
                                        "change_type": "date_cleaning",
                                    }
                            except Exception as e:
                                cleaning_report.errors.append(
                                    f"Error cleaning date column {column}: {str(e)}"
                                )
                                continue

                        # Clean categorical columns
                        elif df[column].dtype == "object":
                            try:
                                original_values = df[column].copy()

                                # Handle non-null values only
                                mask = df[column].notna()
                                if (
                                    mask.any()
                                ):  # Only process if there are any non-null values
                                    # Convert to string only if not already string
                                    temp_series = df.loc[mask, column]
                                    if not pd.api.types.is_string_dtype(temp_series):
                                        temp_series = temp_series.astype(str)

                                    # Only strip leading/trailing spaces, preserve internal spaces
                                    df.loc[mask, column] = temp_series.str.strip()

                                # Compare before and after
                                if not df[column].equals(original_values):
                                    cleaning_report.columns_cleaned.append(column)
                                    cleaning_report.value_counts[column] = {
                                        "before": original_counts,
                                        "after": df[column].value_counts().to_dict(),
                                        "change_type": "categorical_cleaning",
                                    }
                            except Exception as e:
                                cleaning_report.errors.append(
                                    f"Error cleaning categorical column {column}: {str(e)}"
                                )
                                continue

                    except Exception as e:
                        cleaning_report.errors.append(
                            f"Error processing column {column}: {str(e)}"
                        )
                        continue

                # Create DatasetOutput - ensure all data is JSON serializable
                cleaned_dataset = DatasetOutput(
                    name=dataset.name,
                    data=df.replace({pd.NaT: None}).to_dict(
                        "records"
                    ),  # Replace NaT with None
                    cleaning_report=cleaning_report,
                )
                cleaned_datasets.append(cleaned_dataset)
                logger.info(f"Successfully cleaned dataset: {dataset.name}")

            except Exception as e:
                logger.error(f"Error processing dataset {dataset.name}: {str(e)}")
                raise

        return CleanseResult(
            datasets=cleaned_datasets,
            metadata={
                "total_datasets": total_datasets,
                "timestamp": datetime.now().isoformat(),
                "version": "1.0",
            },
        )

    except Exception as e:
        msg = type(e).__name__ + f": {str(e)}"
        logger.error(f"Error in cleanse_dataframes: {msg}")
        raise


async def get_dictionary(datasets: list[DatasetInput]) -> DataDictionariesAndMetadata:
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

        metadata = DataDictionaryMetadata(
            total_datasets=len(datasets),
            processing_start=datetime.now().isoformat(),
            batch_times=[],
            errors=[],
        )

        # Process datasets using ThreadPoolExecutor instead of ProcessPoolExecutor
        with ThreadPoolExecutor() as executor:
            # Map datasets to futures
            dataset_futures = {
                executor.submit(_process_dataset, dataset): dataset.name
                for dataset in datasets
            }

            # Add debug logging
            logger.info(f"Created {len(dataset_futures)} dataset futures")

            # Collect results as they complete
            results = []
            for future in as_completed(dataset_futures):
                dataset_name = dataset_futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    metadata.batch_times.append(result.batch_time)
                    logger.info(
                        f"Processed dataset {dataset_name} with {len(result.dictionary)} entries"
                    )
                except Exception as e:
                    error_msg = f"Error processing dataset {dataset_name}: {str(e)}"
                    logger.error(error_msg)
                    metadata.errors.append(error_msg)
                    results.append(
                        DataDictionary(
                            name=dataset_name,
                            dictionary=[],
                            cache_hit=False,
                            batch_time=0,
                        )
                    )

        metadata.processing_end = datetime.now().isoformat()
        metadata.total_time = (
            datetime.fromisoformat(metadata.processing_end)
            - datetime.fromisoformat(metadata.processing_start)
        ).total_seconds()

        response = DataDictionariesAndMetadata(metadata=metadata, dictionaries=results)

        logger.info(f"Returning dictionary response with {len(results)} results")
        return response

    except Exception as e:
        msg = type(e).__name__ + f": {str(e)}"
        logger.error(f"Error in get_dictionary: {msg}")
        raise


async def suggest_questions(datasets: list[DatasetInput]) -> QuestionSuggestions:
    """
    Generate and validate suggested analysis questions.

    Parameters:
    - datasets: list[DatasetInput] containing dataset information

    Returns:
    - Dictionary containing suggested questions and metadata
    """
    if not datasets:
        raise ValueError("Must provide at least one dataset")
    try:
        # Convert dictionary list to DataFrame
        dict_df = pd.DataFrame(
            [
                {
                    "column": f"{dataset.name}.{col}",
                    "description": f"Column {col} from dataset {dataset.name}",
                    "data_type": str(pd.DataFrame(dataset.data)[col].dtype),
                }
                for dataset in datasets
                for col in pd.DataFrame(dataset.data).columns
            ]
        )

        return await _generate_question_suggestions(dict_df)
    except Exception as e:
        msg = type(e).__name__ + f": {str(e)}"
        logger.error(f"Error in suggest_questions: {msg}")
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

    completion: EnhancedUserMessageForChat = client.chat.completions.create(
        response_model=EnhancedUserMessageForChat,
        model="gpt-4o",
        messages=prompt_messages,
    )

    return completion.model_dump()


async def run_charts(request: RunChartsRequest) -> RunChartsResult:
    """
    Generate and execute chart code with validation.
    """
    # TODO: this needs a refactor, does duplicative transformations, loop appears broken, etc.
    # Convert JSON to DataFrame
    df = pd.DataFrame(request.data)
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
                df=df.head(25),
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
                    "validation_errors": result.validation_errors
                    if "result" in locals()
                    else [],
                    "execution_errors": result.execution_errors
                    if "result" in locals()
                    else [],
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
        df = pd.DataFrame(request.data)

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
                content=f"Data Dictionary:\n{json.dumps(request.dictionary)}",
            ),
        ]

        completion: BusinessAnalysisGeneration = client.chat.completions.create(
            response_model=BusinessAnalysisGeneration,
            model="gpt-4o",
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


async def run_analysis(request: RunAnalysisRequest) -> RunAnalysisResult:
    """
    Execute analysis workflow on datasets.

    Contains integration and retry logic
    """
    # TODO: should align to the run_charts refactor
    max_attempts = 3
    attempts = 0
    error_history: list[AnalysisError] = []

    # Input validation
    if not request.data:
        raise ValueError("Input data cannot be empty")

    try:
        # Convert JSON to DataFrames dictionary
        dataframes: dict[str, pd.DataFrame] = {}
        for dataset_name, dataset_records in request.data.items():
            if dataset_records:
                df = pd.DataFrame(dataset_records)
                dataframes[dataset_name] = df
            else:
                dataframes[dataset_name] = pd.DataFrame()

        while True:  # Changed from while attempts < max_attempts
            attempts += 1

            try:
                # Update request with error context if available
                if error_history:
                    request.error_message = error_history[-1].error
                    request.failed_code = error_history[-1].code

                # Generate code
                code_result = await _generate_analysis_code(request, max_attempts=4)

                # Validate the generated code
                if not code_result.validation.is_valid:
                    error_history.append(
                        AnalysisError(
                            attempt=attempts,
                            error=code_result.validation.message,
                            error_type="validation_error",
                            code=code_result.code,
                            timestamp=datetime.now().isoformat(),
                            memory_usage=_get_memory_usage(),
                        )
                    )
                    continue

                # Create namespace for execution
                namespace = {"pd": pd, "np": np, "dfs": dataframes}

                # Capture stdout and stderr
                stdout = io.StringIO()
                stderr = io.StringIO()

                # Execute the code
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    exec(code_result.code, namespace)

                    if "analyze_data" not in namespace:
                        raise ValueError(
                            "Generated code did not define analyze_data function"
                        )

                    if not isinstance(namespace["analyze_data"], FunctionType):
                        raise ValueError("analyze_data is not a function")

                    result = namespace["analyze_data"](dataframes)

                    if not isinstance(result, (pd.DataFrame, list, dict)):
                        result = pd.DataFrame(result)
                    if isinstance(result, pd.DataFrame):
                        result = result.to_dict("records")

                # If we get here, execution was successful - explicit return
                return RunAnalysisResult(
                    status="success",
                    code=code_result.code,
                    data=result,
                    metadata=RunAnanlysisResultMetadata(
                        timestamp=datetime.now().isoformat(),
                        attempts=attempts,
                        error_history=error_history,
                        stdout=stdout.getvalue(),
                        stderr=stderr.getvalue(),
                        datasets_analyzed=len(dataframes),
                        total_rows_analyzed=sum(
                            len(df) for df in dataframes.values() if not df.empty
                        ),
                        total_columns_analyzed=sum(
                            len(df.columns)
                            for df in dataframes.values()
                            if not df.empty
                        ),
                    ),
                )

            except Exception as e:
                error_history.append(
                    AnalysisError(
                        attempt=attempts,
                        error=str(e),
                        error_type=type(e).__name__,
                        code=code_result.code if "code_result" in locals() else None,
                        stdout=stdout.getvalue() if "stdout" in locals() else "",
                        stderr=stderr.getvalue() if "stderr" in locals() else "",
                        timestamp=datetime.now().isoformat(),
                        memory_usage=_get_memory_usage(),
                    )
                )

                if attempts >= max_attempts:
                    # Explicit return for failure case
                    return RunAnalysisResult(
                        status="failed",
                        suggestions="Consider reformulating the question or checking data quality",
                        metadata=RunAnanlysisResultMetadata(
                            timestamp=datetime.now().isoformat(),
                            attempts=attempts,
                            error_history=error_history,
                        ),
                    )
                # If not max attempts, continue the loop
                continue

    except Exception as e:
        msg = type(e).__name__ + f": {str(e)}"
        logger.error(f"Error in run_analysis: {msg}")
        raise


async def _get_snowflake_analysis_code(
    request: SnowflakeAnalysisRequest,
) -> SnowflakeAnalysisCode:
    """
    Generate Snowflake SQL analysis code based on data samples and question.

    Parameters:
    - request: SnowflakeAnalysisRequest containing data samples and question

    Returns:
    - Dictionary containing generated code and description
    """
    try:
        # Convert dictionary data structure to list of columns for all tables
        all_tables_info = []

        for table_name, dictionary_list in request.dictionary.items():
            table_info = {
                "table_name": table_name,
                "columns": [],
                "descriptions": [],
                "data_types": [],
            }

            for entry in dictionary_list:
                if isinstance(entry, dict):
                    table_info["columns"].append(entry.get("column", ""))
                    table_info["descriptions"].append(entry.get("description", ""))
                    table_info["data_types"].append(entry.get("data_type", ""))

            all_tables_info.append(table_info)

        # Get sample data for all tables
        all_samples = []
        for table_name, sample_data in request.data.items():
            df = pd.DataFrame(sample_data)
            sample_str = f"Table: {table_name}\n{df.head(10).to_string()}"
            all_samples.append(sample_str)

        # Create messages for OpenAI
        messages: list[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(
                role="system",
                content=prompts.SYSTEM_PROMPT_SNOWFLAKE.format(
                    warehouse=SNOWFLAKE_CREDENTIALS.warehouse,
                    database=SNOWFLAKE_CREDENTIALS.database,
                    schema=SNOWFLAKE_CREDENTIALS.db_schema,
                ),
            ),
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
            response_model=SnowflakeAnalysisCode,
            model="gpt-4o",
            temperature=0.1,
            messages=messages,
        )

        return completion

    except Exception as e:
        msg = type(e).__name__ + f": {str(e)}"
        logger.error(f"Error in _get_snowflake_analysis_code: {msg}")
        raise


async def run_snowflake_analysis(
    request: SnowflakeAnalysisRequest, max_attempts: int = 3, timeout: int = 300
) -> SnowflakeAnalysisResult:
    """Execute Snowflake analysis with retry logic and error handling"""
    attempts = 0
    error_history: list[AnalysisError] = []
    conn = None
    last_generated_code = None
    start_time = time.time()

    if not request.data:
        raise ValueError("Input data cannot be empty")

    try:
        conn = create_snowflake_connection()

        while True:  # Changed from while attempts < max_attempts
            attempts += 1

            try:
                # Update request with error context if available
                if error_history:
                    request.error_message = error_history[-1].error
                    request.failed_code = error_history[-1].code

                # Generate SQL code
                code_result = await _get_snowflake_analysis_code(request)
                sql_code = code_result.code
                last_generated_code = sql_code

                results, query_metadata = execute_snowflake_query(
                    conn=conn, query=sql_code, timeout=timeout
                )

                return SnowflakeAnalysisResult(
                    status="success",
                    code=sql_code,
                    description=code_result.description,
                    data=results,
                    metadata=SnowflakeAnalysisMetadata(
                        attempts=attempts,
                        execution_time=time.time() - start_time,
                        error_history=error_history,
                        memory_usage=_get_memory_usage(),
                        query_metadata=query_metadata,
                        tables_analyzed=len(request.data),
                        total_sample_rows=sum(
                            len(samples) for samples in request.data.values()
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
                    return SnowflakeAnalysisResult(
                        status="failed",
                        last_generated_code=last_generated_code,
                        metadata=SnowflakeAnalysisMetadata(
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
        logger.error(f"Error in run_snowflake_analysis: {str(e)}")
        return SnowflakeAnalysisResult(
            status="failed",
            last_generated_code=last_generated_code,
            metadata=SnowflakeAnalysisMetadata(
                attempts=attempts,
                error_history=error_history,
                execution_time=time.time() - start_time,
                memory_usage=_get_memory_usage(),
            ),
        )
    finally:
        if conn:
            conn.close()
