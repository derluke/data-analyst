import ast
import base64
import hashlib
import io
import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import datarobot as dr
from fastapi import HTTPException
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import psutil
from openai import OpenAI
from plotly.subplots import make_subplots
from pydantic import ValidationError

sys.path.append("..")

from utils import prompts
from utils.resources import ChatAgentDeployment
from utils.schema import (
    BusinessAnalysisRequest,
    ChartGenerationResult,
    ChatRequest,
    CodeGenerationResult,
    CodeValidator,
    DataDictionaryColumn,
    DatasetInput,
    DictionaryRequest,
    ProcessedDatasetStats,
    QuestionValidationResult,
    RunAnalysisRequest,
    RunChartsRequest,
)

try:
    chat_agent_deployment_id = ChatAgentDeployment().id
    deployment_chat_base_url = (
        dr.Client().endpoint + f"/deployments/{chat_agent_deployment_id}/"
    )

    client = OpenAI(api_key=dr.Client().token, base_url=deployment_chat_base_url)

except ValidationError as e:
    raise ValueError(
        "Unable to load Deployment ID."
        "If running locally, verify you have selected the correct "
        "stack and that it is active using `pulumi stack output`. "
        "If running in DataRobot, verify your runtime parameters have been set correctly."
    ) from e


# Cache key generator for DataFrames
def _generate_df_hash(df: pd.DataFrame) -> str:
    """Generate a hash key for DataFrame caching based on content"""
    # Get sample of data and column info for hash
    sample = df.head(100).to_json()
    cols = ",".join(df.columns)
    dtypes = ",".join(df.dtypes.astype(str))

    # Create hash
    hash_input = f"{sample}{cols}{dtypes}".encode()
    return hashlib.md5(hash_input).hexdigest()


def process_column_batch(
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
            except:
                continue

    # Create messages for OpenAI
    messages = [
        {"role": "system", "content": prompts.SYSTEM_PROMPT_GET_DICTIONARY},
        {"role": "user", "content": f"Data: {json.dumps(sample_data)}"},
        {
            "role": "user",
            "content": f"Statistical Summary: {json.dumps(numeric_summary)}",
        },
    ]

    if categories:
        messages.append(
            {"role": "user", "content": f"Categorical Values: {json.dumps(categories)}"}
        )

    # Get descriptions from OpenAI
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        response_format={"type": "json_object"},
        stream=False,
    )

    response = json.loads(completion.choices[0].message.content)

    # Extract descriptions from response and map to columns
    descriptions = {}
    if isinstance(response, dict):
        # Check if response has descriptions list
        if "descriptions" in response and isinstance(response["descriptions"], list):
            # Map descriptions to columns, handling potential length mismatch
            for col, desc in zip(columns, response["descriptions"]):
                descriptions[col] = desc
        # Fallback: check if response has column-specific descriptions
        else:
            for col in columns:
                descriptions[col] = response.get(col, "No description available")

    return descriptions


def process_dataset(dataset: DatasetInput) -> ProcessedDatasetStats:
    """Process a single dataset with parallel column batch processing"""
    try:
        batch_start = datetime.now()

        # Convert JSON to DataFrame
        df = pd.DataFrame(dataset.data)

        # Add debug logging
        logging.info(f"Processing dataset {dataset.name} with shape {df.shape}")

        # Handle empty dataset
        if df.empty:
            logging.warning(f"Dataset {dataset.name} is empty")
            return ProcessedDatasetStats(
                name=dataset.name,
                dictionary=[],
                cache_hit=False,
                batch_time=0,
            )

        # Split columns into batches
        column_batches = [
            list(df.columns[i : i + prompts.DICTIONARY_BATCH_SIZE])
            for i in range(0, len(df.columns), prompts.DICTIONARY_BATCH_SIZE)
        ]
        logging.info(
            f"Created {len(column_batches)} batches for {len(df.columns)} columns"
        )

        # Process column batches using ThreadPoolExecutor
        batch_results = (
            {}
        )  # Change to dictionary to maintain column-description mapping
        with ThreadPoolExecutor() as executor:
            batch_futures = {
                executor.submit(
                    process_column_batch, batch, df, prompts.DICTIONARY_BATCH_SIZE
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
                    logging.error(f"Error processing batch: {str(e)}")
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

        logging.info(
            f"Created dictionary with {len(dictionary)} entries for dataset {dataset.name}"
        )

        batch_time = (datetime.now() - batch_start).total_seconds()

        return ProcessedDatasetStats(
            name=dataset.name,
            dictionary=dictionary,
            cache_hit=False,
            batch_time=batch_time,
        )

    except Exception as e:
        logging.error(f"Error processing dataset {dataset.name}: {str(e)}")
        raise Exception(f"Error processing dataset {dataset.name}: {str(e)}")


# Add memory management helper
def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        "rss": memory_info.rss / 1024 / 1024,  # RSS in MB
        "vms": memory_info.vms / 1024 / 1024,  # VMS in MB
        "percent": process.memory_percent(),
    }


def validate_question_feasibility(
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


async def generate_question_suggestions(
    dictionary: pd.DataFrame, max_columns: int = 40
) -> Dict[str, Any]:
    """Generate and validate suggested analysis questions

    Args:
        dictionary: DataFrame containing data dictionary
        max_columns: Maximum number of columns to include in prompt

    Returns:
        Dict containing:
            - questions: List of validated question objects
            - metadata: Dictionary of processing information
    """
    try:
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
        messages = [
            {"role": "system", "content": prompts.SYSTEM_PROMPT_SUGGEST_A_QUESTION},
            {"role": "user", "content": f"Data Dictionary:\n{json.dumps(dict_data)}"},
        ]

        # Get suggestions from OpenAI
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            response_format={"type": "json_object"},
        )

        # Parse response
        response = json.loads(completion.choices[0].message.content)

        # Validate each suggested question
        available_columns = dictionary["column"].tolist()
        validated_questions = []

        for key in ["question1", "question2", "question3"]:
            if question := response.get(key):
                validation = validate_question_feasibility(question, available_columns)
                validated_questions.append(
                    {
                        "question": validation.question,
                        "is_valid": validation.is_valid,
                        "available_columns": validation.available_columns,
                        "missing_columns": validation.missing_columns,
                        "validation_message": validation.validation_message,
                    }
                )

        # Prepare metadata
        metadata = {
            "total_columns": total_columns,
            "columns_used": len(dictionary),
            "timestamp": datetime.now().isoformat(),
            "questions_generated": len(validated_questions),
            "valid_questions": sum(1 for q in validated_questions if q["is_valid"]),
        }

        return {"questions": validated_questions, "metadata": metadata}

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def validate_chart_code(code: str) -> Tuple[bool, str]:
    """Validate chart generation code for safety and correctness"""
    try:
        tree = ast.parse(code)
        imports = []

        # Check imports
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    imports.extend(n.name.split(".")[0] for n in node.names)
                else:
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


def figure_to_base64(fig: go.Figure) -> Optional[str]:
    """Convert Plotly figure to base64 encoded PNG"""
    try:
        if not isinstance(fig, go.Figure):
            raise ValueError(f"Expected plotly.graph_objects.Figure, got {type(fig)}")
        img_bytes = fig.to_image(format="png")
        return base64.b64encode(img_bytes).decode("utf-8")
    except Exception as e:
        logging.error(f"Failed to convert figure to base64: {str(e)}")
        return None


async def create_charts(
    df: pd.DataFrame,
    question: str,
    metadata: Dict[str, Any],
    error_message: Optional[str] = None,
    failed_code: Optional[str] = None,
    max_attempts: int = 3,
) -> ChartGenerationResult:
    """Generate and validate chart code with retry logic"""
    attempts = 0
    validation_errors = []
    execution_errors = []
    code_history = []

    while attempts < max_attempts:
        attempts += 1

        try:
            # Get chart code from OpenAI
            completion = client.chat.completions.create(
                model="gpt-4o",
                temperature=0,
                messages=[
                    {"role": "system", "content": prompts.SYSTEM_PROMPT_PLOTLY_CHART},
                    {"role": "user", "content": f"Question: {question}"},
                    {
                        "role": "user",
                        "content": f"Data Metadata:\n{json.dumps(metadata)}",
                    },
                    {
                        "role": "user",
                        "content": f"Data top 25 rows:\n{df.head(25).to_string()}",
                    },
                    *(
                        [
                            {
                                "role": "user",
                                "content": f"Previous error: {error_message}",
                            },
                            {"role": "user", "content": f"Failed code:\n{failed_code}"},
                        ]
                        if error_message and failed_code
                        else []
                    ),
                ],
                response_format={"type": "json_object"},
                stream=False,
            )

            response = json.loads(completion.choices[0].message.content)
            code = response.get("code")

            # Track code history
            code_history.append(
                {
                    "attempt": attempts,
                    "code": code,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            # Validate the generated code
            is_valid, validation_message = validate_chart_code(code)

            if not is_valid:
                validation_errors.append(
                    {
                        "attempt": attempts,
                        "error": validation_message,
                        "code": code,
                        "timestamp": datetime.now().isoformat(),
                    }
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
                    validation={"is_valid": True, "message": validation_message},
                    metadata={
                        "timestamp": datetime.now().isoformat(),
                        "question": question,
                        "attempts": attempts,
                        "stdout": stdout.getvalue(),
                        "stderr": stderr.getvalue(),
                    },
                    attempts=attempts,
                    validation_errors=validation_errors,
                    execution_errors=execution_errors,
                    code_history=code_history,
                )

            except Exception as exec_error:
                execution_errors.append(
                    {
                        "attempt": attempts,
                        "error_type": type(exec_error).__name__,
                        "error_message": str(exec_error),
                        "code": code,
                        "stdout": stdout.getvalue() if "stdout" in locals() else "",
                        "stderr": stderr.getvalue() if "stderr" in locals() else "",
                        "timestamp": datetime.now().isoformat(),
                    }
                )

                if attempts == max_attempts:
                    raise ValueError(
                        f"Failed to execute charts after {max_attempts} attempts. Last error: {str(exec_error)}"
                    )

        except Exception as e:
            execution_errors.append(
                {
                    "attempt": attempts,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "code": code if "code" in locals() else None,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            if attempts == max_attempts:
                raise ValueError(
                    f"Failed to generate valid charts after {max_attempts} attempts: {str(e)}"
                )

    raise ValueError(f"Failed to generate valid charts after {max_attempts} attempts")


async def process_chat(messages: List[Dict[str, str]]) -> Dict[str, str]:
    """Process chat messages and return complete response

    Args:
        messages: List of message dictionaries with 'role' and 'content' fields

    Returns:
        Dict[str, str]: Dictionary containing response content

    Raises:
        Exception: If OpenAI API call fails
    """
    # Convert messages to string format for prompt
    messages_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

    prompt_messages = [
        {"role": "system", "content": prompts.SYSTEM_PROMPT_CHAT},
        {"role": "user", "content": f"Message History:\n{messages_str}"},
    ]

    # Get non-streaming response from OpenAI
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=prompt_messages,
        response_format={"type": "json_object"},
        stream=False,
    )

    return json.loads(completion.choices[0].message.content)


async def generate_analysis_code(
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
    validation_errors = []

    while attempts < max_attempts:
        attempts += 1

        try:
            # Get code from OpenAI
            code_response = await get_python_analysis_code(request)

            # Validate the generated code
            is_valid, validation_message = CodeValidator.validate_imports(
                code_response["code"]
            )

            if is_valid:
                return CodeGenerationResult(
                    code=code_response["code"],
                    description=code_response["description"],
                    validation={"is_valid": True, "message": validation_message},
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

        except Exception as e:
            validation_errors.append(str(e))
            if attempts == max_attempts:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to generate valid code after {max_attempts} attempts: {str(e)}",
                )

    # If we get here, we've exhausted our attempts
    raise HTTPException(
        status_code=500,
        detail=f"Failed to generate valid code after {max_attempts} attempts",
    )
