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
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import datarobot as dr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import psutil
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from openai import OpenAI
from plotly.subplots import make_subplots
from pydantic import BaseModel, ValidationError, validator

sys.path.append("..")

from utils import prompts
from utils.resources import ChatAgentDeployment

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

# Initialize FastAPI app
app = FastAPI(
    title="Data Analyst API",
    description="""
    An intelligent API for data analysis that provides capabilities including:
    - Data cleansing and standardization
    - Data dictionary generation
    - Question suggestions
    - Python code generation
    - Chart creation
    - Business analysis
    
    The API uses OpenAI's GPT models for intelligent analysis and response generation.
    """,
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Add custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {"type": "apiKey", "in": "header", "name": "X-API-Key"}
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# Add custom exceptions at the top of the file
class NumericCleaningError(Exception):
    """Raised when numeric cleaning fails"""

    pass


class DateCleaningError(Exception):
    """Raised when date parsing fails"""

    pass


class CategoryCleaningError(Exception):
    """Raised when categorical cleaning fails"""

    pass


class EmptyDataError(Exception):
    """Raised when cleaning results in empty data"""

    pass


class DatasetInput(BaseModel):
    name: str
    data: List[Dict[str, Any]]


class CleanseRequest(BaseModel):
    datasets: List[DatasetInput]


class CleansingReport(BaseModel):
    columns_cleaned: List[str]
    value_counts: Dict[str, int]
    errors: List[str]
    warnings: List[str]


class DatasetOutput(BaseModel):
    name: str
    data: List[Dict[str, Any]]
    cleaning_report: CleansingReport


class CleanseResponse(BaseModel):
    datasets: List[DatasetOutput]
    metadata: Dict[str, Any]


class DictionaryRequest(BaseModel):
    data: List[Dict[str, Any]]

    @validator("data")
    def validate_data(cls, v):
        if not isinstance(v, list):
            raise ValueError("Input data must be a list of dictionaries")
        return v


def convert_to_datetime(value: Any, column: str) -> Optional[datetime]:
    """Convert a value to datetime with flexible format handling

    Args:
        value: Value to convert
        column: Column name for error reporting

    Returns:
        datetime or None if conversion fails
    """
    if pd.isna(value):
        return None

    try:
        # First try pandas to_datetime with coerce
        result = pd.to_datetime(value, infer_datetime_format=True)
        # Convert Timestamp to datetime
        if isinstance(result, pd.Timestamp):
            return result.to_pydatetime()
        return result
    except:
        try:
            # Try dateutil parser as fallback
            from dateutil import parser

            parsed = parser.parse(str(value))
            # Ensure we return a datetime object
            return parsed.replace(tzinfo=None)
        except:
            return None


@app.post(
    "/cleanse_dataframes",
    response_model=CleanseResponse,
    summary="Cleanse and standardize multiple datasets",
    description="""
    Clean and standardize multiple pandas DataFrames with progress reporting.
    
    The endpoint handles:
    - Column name standardization
    - Numeric data cleaning
    - Date format standardization
    - Categorical data cleaning
    
    Returns a detailed cleaning report for each dataset.
    """,
    response_description="Cleaned datasets with cleaning reports",
    tags=["Data Cleaning"],
)
async def cleanse_dataframes(
    request: CleanseRequest,
    progress_callback: Optional[Callable[[str, int], None]] = None,
) -> CleanseResponse:
    """
    Clean and standardize multiple pandas DataFrames.

    Parameters:
    - request: CleanseRequest containing datasets to clean
    - progress_callback: Optional callback for progress reporting

    Returns:
    - CleanseResponse containing cleaned datasets and metadata

    Raises:
    - HTTPException: If cleaning fails
    """
    try:
        logging.info("Starting cleanse_dataframes")
        cleaned_datasets = []
        total_datasets = len(request.datasets)

        for idx, dataset in enumerate(request.datasets):
            try:
                logging.info(f"Processing dataset: {dataset.name}")

                # Convert JSON to DataFrame
                df = pd.DataFrame(dataset.data)
                logging.debug(f"Created DataFrame with shape: {df.shape}")

                if df.empty:
                    raise EmptyDataError("Input DataFrame is empty")

                # Initialize cleaning report
                cleaning_report = CleansingReport(
                    columns_cleaned=[], value_counts={}, errors=[], warnings=[]
                )

                # Clean column names - remove consecutive whitespace
                original_columns = df.columns.tolist()
                df.columns = [" ".join(col.split()) for col in df.columns]
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

                        # Clean numeric columns
                        if (
                            pd.api.types.is_numeric_dtype(df[column])
                            or df[column].str.contains(r"[$%,]").any()
                        ):
                            try:
                                # Remove currency symbols, commas, and percentages
                                df[column] = pd.to_numeric(
                                    df[column]
                                    .astype(str)
                                    .str.replace(r"[$%,]", "", regex=True),
                                    errors="coerce",
                                )

                                # Compare value counts and report changes
                                new_counts = df[column].value_counts().to_dict()
                                if original_counts != new_counts:
                                    cleaning_report.value_counts[column] = {
                                        "before": original_counts,
                                        "after": new_counts,
                                        "change_type": "numeric_cleaning",
                                    }
                                    cleaning_report.columns_cleaned.append(column)
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
                                df[column] = (
                                    df[column].astype(str).str.strip().str.title()
                                )

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
                logging.info(f"Successfully cleaned dataset: {dataset.name}")

                # Report progress if callback provided
                if progress_callback:
                    progress = int((idx + 1) / total_datasets * 100)
                    await progress_callback(f"Processed {dataset.name}", progress)

            except Exception as e:
                logging.error(f"Error processing dataset {dataset.name}: {str(e)}")
                raise

        return CleanseResponse(
            datasets=cleaned_datasets,
            metadata={
                "total_datasets": total_datasets,
                "timestamp": datetime.now().isoformat(),
                "version": "1.0",
            },
        )

    except Exception as e:
        logging.error(f"Error in cleanse_dataframes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Cache key generator for DataFrames
def generate_df_hash(df: pd.DataFrame) -> str:
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


@app.post(
    "/get_dictionary",
    response_model=Dict[str, Any],
    summary="Generate data dictionary",
    description="""
    Generate comprehensive data dictionary for multiple datasets.
    
    The endpoint:
    - Analyzes column metadata
    - Generates column descriptions
    - Provides data types and sample values
    - Handles parallel processing for large datasets
    
    Returns detailed dictionary entries for all columns.
    """,
    response_description="Data dictionary with column descriptions",
    tags=["Data Dictionary"],
)
async def get_dictionary(request: DictionaryRequest) -> Dict[str, Any]:
    """
    Generate data dictionary for multiple datasets.

    Parameters:
    - request: DictionaryRequest containing datasets

    Returns:
    - Dictionary containing column descriptions and metadata

    Raises:
    - HTTPException: If dictionary generation fails
    """
    try:
        # Add debug logging
        logging.info(
            f"Received dictionary request with {len(request.datasets)} datasets"
        )

        metadata = {
            "total_datasets": len(request.datasets),
            "processing_start": datetime.now().isoformat(),
            "batch_times": [],
            "errors": [],
        }

        # Process datasets using ThreadPoolExecutor instead of ProcessPoolExecutor
        with ThreadPoolExecutor() as executor:
            # Map datasets to futures
            dataset_futures = {
                executor.submit(process_dataset, dataset): dataset.name
                for dataset in request.datasets
            }

            # Add debug logging
            logging.info(f"Created {len(dataset_futures)} dataset futures")

            # Collect results as they complete
            results = []
            for future in as_completed(dataset_futures):
                dataset_name = dataset_futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    metadata["batch_times"].append(result["batch_time"])
                    logging.info(
                        f"Processed dataset {dataset_name} with {len(result.get('dictionary', []))} entries"
                    )
                except Exception as e:
                    error_msg = f"Error processing dataset {dataset_name}: {str(e)}"
                    logging.error(error_msg)
                    metadata["errors"].append(error_msg)
                    results.append(
                        {
                            "name": dataset_name,
                            "dictionary": [],
                            "cache_hit": False,
                            "error": error_msg,
                        }
                    )

        metadata["processing_end"] = datetime.now().isoformat()
        metadata["total_time"] = (
            datetime.fromisoformat(metadata["processing_end"])
            - datetime.fromisoformat(metadata["processing_start"])
        ).total_seconds()

        response = {"dictionaries": results, "metadata": metadata}
        logging.info(f"Returning dictionary response with {len(results)} results")
        return response

    except Exception as e:
        logging.error(f"Error in get_dictionary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def process_dataset(dataset: DatasetInput) -> Dict[str, Any]:
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
            return {
                "name": dataset.name,
                "dictionary": [],
                "cache_hit": False,
                "batch_time": 0,
            }

        # Generate cache key
        df_hash = generate_df_hash(df)

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
            {
                "data_type": str(df[col].dtype),
                "column": col,
                "description": batch_results.get(col, "No description available"),
            }
            for col in df.columns
        ]

        logging.info(
            f"Created dictionary with {len(dictionary)} entries for dataset {dataset.name}"
        )

        batch_time = (datetime.now() - batch_start).total_seconds()

        return {
            "name": dataset.name,
            "dictionary": dictionary,
            "cache_hit": False,
            "batch_time": batch_time,
        }

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


@dataclass
class QuestionValidationResult:
    """Stores validation results for suggested questions"""

    question: str
    is_valid: bool
    available_columns: List[str]
    missing_columns: List[str]
    validation_message: str


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
            stream=False,
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


@app.post(
    "/suggest_questions",
    response_model=Dict[str, Any],
    summary="Suggest analysis questions",
    description="""
    Generate and validate suggested analysis questions based on available data.
    
    The endpoint:
    - Analyzes available columns
    - Suggests relevant business questions
    - Validates question feasibility
    - Provides context for each suggestion
    
    Returns validated questions with metadata.
    """,
    response_description="Suggested analysis questions with validation",
    tags=["Question Generation"],
)
async def suggest_questions(request: DictionaryRequest) -> Dict[str, Any]:
    """
    Generate and validate suggested analysis questions.

    Parameters:
    - request: DictionaryRequest containing dataset information

    Returns:
    - Dictionary containing suggested questions and metadata

    Raises:
    - HTTPException: If question generation fails
    """
    try:
        # Input validation
        if not request.datasets:
            raise ValueError("Dictionary cannot be empty")

        # Convert dictionary list to DataFrame
        dict_df = pd.DataFrame(
            [
                {
                    "column": f"{dataset.name}.{col}",
                    "description": f"Column {col} from dataset {dataset.name}",
                    "data_type": str(pd.DataFrame(dataset.data)[col].dtype),
                }
                for dataset in request.datasets
                for col in pd.DataFrame(dataset.data).columns
            ]
        )

        return await generate_question_suggestions(dict_df)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class RunAnalysisRequest(BaseModel):
    """Request model for analysis endpoint

    Attributes:
        data: Dictionary of datasets, where each dataset is a list of dictionaries
        dictionary: Dictionary of data dictionaries, where each dictionary describes a dataset's columns
        question: Business question to analyze
        error_message: Optional error from previous attempt
        failed_code: Optional code that failed in previous attempt
    """

    data: Dict[str, List[Dict[str, Any]]]
    dictionary: Dict[str, List[Dict[str, str]]]
    question: str
    error_message: Optional[str] = None
    failed_code: Optional[str] = None

    @validator("data")
    def validate_data(cls, v):
        if not isinstance(v, dict):
            raise ValueError("Input data must be a dictionary of datasets")
        if not all(isinstance(dataset, list) for dataset in v.values()):
            raise ValueError("Each dataset must be a list of dictionaries")
        return v

    @validator("dictionary")
    def validate_dictionary(cls, v):
        if not isinstance(v, dict):
            raise ValueError("Dictionary must be a dictionary of dataset descriptions")
        if not all(isinstance(desc, list) for desc in v.values()):
            raise ValueError("Each dataset description must be a list")
        return v


class PythonAnalysisRequest(BaseModel):
    data: List[Dict[str, Any]]  # Changed from DataFrame to List of JSON objects
    dictionary: List[
        Dict[str, Any]
    ]  # Changed from DataFrame to List of dictionary entries
    question: str
    error_message: Optional[str] = None
    failed_code: Optional[str] = None

    @validator("data")
    def validate_data(cls, v):
        if not isinstance(v, list):
            raise ValueError("Input data must be a list of JSON objects")
        if len(v) == 0:
            raise ValueError("Data cannot be empty")
        return v

    @validator("dictionary")
    def validate_dictionary(cls, v):
        if not isinstance(v, list):
            raise ValueError("Dictionary must be a list")
        required_keys = {"column", "description", "data_type"}
        if not all(required_keys.issubset(d.keys()) for d in v):
            raise ValueError(f"Dictionary entries must contain keys: {required_keys}")
        return v

    @validator("question")
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()


@app.post(
    "/get_python_analysis_code",
    response_model=Dict[str, str],
    summary="Generate Python analysis code",
    description="""
    Generate Python code to analyze data based on a business question.
    
    The endpoint:
    - Interprets the business question
    - Generates appropriate analysis code
    - Validates code safety
    - Provides execution context
    
    Returns validated Python code with description.
    """,
    response_description="Generated Python code with description",
    tags=["Code Generation"],
)
async def get_python_analysis_code(request: RunAnalysisRequest) -> Dict[str, str]:
    """
    Generate Python analysis code based on JSON data and question.

    Parameters:
    - request: RunAnalysisRequest containing data and question

    Returns:
    - Dictionary containing generated code and description

    Raises:
    - HTTPException: If code generation fails
    """
    try:
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
            all_shapes.append(
                f"{dataset_name}: {df.shape[0]} rows x {df.shape[1]} columns"
            )
            # Limit sample to 10 rows
            sample_df = df.head(10)
            all_samples.append(f"{dataset_name}:\n{sample_df.to_string()}")

        shape_info = "\n".join(all_shapes)
        sample_data = "\n\n".join(all_samples)

        # Create messages for OpenAI
        messages = [
            {"role": "system", "content": prompts.SYSTEM_PROMPT_PYTHON_ANALYST},
            {"role": "user", "content": f"Business Question: {request.question}"},
            {"role": "user", "content": f"Data Shapes:\n{shape_info}"},
            {"role": "user", "content": f"Sample Data:\n{sample_data}"},
            {
                "role": "user",
                "content": f"Data Dictionary:\n{json.dumps(dictionary_data)}",
            },
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
            model="gpt-4o",
            temperature=0.1,
            messages=messages,
            response_format={"type": "json_object"},
            stream=False,
        )

        # Parse and return the response
        response = json.loads(completion.choices[0].message.content)

        return {
            "code": response.get("code", ""),
            "description": response.get("description", ""),
        }

    except Exception as e:
        logging.error(f"Error generating analysis code: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


class ChartRequest(BaseModel):
    """Request model for charts endpoint

    Attributes:
        data: List of dictionaries representing a single dataset
        question: Business question to visualize
        error_message: Optional error from previous attempt
        failed_code: Optional code that failed in previous attempt
    """

    data: List[Dict[str, Any]]
    question: str
    error_message: Optional[str] = None
    failed_code: Optional[str] = None

    @validator("data")
    def validate_data(cls, v):
        if not isinstance(v, list):
            raise ValueError("Input data must be a list of dictionaries")
        if not all(isinstance(record, dict) for record in v):
            raise ValueError("Each record must be a dictionary")
        return v


@dataclass
class ChartGenerationResult:
    """Container for chart generation results"""

    fig1: go.Figure
    fig2: go.Figure
    code: str
    validation: Dict[str, Any]
    metadata: Dict[str, Any]
    attempts: int
    validation_errors: List[str]
    execution_errors: List[Dict[str, Any]]
    code_history: List[Dict[str, Any]]


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


class RunChartsRequest(BaseModel):
    """Request model for charts endpoint

    Attributes:
        data: List of dictionaries representing a single dataset
        question: Business question to visualize
        error_message: Optional error from previous attempt
        failed_code: Optional code that failed in previous attempt
    """

    data: List[Dict[str, Any]]
    question: str
    error_message: Optional[str] = None
    failed_code: Optional[str] = None

    @validator("data")
    def validate_data(cls, v):
        if not isinstance(v, list):
            raise ValueError("Input data must be a list of dictionaries")
        if not all(isinstance(record, dict) for record in v):
            raise ValueError("Each record must be a dictionary")
        return v


@app.post("/run_charts")
async def run_charts(request: RunChartsRequest) -> Dict[str, Any]:
    """
    Generate and execute chart code with validation.
    """
    try:
        # Convert JSON to DataFrame
        df = pd.DataFrame(request.data)
        if df.empty:
            raise ValueError("Input DataFrame cannot be empty")

        # Generate metadata about the dataframe
        metadata = {
            "metadata_shape": list(df.shape),
            "metadata_describe": json.loads(df.describe(include="all").to_json()),
            "metadata_dtypes": json.loads(df.dtypes.astype(str).to_json()),
        }

        max_attempts = 3
        attempt = 0
        last_error = None
        last_failed_code = None

        while attempt < max_attempts:
            try:
                # Generate charts with retry logic
                result = await create_charts(
                    df=df.head(25),
                    question=request.question,
                    metadata=metadata,
                    error_message=last_error,
                    failed_code=last_failed_code,
                )

                # Convert figures to base64
                fig1_base64 = figure_to_base64(result.fig1) if result.fig1 else None
                fig2_base64 = figure_to_base64(result.fig2) if result.fig2 else None

                return {
                    "fig1": result.fig1,
                    "fig2": result.fig2,
                    "fig1_base64": fig1_base64,
                    "fig2_base64": fig2_base64,
                    "code": result.code,
                    "metadata": {
                        **result.metadata,
                        "dataframe_metadata": metadata,
                        "validation": result.validation,
                        "attempts": result.attempts,
                        "validation_errors": result.validation_errors,
                        "execution_errors": result.execution_errors,
                        "code_history": result.code_history,
                        "performance": {
                            "memory_usage": get_memory_usage(),
                            "total_time": (
                                datetime.fromisoformat(result.metadata["timestamp"])
                                - datetime.fromisoformat(
                                    result.code_history[0]["timestamp"]
                                )
                            ).total_seconds(),
                        },
                    },
                }

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
                        "code_history": (
                            result.code_history if "result" in locals() else []
                        ),
                        "attempts": attempt,
                        "timestamp": datetime.now().isoformat(),
                    }
                    raise HTTPException(
                        status_code=500,
                        detail={"error": str(e), "context": error_context},
                    )

    except ValueError as e:
        # Only catch and re-raise validation errors without retrying
        raise HTTPException(status_code=422, detail=str(e))


class BusinessAnalysisRequest(BaseModel):
    data: List[Dict[str, Any]]  # JSON data
    dictionary: List[Dict[str, str]]  # JSON dictionary
    question: str

    @validator("data")
    def validate_data(cls, v):
        if not isinstance(v, list):
            raise ValueError("Input data must be a list of JSON objects")
        if len(v) == 0:
            raise ValueError("Data cannot be empty")
        return v

    @validator("dictionary")
    def validate_dictionary(cls, v):
        if not isinstance(v, list):
            raise ValueError("Dictionary must be a list of objects")
        required_keys = {"column", "description", "data_type"}
        if not all(required_keys.issubset(d.keys()) for d in v):
            raise ValueError(f"Dictionary entries must contain keys: {required_keys}")
        return v

    @validator("question")
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()


@app.post(
    "/get_business_analysis",
    response_model=Dict[str, Any],
    summary="Generate business analysis",
    description="""
    Generate comprehensive business analysis based on data and question.
    
    The endpoint provides:
    - Bottom line answer
    - Additional insights
    - Follow-up questions
    - Analysis context
    
    Returns structured analysis response.
    """,
    response_description="Business analysis with insights",
    tags=["Business Analysis"],
)
async def get_business_analysis(request: BusinessAnalysisRequest) -> Dict[str, Any]:
    """
    Generate business analysis based on data and question.

    Parameters:
    - request: BusinessAnalysisRequest containing data and question

    Returns:
    - Dictionary containing analysis components

    Raises:
    - HTTPException: If analysis generation fails
    """
    try:
        # Convert JSON data to DataFrame for analysis
        df = pd.DataFrame(request.data)

        # Get first 1000 rows as CSV with quoted values for context
        df_csv = df.head(750).to_csv(index=False, quoting=1)

        messages = [
            {"role": "system", "content": prompts.SYSTEM_PROMPT_BUSINESS_ANALYSIS},
            {"role": "user", "content": f"Business Question: {request.question}"},
            {"role": "user", "content": f"Analyzed Data:\n{df_csv}"},
            {
                "role": "user",
                "content": f"Data Dictionary:\n{json.dumps(request.dictionary)}",
            },
        ]

        # Get non-streaming response from OpenAI
        completion = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.1,
            messages=messages,
            response_format={"type": "json_object"},
            stream=False,
        )

        # Parse the response
        response = json.loads(completion.choices[0].message.content)

        # Ensure all response fields are present
        result = {
            "bottom_line": response.get("bottom_line", ""),
            "additional_insights": response.get("additional_insights", ""),
            "follow_up_questions": response.get("follow_up_questions", []),
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "question": request.question,
                "rows_analyzed": len(df),
                "columns_analyzed": len(df.columns),
            },
        }

        return result

    except Exception as e:
        logging.error(f"Error generating business analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


class ChatRequest(BaseModel):
    """Request model for chat history processing

    Attributes:
        messages: List of dictionaries containing chat messages
                 Each message must have 'role' and 'content' fields
                 Role must be one of: 'user', 'assistant', 'system'
    """

    messages: List[Dict[str, str]]

    @validator("messages")
    def validate_messages(cls, v):
        if not v:
            raise ValueError("Messages list cannot be empty")

        for msg in v:
            if "role" not in msg or "content" not in msg:
                raise ValueError("Each message must have 'role' and 'content' fields")
            if msg["role"] not in ["user", "assistant", "system"]:
                raise ValueError(
                    "Message role must be 'user', 'assistant', or 'system'"
                )
            if not msg["content"].strip():
                raise ValueError("Message content cannot be empty")

        return v


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


@app.post(
    "/chat",
    response_model=Dict[str, Any],
    summary="Process chat history",
    description="""
    Process chat history and return enhanced message.
    
    The endpoint:
    - Analyzes conversation context
    - Enhances user messages
    - Maintains conversation coherence
    
    Returns enhanced message response.
    """,
    response_description="Enhanced chat message",
    tags=["Chat Processing"],
)
async def chat(request: ChatRequest) -> Dict[str, Any]:
    """
    Process chat history and return enhanced message.

    Parameters:
    - request: ChatRequest containing chat messages

    Returns:
    - Dictionary containing enhanced message

    Raises:
    - HTTPException: If chat processing fails
    """
    try:
        response = await process_chat(request.messages)
        return response

    except Exception as e:
        logging.error(f"Error processing chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@dataclass
class AnalysisResult:
    """Container for analysis results"""

    data: pd.DataFrame
    stdout: str
    stderr: str
    code: str
    execution_time: float
    memory_usage: Dict[str, float]


class CodeValidator:
    """Validates Python code for safety and correctness"""

    ALLOWED_MODULES = {"pandas", "numpy", "scipy", "statsmodels", "sklearn"}

    @staticmethod
    def validate_imports(code: str) -> Tuple[bool, str]:
        """Check if code only imports allowed modules"""
        try:
            tree = ast.parse(code)
            imports = []

            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        imports.extend(n.name.split(".")[0] for n in node.names)
                    else:
                        imports.append(node.module.split(".")[0])

            illegal_imports = set(imports) - CodeValidator.ALLOWED_MODULES
            if illegal_imports:
                return False, f"Illegal imports detected: {illegal_imports}"

            return True, "Validation passed"

        except SyntaxError as e:
            return False, f"Syntax error in code: {str(e)}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"


@dataclass
class GeneratedCode:
    """Container for generated analysis code"""

    code: str
    description: str
    estimated_complexity: str
    validation_result: Tuple[bool, str]


@dataclass
class CodeGenerationResult:
    """Container for code generation results"""

    code: str
    description: str
    validation: Dict[str, Any]
    metadata: Dict[str, Any]
    attempts: int
    validation_errors: List[str]


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


# Update original endpoint to use new retry logic
@app.post("/run_analysis")
async def run_analysis(request: RunAnalysisRequest) -> Dict[str, Any]:
    """
    Execute complete data analysis workflow with integrated generation and execution retry logic.
    """
    max_attempts = 5  # Single attempt counter for the generate-execute cycle
    attempts = 0
    error_history = []

    try:
        # Input validation
        if not request.data:
            raise HTTPException(status_code=422, detail="Input data cannot be empty")

        # Convert JSON to DataFrames dictionary
        dataframes = {}
        for dataset_name, dataset_records in request.data.items():
            if dataset_records:
                df = pd.DataFrame(dataset_records)
                dataframes[dataset_name] = df
            else:
                dataframes[dataset_name] = pd.DataFrame()

        while attempts < max_attempts:
            attempts += 1

            try:
                # Update request with error context if available
                if error_history:
                    request.error_message = error_history[-1]["error"]
                    request.failed_code = error_history[-1]["code"]

                # Generate code
                code_result = await generate_analysis_code(request)

                # Validate the generated code
                if not code_result.validation["is_valid"]:
                    error_info = {
                        "attempt": attempts,
                        "error": code_result.validation["message"],
                        "code": code_result.code,
                        "timestamp": datetime.now().isoformat(),
                        "type": "validation_error",
                    }
                    error_history.append(error_info)
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

                    result = namespace["analyze_data"](dataframes)

                    if not isinstance(result, (pd.DataFrame, list, dict)):
                        result = pd.DataFrame(result)

                # If we get here, execution was successful
                return {
                    "status": "success",
                    "code": code_result.code,
                    "data": (
                        result.to_dict("records")
                        if isinstance(result, pd.DataFrame)
                        else result
                    ),
                    "metadata": {
                        "execution_time": datetime.now().isoformat(),
                        "attempts": attempts,
                        "error_history": error_history,
                        "execution_details": {
                            "stdout": stdout.getvalue(),
                            "stderr": stderr.getvalue(),
                        },
                        "datasets_analyzed": len(dataframes),
                        "total_rows_analyzed": sum(
                            len(df) for df in dataframes.values() if not df.empty
                        ),
                        "total_columns_analyzed": sum(
                            len(df.columns)
                            for df in dataframes.values()
                            if not df.empty
                        ),
                    },
                }

            except Exception as e:
                # Classify and record the error
                error_info = {
                    "attempt": attempts,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "code": code_result.code if "code_result" in locals() else None,
                    "stdout": stdout.getvalue() if "stdout" in locals() else "",
                    "stderr": stderr.getvalue() if "stderr" in locals() else "",
                    "timestamp": datetime.now().isoformat(),
                }
                error_history.append(error_info)

                if attempts >= max_attempts:
                    return {
                        "status": "failed",
                        "error_history": error_history,
                        "last_error": str(e),
                        "suggestions": "Consider reformulating the question or checking data quality",
                    }

    except Exception as e:
        logging.error(f"Error in run_analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def is_date_column(series: pd.Series) -> bool:
    """Check if a pandas Series likely contains date values

    Args:
        series: pandas Series to check

    Returns:
        bool: True if series likely contains dates, False otherwise
    """
    # Skip if series is empty
    if series.empty:
        return False

    # Get non-null values
    sample = series.dropna().head(100)
    if sample.empty:
        return False

    # Common date patterns
    date_patterns = [
        r"\d{4}-\d{2}-\d{2}",  # YYYY-MM-DD
        r"\d{2}-\d{2}-\d{4}",  # DD-MM-YYYY or MM-DD-YYYY
        r"\d{2}/\d{2}/\d{4}",  # DD/MM/YYYY or MM/DD/YYYY
        r"\d{4}/\d{2}/\d{2}",  # YYYY/MM/DD
        r"\d{2}\.\d{2}\.\d{4}",  # DD.MM.YYYY or MM.DD.YYYY
        r"\d{4}\.\d{2}\.\d{2}",  # YYYY.MM.DD
    ]

    # Check if any values match date patterns
    pattern = "|".join(date_patterns)
    matches = sample.astype(str).str.match(pattern)
    match_ratio = matches.mean() if not matches.empty else 0

    return match_ratio > 0.8  # Return True if >80% of values match date patterns


def convert_to_datetime(value: Any, column: str) -> Optional[datetime]:
    """Convert a value to datetime with flexible format handling

    Args:
        value: Value to convert
        column: Column name for error reporting

    Returns:
        datetime or None if conversion fails
    """
    if pd.isna(value):
        return None

    try:
        # First try pandas to_datetime with coerce
        result = pd.to_datetime(value, infer_datetime_format=True)
        # Convert Timestamp to datetime
        if isinstance(result, pd.Timestamp):
            return result.to_pydatetime()
        return result
    except:
        try:
            # Try dateutil parser as fallback
            from dateutil import parser

            parsed = parser.parse(str(value))
            # Ensure we return a datetime object
            return parsed.replace(tzinfo=None)
        except:
            return None


def convert_datetime_series(series: pd.Series) -> pd.Series:
    """Convert a series of values to datetime using vectorized operations

    Args:
        series: pandas Series to convert

    Returns:
        pandas Series with ISO format datetime strings
    """
    try:
        # Convert to datetime
        result = pd.to_datetime(series, infer_datetime_format=True, errors="coerce")
        # Convert to ISO format strings
        return result.dt.strftime("%Y-%m-%dT%H:%M:%S")
    except Exception as e:
        logging.warning(f"Initial datetime conversion failed: {str(e)}")
        return series


class AnalysisError(Exception):
    def __init__(self, message: str, error_type: str, code: str = None):
        self.message = message
        self.error_type = error_type
        self.code = code
        super().__init__(self.message)


def classify_error(error: Exception, code: str = None) -> AnalysisError:
    """Classify the type of error to inform retry strategy"""
    if isinstance(error, SyntaxError):
        return AnalysisError(str(error), "syntax", code)
    elif isinstance(error, NameError):
        return AnalysisError(str(error), "undefined_variable", code)
    elif isinstance(error, ValueError):
        return AnalysisError(str(error), "value_error", code)
    # ... add more classifications
    return AnalysisError(str(error), "unknown", code)
