import io
import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

sys.path.append("..")

from utils import prompts
from utils.api import (
    client,
    create_charts,
    create_snowflake_connection,
    execute_snowflake_query,
    figure_to_base64,
    generate_analysis_code,
    generate_python_analysis_code,
    generate_question_suggestions,
    get_memory_usage,
    process_chat,
    process_dataset,
)
from utils.datetime_helpers import convert_datetime_series, is_date_column
from utils.errors import EmptyDataError
from utils.schema import (
    BusinessAnalysisRequest,
    ChatRequest,
    CleanseRequest,
    CleanseResponse,
    CleansingReport,
    DataDictionariesAndMetadata,
    DataDictionary,
    DataDictionaryMetadata,
    DatasetOutput,
    DictionaryRequest,
    RunAnalysisRequest,
    RunChartsRequest,
    SnowflakeAnalysisRequest,
)

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
    version="1.0.0",
    contact={"name": "API Support", "email": "support@example.com"},
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)

MODEL_MODE = "openai"  # "openai", "gemini", anthropic

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

                # Clean column names - only remove leading/trailing whitespace and consecutive spaces
                original_columns = df.columns.tolist()
                df.columns = [re.sub(r"\s+", " ", col.strip()) for col in df.columns]
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
async def get_dictionary(request: DictionaryRequest) -> DataDictionariesAndMetadata:
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

        metadata = DataDictionaryMetadata(
            total_datasets=len(request.datasets),
            processing_start=datetime.now().isoformat(),
            batch_times=[],
            errors=[],
        )

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
                    metadata.batch_times.append(result.batch_time)
                    logging.info(
                        f"Processed dataset {dataset_name} with {len(result.dictionary)} entries"
                    )
                except Exception as e:
                    error_msg = f"Error processing dataset {dataset_name}: {str(e)}"
                    logging.error(error_msg)
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

        logging.info(f"Returning dictionary response with {len(results)} results")
        return response

    except Exception as e:
        logging.error(f"Error in get_dictionary: {str(e)}")
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
        return generate_python_analysis_code(request)

    except Exception as e:
        logging.error(f"Error generating analysis code: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


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

        # Create messages for OpenAI
        messages = [
            {"role": "system", "content": prompts.SYSTEM_PROMPT_BUSINESS_ANALYSIS},
            {"role": "user", "content": f"Business Question: {request.question}"},
            {"role": "user", "content": f"Analyzed Data:\n{df_csv}"},
            {
                "role": "user",
                "content": f"Data Dictionary:\n{json.dumps(request.dictionary)}",
            },
        ]

        # Get response based on model mode
        if MODEL_MODE == "openai":
            completion = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.1,
                messages=messages,
                response_format={"type": "json_object"},
                stream=False,
            )
            response = json.loads(completion.choices[0].message.content)
        elif MODEL_MODE in ["gemini", "anthropic"]:
            completion = client.chat.completions.create(
                model="gemini-1.5-pro",
                messages=messages,  # or appropriate model name
            )
            # Extract JSON from response by looking for ```json blocks
            content = completion.choices[0].message.content
            json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
            if json_match:
                response = json.loads(json_match.group(1))
            else:
                raise ValueError("No JSON block found in model response")

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


@app.post(
    "/get_snowflake_analysis_code",
    response_model=Dict[str, str],
    summary="Generate Snowflake SQL analysis code",
    description="""
    Generate Snowflake SQL code to analyze data based on a business question.
    
    The endpoint:
    - Interprets the business question
    - Generates appropriate Snowflake SQL code
    - Validates code safety
    - Provides execution context
    
    Returns validated Snowflake SQL code with description.
    """,
    response_description="Generated Snowflake SQL code with description",
    tags=["Code Generation"],
)
async def get_snowflake_analysis_code(
    request: SnowflakeAnalysisRequest,
) -> Dict[str, str]:
    """
    Generate Snowflake SQL analysis code based on data samples and question.

    Parameters:
    - request: SnowflakeAnalysisRequest containing data samples and question

    Returns:
    - Dictionary containing generated code and description

    Raises:
    - HTTPException: If code generation fails
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
        messages = [
            {
                "role": "system",
                "content": prompts.SYSTEM_PROMPT_SNOWFLAKE.format(
                    warehouse=request.warehouse,
                    database=request.database,
                    schema=request.schema,
                ),
            },
            {"role": "user", "content": f"Business Question: {request.question}"},
            {"role": "user", "content": f"Sample Data:\n{chr(10).join(all_samples)}"},
            {
                "role": "user",
                "content": f"Data Dictionary:\n{json.dumps(all_tables_info)}",
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
        if MODEL_MODE == "openai":
            completion = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.1,
                messages=messages,
                response_format={"type": "json_object"},
                stream=False,
            )
            response = json.loads(completion.choices[0].message.content)
        elif MODEL_MODE in ["gemini", "anthropic"]:
            completion = client.chat.completions.create(
                model="gemini-1.5-pro",
                messages=messages,  # or appropriate model name
            )
            # Extract JSON from response by looking for ```json blocks
            content = completion.choices[0].message.content
            json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
            if json_match:
                response = json.loads(json_match.group(1))
            else:
                raise ValueError("No JSON block found in model response")

        return {
            "code": response.get("code", ""),
            "description": response.get("description", ""),
        }

    except Exception as e:
        logging.error(f"Error generating Snowflake analysis code: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run_snowflake_analysis")
async def run_snowflake_analysis(
    request: SnowflakeAnalysisRequest, max_attempts: int = 3, timeout: int = 300
) -> Dict[str, Any]:
    """Execute Snowflake analysis with retry logic and error handling"""
    attempts = 0
    error_history = []
    conn = None
    last_generated_code = None
    start_time = time.time()

    try:
        # Validate input
        if not request.data:
            raise HTTPException(status_code=422, detail="Input data cannot be empty")

        # Create Snowflake connection
        conn = create_snowflake_connection(
            warehouse=request.warehouse,
            database=request.database,
            schema=request.schema,
        )

        while attempts < max_attempts:
            attempts += 1

            try:
                # Update request with error context if available
                if error_history:
                    request.error_message = error_history[-1]["error"]
                    request.failed_code = error_history[-1]["code"]

                # Generate SQL code
                code_result = await get_snowflake_analysis_code(request)
                sql_code = code_result["code"]
                last_generated_code = sql_code

                # Execute query with timeout
                results, query_metadata = execute_snowflake_query(
                    conn=conn, query=sql_code, timeout=timeout
                )

                # Return successful result
                return {
                    "status": "success",
                    "code": sql_code,
                    "description": code_result["description"],
                    "data": results,
                    "metadata": {
                        "execution_time": time.time() - start_time,
                        "attempts": attempts,
                        "error_history": error_history,
                        "query_metadata": query_metadata,
                        "tables_analyzed": len(request.data),
                        "total_sample_rows": sum(
                            len(samples) for samples in request.data.values()
                        ),
                        "performance": {
                            "memory_usage": get_memory_usage(),
                            "total_time": time.time() - start_time,
                        },
                    },
                }

            except Exception as e:
                # Classify and record error
                error_info = {
                    "attempt": attempts,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "code": sql_code if "sql_code" in locals() else None,
                    "timestamp": datetime.now().isoformat(),
                    "memory_usage": get_memory_usage(),
                }
                error_history.append(error_info)

                # On last attempt, return detailed error response
                if attempts >= max_attempts:
                    return {
                        "status": "failed",
                        "error": str(e),
                        "last_generated_code": last_generated_code,
                        "error_history": error_history,
                        "metadata": {
                            "total_attempts": attempts,
                            "total_time": time.time() - start_time,
                            "performance": {
                                "memory_usage": get_memory_usage(),
                                "error_timestamp": datetime.now().isoformat(),
                            },
                        },
                        "suggestions": "Consider reformulating the question or checking data access permissions",
                    }

                # Exponential backoff between attempts
                time.sleep(min(2**attempts, 10))

    except Exception as e:
        logging.error(f"Error in run_snowflake_analysis: {str(e)}")
        return {
            "status": "failed",
            "error": str(e),
            "last_generated_code": last_generated_code,
            "error_history": error_history,
            "metadata": {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "attempts": attempts,
                "total_time": time.time() - start_time,
                "performance": {
                    "memory_usage": get_memory_usage(),
                    "error_timestamp": datetime.now().isoformat(),
                },
            },
        }
    finally:
        if conn:
            conn.close()
