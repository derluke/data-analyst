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

import sys
from typing import Any, Callable, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

sys.path.append("..")

from utils.api import (
    cleanse_dataframes,
    download_catalog_datasets,
    get_business_analysis,
    get_dictionary,
    list_catalog_datasets,
    rephrase_message,
    run_analysis,
    run_charts,
    run_snowflake_analysis,
    suggest_questions,
)
from utils.schema import (
    AiCatalogDataset,
    BusinessAnalysisRequest,
    BusinessAnalysisResult,
    ChatRequest,
    CleanseRequest,
    CleanseResult,
    DataDictionariesAndMetadata,
    DictionaryRequest,
    QuestionSuggestions,
    RunAnalysisRequest,
    RunAnalysisResult,
    RunChartsRequest,
    RunChartsResult,
    SnowflakeAnalysisRequest,
    SnowflakeAnalysisResult,
)
from utils.snowflake_helpers import get_snowflake_data, get_snowflake_tables

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


@app.get("/list_catalog_datasets")
async def list_catalog_datasets_endpoint(limit: int = 100) -> list[AiCatalogDataset]:
    return list_catalog_datasets(limit)


@app.get("/download_catalog_datasets")
async def download_catalog_datasets_endpoint(
    dataset_ids: list[str],
) -> dict[str, dict[str, Any]]:
    return download_catalog_datasets(*dataset_ids)


@app.get("/get_snowflake_tables")
async def get_snowflake_tables_endpoint() -> list[str]:
    return get_snowflake_tables()


@app.get("/get_snowflake_data")
async def get_snowflake_data_endpoint(
    table_names: list[str], sample_size: int = 5000
) -> dict[str, list[dict[str, Any]]]:
    return get_snowflake_data(*table_names, sample_size=sample_size)


@app.post("/cleanse_dataframes")
async def cleanse_dataframes_endpoint(
    request: CleanseRequest,
    progress_callback: Optional[Callable[[str, int], None]] = None,
) -> CleanseResult:
    return cleanse_dataframes(request, progress_callback=progress_callback)


@app.post("/get_dictionary")
async def get_dictionary_endpoint(
    request: DictionaryRequest,
) -> DataDictionariesAndMetadata:
    return get_dictionary(request)


@app.post("/suggest_questions")
async def suggest_questions_endpoint(request: DictionaryRequest) -> QuestionSuggestions:
    return suggest_questions(request)


@app.post("/run_charts")
async def run_charts_endpoint(request: RunChartsRequest) -> RunChartsResult:
    return run_charts(request)


@app.post("/get_business_analysis")
async def get_business_analysis_endpoint(
    request: BusinessAnalysisRequest,
) -> BusinessAnalysisResult:
    return get_business_analysis(request)


@app.post("/chat")
async def rephrase_message_endpoint(request: ChatRequest) -> dict[str, Any]:
    return rephrase_message(request)


@app.post("/run_analysis")
async def run_analysis_endpoint(request: RunAnalysisRequest) -> RunAnalysisResult:
    return run_analysis(request)


@app.post("/run_snowflake_analysis")
async def run_snowflake_analysis_endpoint(
    request: SnowflakeAnalysisRequest, max_attempts: int = 3, timeout: int = 300
) -> SnowflakeAnalysisResult:
    return run_snowflake_analysis(
        request=request, max_attempts=max_attempts, timeout=timeout
    )
