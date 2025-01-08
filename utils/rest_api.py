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

import sys
from typing import Any, Callable, Dict, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

sys.path.append("..")

from utils.api import (
    cleanse_dataframes,
    get_business_analysis,
    get_catalog_datasets,
    get_dictionary,
    process_chat,
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


@app.get("/get_catalog_datasets")
async def get_catalog_datasets_endpoint(limit: int = 100) -> list[AiCatalogDataset]:
    return get_catalog_datasets(limit)


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
async def process_chat_endpoint(request: ChatRequest) -> Dict[str, Any]:
    return process_chat(request)


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
