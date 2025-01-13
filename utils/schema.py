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
from typing import Any, Type, Union

import plotly.graph_objects as go
from pydantic import BaseModel, ConfigDict, field_validator


class LLMDeploymentSettings(BaseModel):
    target_feature_name: str = "resultText"
    prompt_feature_name: str = "promptText"


class AiCatalogDataset(BaseModel):
    id: str
    name: str
    created: str
    size: str


class ChatAgentDeploymentSettings(BaseModel):
    target_feature_name: str = "content"
    prompt_feature_name: str = "promptText"
    request_timeout: int = 60
    max_retries: int = 1
    temperature: int = 0


class DatasetInput(BaseModel):
    name: str
    data: list[dict[str, Any]]


class CleansingReport(BaseModel):
    columns_cleaned: list[str]
    value_counts: dict[str, Any]
    errors: list[str]
    warnings: list[str]


class DatasetOutput(BaseModel):
    name: str
    data: list[dict[str, Any]]
    cleaning_report: CleansingReport


class CleanseResult(BaseModel):
    datasets: list[DatasetOutput]
    metadata: dict[str, Any]


class DataDictionaryColumn(BaseModel):
    data_type: str
    column: str
    description: str


class DataDictionary(BaseModel):
    name: str
    dictionary: list[DataDictionaryColumn]
    cache_hit: bool
    batch_time: float


class DataDictionaryMetadata(BaseModel):
    total_datasets: int
    processing_start: str
    batch_times: list[float]
    errors: list[str]
    processing_end: str | None = None
    total_time: float | None = None


class DataDictionariesAndMetadata(BaseModel):
    metadata: DataDictionaryMetadata
    dictionaries: list[DataDictionary]


class DictionaryResult(BaseModel):
    """Validates LLM responses for data dictionary generation

    Attributes:
        columns: List of column names
        descriptions: List of column descriptions

    Raises:
        ValueError: If validation fails
    """

    columns: list[str]
    descriptions: list[str]

    @field_validator("descriptions")
    @classmethod
    def validate_descriptions(
        cls: Type["DictionaryResult"], v: Any, values: Any
    ) -> Any:
        # Check if columns exists in values
        if "columns" not in values.data:
            raise ValueError("Columns must be provided before descriptions")

        # Check if lengths match
        if len(v) != len(values.data["columns"]):
            raise ValueError(
                f"Number of descriptions ({len(v)}) must match number of columns ({len(values['columns'])})"
            )

        # Validate each description
        for desc in v:
            if not desc or not isinstance(desc, str):
                raise ValueError("Each description must be a non-empty string")
            if len(desc.strip()) < 10:
                raise ValueError("Descriptions must be at least 10 characters long")

        return v

    @field_validator("columns")
    @classmethod
    def validate_columns(cls: Type["DictionaryResult"], v: Any) -> Any:
        if not v:
            raise ValueError("Columns list cannot be empty")

        # Check for duplicates
        if len(v) != len(set(v)):
            raise ValueError("Duplicate column names are not allowed")

        # Validate each column name
        for col in v:
            if not col or not isinstance(col, str):
                raise ValueError("Each column name must be a non-empty string")

        return v

    def to_dict(self) -> dict[str, str]:
        """Convert columns and descriptions to dictionary format

        Returns:
            Dict mapping column names to their descriptions
        """
        return dict(zip(self.columns, self.descriptions))


class RunAnalysisRequest(BaseModel):
    """Request model for analysis endpoint

    Attributes:
        data: Dictionary of datasets, where each dataset is a list of dictionaries
        dictionary: Dictionary of data dictionaries, where each dictionary describes a dataset's columns
        question: Business question to analyze
        error_message: Optional error from previous attempt
        failed_code: Optional code that failed in previous attempt
    """

    data: dict[str, list[dict[str, Any]]]
    dictionary: dict[
        str, list[dict[str, Union[str, dict[str, str]]]]
    ]  # Allow dictionary values for description
    question: str
    error_message: str | None = None
    failed_code: str | None = None

    @field_validator("data")
    @classmethod
    def validate_data(cls: Type["RunAnalysisRequest"], v: Any) -> Any:
        """Validate that the input data is a dictionary of datasets"""
        if not isinstance(v, dict):
            raise ValueError("Input data must be a dictionary of datasets")
        if not all(isinstance(dataset, list) for dataset in v.values()):
            raise ValueError("Each dataset must be a list of dictionaries")
        return v

    @field_validator("dictionary")
    @classmethod
    def validate_dictionary(cls: Type["RunAnalysisRequest"], v: Any) -> Any:
        """Validate that the input dictionary is a dictionary of dataset descriptions"""
        if not isinstance(v, dict):
            raise ValueError("dictionary must be a dictionary of dataset descriptions")

        # Process dictionary values to ensure descriptions are strings
        processed = {}
        for dataset_name, descriptions in v.items():
            processed_descriptions = []
            for desc in descriptions:
                if not isinstance(desc, dict):
                    raise ValueError("Each description must be a dictionary")

                # Convert any dictionary values in description to strings
                processed_desc = desc.copy()
                if "description" in desc and isinstance(desc["description"], dict):
                    # Join key-value pairs from the description dictionary
                    desc_str = "; ".join(
                        f"{k}: {v}" for k, v in desc["description"].items()
                    )
                    processed_desc["description"] = desc_str

                processed_descriptions.append(processed_desc)
            processed[dataset_name] = processed_descriptions

        return processed

    model_config = ConfigDict(arbitrary_types_allowed=True)


class RunAnanlysisResultMetadata(BaseModel):
    timestamp: str
    attempts: int
    error_history: list[AnalysisError]
    stdout: str | None = None
    stderr: str | None = None
    datasets_analyzed: int | None = None
    total_rows_analyzed: int | None = None
    total_columns_analyzed: int | None = None


class RunAnalysisResult(BaseModel):
    status: str
    metadata: SnowflakeAnalysisMetadata | RunAnanlysisResultMetadata
    data: list[dict[str, Any]] | None = None
    code: str | None = None
    suggestions: str | None = None


class ChartRequest(BaseModel):
    """Request model for charts endpoint

    Attributes:
        data: list of dictionaries representing a single dataset
        question: Business question to visualize
        error_message: Optional error from previous attempt
        failed_code: Optional code that failed in previous attempt
    """

    data: list[dict[str, Any]]
    question: str
    error_message: str | None = None
    failed_code: str | None = None

    @field_validator("data")
    @classmethod
    def validate_data(cls: Type["ChartRequest"], v: Any) -> Any:
        """Validate that the data is a list of dictionaries"""
        if not isinstance(v, list):
            raise ValueError("Input data must be a list of dictionaries")
        if not all(isinstance(record, dict) for record in v):
            raise ValueError("Each record must be a dictionary")
        return v


class ChartGenerationResult(BaseModel):
    fig1: go.Figure
    fig2: go.Figure
    code: str
    validation: ValidationMessage
    metadata: ChartGenerationMetadata
    attempts: int
    validation_errors: list[ChartValidationError]
    execution_errors: list[ChartExecutionError]
    code_history: list[ChartCodeHistory]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class RunChartsRequest(BaseModel):
    """Request model for charts endpoint

    Attributes:
        data: list of dictionaries representing a single dataset
        question: Business question to visualize
        error_message: Optional error from previous attempt
        failed_code: Optional code that failed in previous attempt
    """

    data: list[dict[Union[str, int], Any]]  # Allow both string and integer keys
    question: str
    error_message: str | None = None
    failed_code: str | None = None

    @field_validator("data")
    @classmethod
    def validate_data(cls: Type["RunChartsRequest"], v: Any) -> list[dict[str, Any]]:
        """Validate that the input data is a list of dictionaries and convert numeric keys to strings."""
        if not isinstance(v, list):
            raise ValueError("Input data must be a list of dictionaries")
        if not all(isinstance(record, dict) for record in v):
            raise ValueError("Each record must be a dictionary")

        # Convert numeric keys to strings in nested dictionaries
        def convert_numeric_keys(d: Any) -> Any:
            """Recursively convert numeric keys to strings."""
            if not isinstance(d, dict):
                return d
            return {
                str(k): convert_numeric_keys(v) if isinstance(v, dict) else v
                for k, v in d.items()
            }

        # Convert all records
        converted = [convert_numeric_keys(record) for record in v]

        # Ensure all keys are strings after conversion
        for record in converted:
            if not all(isinstance(k, str) for k in record.keys()):
                raise ValueError("All dictionary keys must be strings after conversion")

        return converted

    model_config = ConfigDict(arbitrary_types_allowed=True)


class RunChartsResult(BaseModel):
    fig1: go.Figure
    fig2: go.Figure
    fig1_base_64: str | None
    fig2_base_64: str | None
    code: str
    metadata: RunChartsResultMetadata

    model_config = ConfigDict(arbitrary_types_allowed=True)


class BusinessAnalysisMetadata(BaseModel):
    timestamp: str
    question: str
    rows_analyzed: int
    columns_analyzed: int


class BusinessAnalysisGeneration(BaseModel):
    bottom_line: str
    additional_insights: str
    follow_up_questions: list[str]


class BusinessAnalysisResult(BusinessAnalysisGeneration):
    metadata: BusinessAnalysisMetadata


class BusinessAnalysisRequest(BaseModel):
    """Request model for business analysis endpoint

    Attributes:
        data: list of dictionaries representing a single dataset
        dictionary: list of dictionary entries describing columns
        question: Business question to analyze
    """

    data: list[dict[Union[str, int], Any]]  # Allow both string and integer keys
    dictionary: list[dict[Union[str, int], Any]]  # Allow both string and integer keys
    question: str

    @field_validator("data")
    @classmethod
    def validate_data(cls: Type["BusinessAnalysisRequest"], v: Any) -> Any:
        """Validate that the data is a list of JSON objects and not empty"""
        if not isinstance(v, list):
            raise ValueError("Input data must be a list of JSON objects")
        if len(v) == 0:
            raise ValueError("Data cannot be empty")

        # Convert numeric keys to strings in nested dictionaries
        def convert_numeric_keys(d: Any) -> Any:
            """Recursively convert numeric keys to strings"""
            if not isinstance(d, dict):
                return d
            return {
                str(k): convert_numeric_keys(v) if isinstance(v, dict) else v
                for k, v in d.items()
            }

        # Convert all records
        converted = [convert_numeric_keys(record) for record in v]

        # Ensure all keys are strings after conversion
        for record in converted:
            if not all(isinstance(k, str) for k in record.keys()):
                raise ValueError("All dictionary keys must be strings after conversion")

        return converted

    @field_validator("dictionary")
    @classmethod
    def validate_dictionary(cls: Type["BusinessAnalysisRequest"], v: Any) -> Any:
        """Validate that the dictionary field is a list of dictionaries with string keys."""
        if not isinstance(v, list):
            raise ValueError("Dictionary must be a list")

        # Convert numeric keys to strings in dictionary entries
        def convert_numeric_keys(d: Any) -> Any:
            if not isinstance(d, dict):
                return d
            return {
                str(k): convert_numeric_keys(v) if isinstance(v, dict) else v
                for k, v in d.items()
            }

        # Convert all dictionary entries
        converted = [convert_numeric_keys(entry) for entry in v]

        # Validate required keys exist after conversion
        required_keys = {"column", "description", "data_type"}
        if not all(required_keys.issubset(d.keys()) for d in converted):
            raise ValueError(f"dictionary entries must contain keys: {required_keys}")

        return converted

    @field_validator("question")
    @classmethod
    def validate_question(cls: Type["BusinessAnalysisRequest"], v: str) -> str:
        if not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ChatRequest(BaseModel):
    """Request model for chat history processing

    Attributes:
        messages: list of dictionaries containing chat messages
                 Each message must have 'role' and 'content' fields
                 Role must be one of: 'user', 'assistant', 'system'
    """

    messages: list[dict[str, str]]

    @field_validator("messages")
    @classmethod
    def validate_messages(
        cls: Type["ChatRequest"], v: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Validate the messages list"""
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


class CodeValidator:
    """Validates Python code for safety and correctness"""

    ALLOWED_MODULES = {"pandas", "numpy", "scipy", "statsmodels", "sklearn"}

    @staticmethod
    def validate_imports(code: str) -> tuple[bool, str]:
        """Check if code only imports allowed modules"""
        try:
            tree = ast.parse(code)
            imports: list[str] = []

            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        imports.extend(n.name.split(".")[0] for n in node.names)
                    elif node.module is not None:
                        imports.append(node.module.split(".")[0])

            illegal_imports = set(imports) - CodeValidator.ALLOWED_MODULES
            if illegal_imports:
                return False, f"Illegal imports detected: {illegal_imports}"

            return True, "Validation passed"

        except SyntaxError as e:
            return False, f"Syntax error in code: {str(e)}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"


class CodeGenerationResult(BaseModel):
    """Container for code generation results"""

    code: str
    description: str
    validation: ValidationMessage
    metadata: dict[str, Any]
    attempts: int
    validation_errors: list[str]


class QuestionList(BaseModel):
    """Container for list of questions"""

    questions: list[str]


class QuestionSuggestionMetadata(BaseModel):
    """Metadata for question suggestions"""

    total_columns: int
    columns_used: int
    timestamp: str
    questions_generated: int
    valid_questions: int


class QuestionValidationResult(BaseModel):
    """Stores validation results for suggested questions"""

    question: str
    is_valid: bool
    available_columns: list[str]
    missing_columns: list[str]
    validation_message: str


class QuestionSuggestions(BaseModel):
    questions: list[QuestionValidationResult]
    metadata: QuestionSuggestionMetadata


class SnowflakeAnalysisRequest(BaseModel):
    """Request model for Snowflake analysis endpoint

    Attributes:
        data: dictionary of sample data from each table
        dictionary: dictionary of data dictionaries for each table
        question: Business question to analyze
        error_message: Optional error from previous attempt
        failed_code: Optional code that failed in previous attempt
    """

    data: dict[str, list[dict[str, Any]]]  # Sample data from each table
    dictionary: dict[str, dict[str, Any]]  # Pre-generated data dictionary
    question: str
    error_message: str | None = None
    failed_code: str | None = None

    @field_validator("data")
    @classmethod
    def validate_data(
        cls: Type["SnowflakeAnalysisRequest"], v: dict[str, list[dict[str, Any]]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Validate that the input data is a dictionary of lists of dictionaries (records)"""
        if not isinstance(v, dict):
            raise ValueError("Input data must be a dictionary of table samples")
        if not all(isinstance(samples, list) for samples in v.values()):
            raise ValueError("Each table's data must be a list of records")
        return v

    @field_validator("dictionary")
    @classmethod
    def validate_dictionary(
        cls: Type["SnowflakeAnalysisRequest"], v: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate that the input data is a dictionary of table descriptions"""
        if not isinstance(v, dict):
            raise ValueError("dictionary must be a dictionary of table descriptions")
        return v

    @field_validator("question")
    @classmethod
    def validate_question(cls: Type["SnowflakeAnalysisRequest"], v: str) -> str:
        """Validate that the input data is a non-empty string"""
        if not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()


class SnowflakeAnalysisCode(BaseModel):
    code: str
    description: str


class MemoryUsage(BaseModel):
    rss: float
    vms: float
    percent: float


class SnowflakeAnalysisMetadata(BaseModel):
    attempts: int
    execution_time: float
    memory_usage: MemoryUsage
    error_history: list[AnalysisError]
    query_metadata: SnowflakeExecutionMetadata | None = None
    tables_analyzed: int | None = None
    total_sample_rows: int | None = None


class AnalysisResult(BaseModel):
    status: str
    metadata: SnowflakeAnalysisMetadata | RunAnanlysisResultMetadata
    data: list[dict[str, Any]] | None = None
    code: str | None = None
    suggestions: str | None = None


class SnowflakeAnalysisResult(AnalysisResult):
    status: str
    metadata: SnowflakeAnalysisMetadata
    data: list[dict[str, Any]] | None = None
    code: str | None = None
    suggestions: str | None = None
    last_generated_code: str | None = None
    description: str | None = None


class SnowflakeExecutionMetadata(BaseModel):
    query_id: str
    row_count: int
    execution_time: float
    warehouse: str
    database: str
    db_schema: str


class AnalysisError(BaseModel):
    attempt: int
    error: str
    error_type: str
    timestamp: str
    memory_usage: MemoryUsage
    stdout: str | None = None
    stderr: str | None = None
    code: str | None = None


class ChartValidationError(BaseModel):
    attempt: int
    error: str
    timestamp: str
    code: str | None = None


class ValidationMessage(BaseModel):
    is_valid: bool
    message: str


class ChartExecutionError(BaseModel):
    attempt: int
    error_type: str
    error_message: str
    timestamp: str
    stdout: str | None = None
    stderr: str | None = None
    code: str | None = None


class ChartCodeHistory(BaseModel):
    attempt: int
    code: str
    timestamp: str


class ChartGenerationMetadata(BaseModel):
    timestamp: str
    question: str
    stdout: str
    stderr: str


class ChartPerformance(BaseModel):
    total_time: float
    memory_usage: MemoryUsage


class RunChartsResultMetadata(BaseModel):
    timestamp: str
    question: str
    stdout: str
    stderr: str
    dataframe_metadata: dict[str, Any]
    validation: ValidationMessage
    attempts: int
    validation_errors: list[ChartValidationError]
    execution_errors: list[ChartExecutionError]
    code_history: list[ChartCodeHistory]
    performance: ChartPerformance


class EnhancedUserMessageForChat(BaseModel):
    enhanced_user_message: str


class Code(BaseModel):
    code: str
    description: str
