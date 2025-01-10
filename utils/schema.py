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
from typing import Any, Dict, List, Tuple, Union

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


class CleanseResult(BaseModel):
    datasets: List[DatasetOutput]
    metadata: Dict[str, Any]


class DictionaryRequest(BaseModel):
    data: List[Dict[str, Any]]

    @field_validator("data")
    @classmethod
    def validate_data(cls, v):
        if not isinstance(v, list):
            raise ValueError("Input data must be a list of dictionaries")
        return v


class DictionaryDataColumn(BaseModel):
    data_type: str
    column: str
    description: str


class DataDictionary(BaseModel):
    name: str
    dictionary: List[DictionaryDataColumn]
    cache_hit: bool
    batch_time: float


class DataDictionaryMetadata(BaseModel):
    total_datasets: int
    processing_start: str
    batch_times: List[float]
    errors: List[str]
    processing_end: str = None
    total_time: float = None


class DataDictionariesAndMetadata(BaseModel):
    metadata: DataDictionaryMetadata
    dictionaries: list[DataDictionary]


# Add after the DictionaryRequest class
class DictionaryResult(BaseModel):
    """Validates LLM responses for data dictionary generation

    Attributes:
        columns: List of column names
        descriptions: List of column descriptions

    Raises:
        ValueError: If validation fails
    """

    columns: List[str]
    descriptions: List[str]

    @field_validator("descriptions")
    @classmethod
    def validate_descriptions(cls, v, values):
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
    def validate_columns(cls, v):
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

    def to_dict(self) -> Dict[str, str]:
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

    data: Dict[str, List[Dict[str, Any]]]
    dictionary: Dict[
        str, List[Dict[str, Union[str, Dict[str, str]]]]
    ]  # Allow dictionary values for description
    question: str
    error_message: str | None = None
    failed_code: str | None = None

    @field_validator("data")
    @classmethod
    def validate_data(cls, v):
        if not isinstance(v, dict):
            raise ValueError("Input data must be a dictionary of datasets")
        if not all(isinstance(dataset, list) for dataset in v.values()):
            raise ValueError("Each dataset must be a list of dictionaries")
        return v

    @field_validator("dictionary")
    @classmethod
    def validate_dictionary(cls, v):
        if not isinstance(v, dict):
            raise ValueError("Dictionary must be a dictionary of dataset descriptions")

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
    metadata: RunAnanlysisResultMetadata
    code: str | None = None
    data: list[dict[str, Any]] | None = None
    suggestions: str | None = None


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
    error_message: str | None = None
    failed_code: str | None = None

    @field_validator("data")
    @classmethod
    def validate_data(cls, v):
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
    validation_errors: List[ChartValidationError]
    execution_errors: List[ChartExecutionError]
    code_history: List[ChartCodeHistory]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class RunChartsRequest(BaseModel):
    """Request model for charts endpoint

    Attributes:
        data: List of dictionaries representing a single dataset
        question: Business question to visualize
        error_message: Optional error from previous attempt
        failed_code: Optional code that failed in previous attempt
    """

    data: List[Dict[Union[str, int], Any]]  # Allow both string and integer keys
    question: str
    error_message: str | None = None
    failed_code: str | None = None

    @field_validator("data")
    @classmethod
    def validate_data(cls, v):
        if not isinstance(v, list):
            raise ValueError("Input data must be a list of dictionaries")
        if not all(isinstance(record, dict) for record in v):
            raise ValueError("Each record must be a dictionary")

        # Convert numeric keys to strings in nested dictionaries
        def convert_numeric_keys(d):
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
    follow_up_questions: List[str]


class BusinessAnalysisResult(BusinessAnalysisGeneration):
    metadata: BusinessAnalysisMetadata


class BusinessAnalysisRequest(BaseModel):
    """Request model for business analysis endpoint

    Attributes:
        data: List of dictionaries representing a single dataset
        dictionary: List of dictionary entries describing columns
        question: Business question to analyze
    """

    data: List[Dict[Union[str, int], Any]]  # Allow both string and integer keys
    dictionary: List[Dict[Union[str, int], Any]]  # Allow both string and integer keys
    question: str

    @field_validator("data")
    @classmethod
    def validate_data(cls, v):
        if not isinstance(v, list):
            raise ValueError("Input data must be a list of JSON objects")
        if len(v) == 0:
            raise ValueError("Data cannot be empty")

        # Convert numeric keys to strings in nested dictionaries
        def convert_numeric_keys(d):
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
    def validate_dictionary(cls, v):
        if not isinstance(v, list):
            raise ValueError("Dictionary must be a list")

        # Convert numeric keys to strings in dictionary entries
        def convert_numeric_keys(d):
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
            raise ValueError(f"Dictionary entries must contain keys: {required_keys}")

        return converted

    @field_validator("question")
    @classmethod
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ChatRequest(BaseModel):
    """Request model for chat history processing

    Attributes:
        messages: List of dictionaries containing chat messages
                 Each message must have 'role' and 'content' fields
                 Role must be one of: 'user', 'assistant', 'system'
    """

    messages: List[Dict[str, str]]

    @field_validator("messages")
    @classmethod
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


class CodeGenerationResult(BaseModel):
    """Container for code generation results"""

    code: str
    description: str
    validation: ValidationMessage
    metadata: Dict[str, Any]
    attempts: int
    validation_errors: List[str]


class QuestionList(BaseModel):
    """Container for list of questions"""

    questions: List[str]


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
    available_columns: List[str]
    missing_columns: List[str]
    validation_message: str


class QuestionSuggestions(BaseModel):
    questions: List[QuestionValidationResult]
    metadata: QuestionSuggestionMetadata


class SnowflakeAnalysisRequest(BaseModel):
    """Request model for Snowflake analysis endpoint

    Attributes:
        data: Dictionary of sample data from each table
        dictionary: Dictionary of data dictionaries for each table
        question: Business question to analyze
        error_message: Optional error from previous attempt
        failed_code: Optional code that failed in previous attempt
        warehouse: Snowflake warehouse to use
        database: Snowflake database to use
        schema: Snowflake schema to use
    """

    data: Dict[str, List[Dict[str, Any]]]  # Sample data from each table
    dictionary: Dict[str, Dict[str, Any]]  # Pre-generated data dictionary
    question: str
    error_message: str | None = None
    failed_code: str | None = None
    warehouse: str
    database: str
    db_schema: str

    @field_validator("data")
    @classmethod
    def validate_data(cls, v):
        if not isinstance(v, dict):
            raise ValueError("Input data must be a dictionary of table samples")
        if not all(isinstance(samples, list) for samples in v.values()):
            raise ValueError("Each table's data must be a list of records")
        return v

    @field_validator("dictionary")
    @classmethod
    def validate_dictionary(cls, v):
        if not isinstance(v, dict):
            raise ValueError("Dictionary must be a dictionary of table descriptions")
        return v

    @field_validator("question")
    @classmethod
    def validate_question(cls, v):
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


class SnowflakeAnalysisResult(BaseModel):
    status: str
    metadata: SnowflakeAnalysisMetadata
    data: List[Dict[str, Any]] | None = None
    code: str | None = None
    last_generated_code: str | None = None
    description: str | None = None
    suggestions: str | None = None


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
    dataframe_metadata: Dict[str, Any]
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
