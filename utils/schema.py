from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import plotly.graph_objects as go
from pydantic import BaseModel, field_validator


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


class CleanseResponse(BaseModel):
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


@dataclass
class QuestionValidationResult:
    """Stores validation results for suggested questions"""

    question: str
    is_valid: bool
    available_columns: List[str]
    missing_columns: List[str]
    validation_message: str


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
        if not all(isinstance(desc, list) for desc in v.values()):
            raise ValueError("Each dataset description must be a list")
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

    @field_validator("data")
    @classmethod
    def validate_data(cls, v):
        if not isinstance(v, list):
            raise ValueError("Input data must be a list of dictionaries")
        if not all(isinstance(record, dict) for record in v):
            raise ValueError("Each record must be a dictionary")
        return v


class BusinessAnalysisRequest(BaseModel):
    data: List[Dict[str, Any]]  # JSON data
    dictionary: List[Dict[str, str]]  # JSON dictionary
    question: str

    @field_validator("data")
    @classmethod
    def validate_data(cls, v):
        if not isinstance(v, list):
            raise ValueError("Input data must be a list of JSON objects")
        if len(v) == 0:
            raise ValueError("Data cannot be empty")
        return v

    @field_validator("dictionary")
    @classmethod
    def validate_dictionary(cls, v):
        if not isinstance(v, list):
            raise ValueError("Dictionary must be a list of objects")
        required_keys = {"column", "description", "data_type"}
        if not all(required_keys.issubset(d.keys()) for d in v):
            raise ValueError(f"Dictionary entries must contain keys: {required_keys}")
        return v

    @field_validator("question")
    @classmethod
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()


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


@dataclass
class CodeGenerationResult:
    """Container for code generation results"""

    code: str
    description: str
    validation: Dict[str, Any]
    metadata: Dict[str, Any]
    attempts: int
    validation_errors: List[str]
