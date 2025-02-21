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

import json
import logging
import sys
import traceback
from datetime import datetime
from typing import (
    Any,
)

import streamlit as st

sys.path.append("..")
from utils.app_db import AnalystDatasetDuckDB, DuckDBHandler

logger = logging.getLogger("DataAnalyst")


# Add enhanced error logging function
def log_error_details(error: BaseException, context: dict[str, Any]) -> None:
    """Log detailed error information with context

    Args:
        error: The exception that occurred
        context: Dictionary containing error context
    """
    error_details = {
        "timestamp": datetime.now().isoformat(),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "stack_trace": traceback.format_exc(),
        **context,
    }

    logger.error(
        f"\nERROR DETAILS\n=============\n{json.dumps(error_details, indent=2, default=str)}"
    )


empty_session_state = {
    "initialized": True,
    "datasets_names": [],
    "cleansed_data_names": [],
    "selected_catalog_datasets": [],
    "data_source": None,
    "file_uploader_key": 0,
    "processed_file_ids": [],
    "chat_messages": [],
    "chat_input_key": 0,
    "debug_mode": True,
}


def state_empty() -> None:
    for key, value in empty_session_state.items():
        st.session_state[key] = value
    logger.info("Session state has been reset to its initial empty state.")


def state_init() -> None:
    if "initialized" not in st.session_state:
        state_empty()
    if "duckdb_handler" not in st.session_state:
        st.session_state.duckdb_handler = DuckDBHandler(db_path="/tmp/app.db")
        st.session_state.analyst_db = AnalystDatasetDuckDB(
            st.session_state.duckdb_handler
        )
