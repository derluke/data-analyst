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

import asyncio
import logging
import os
import sys
import warnings
from typing import Any, Dict

import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

sys.path.append("..")

from app_settings import PAGE_ICON, get_database_logo, get_page_logo

from utils.api import (
    cleanse_dataframes,
    download_catalog_datasets,
    get_dictionary,
    list_catalog_datasets,
)
from utils.database_helpers import Database, app_infra
from utils.schema import DatasetInput

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Initialize session state variables
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.datasets = {}
    st.session_state.cleansed_data = {}
    st.session_state.data_dictionaries = {}
    st.session_state.data_source = None


# Modify process_data to handle coroutine reuse
def process_data_cached(datasets_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Wrapper function to handle async processing with caching
    """
    return asyncio.run(process_data_async(datasets_dict))


async def process_data_async(datasets_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Process and cleanse the uploaded datasets using the API functions.
    Returns a dictionary containing cleansed data and success status.
    """
    try:
        # Convert the datasets dictionary to a list of DatasetInput objects
        datasets_list = [
            DatasetInput(name=name, data=df.to_dict(orient="records"))
            for name, df in datasets_dict.items()
        ]

        # Cleanse the data
        cleansed_results = await cleanse_dataframes(datasets_list)

        # Format results
        cleansed_data = {
            dataset.name: {
                "data": pd.DataFrame(dataset.data),
                "report": dataset.cleaning_report,
            }
            for dataset in cleansed_results.datasets
        }

        return {"success": True, "data": cleansed_data}

    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        return {"success": False, "error": str(e)}


def generate_dictionaries(
    _cleansed_data: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Wrapper function to handle async dictionary generation
    """
    return asyncio.run(generate_dictionaries_async(_cleansed_data))


# TODO: move to utils.api.py def generate_dictionaries(cleansed_data: CleansedResult) -> DataDictionariesAndMetadata
async def generate_dictionaries_async(
    _cleansed_data: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Generate data dictionaries for all datasets"""
    try:
        # Create a list of DatasetInput objects
        datasets = []
        logger.info(
            f"Starting dictionary generation for {len(_cleansed_data)} datasets"
        )

        for name, data in _cleansed_data.items():
            if isinstance(data, dict) and "data" in data:
                df = data["data"]
                datasets.append(
                    DatasetInput(name=name, data=df.to_dict(orient="records"))
                )
                logger.info(f"Added dataset {name} for dictionary generation")

        _dictionary_response = await get_dictionary(datasets)
        dictionary_response = _dictionary_response.model_dump()

        if dictionary_response and isinstance(dictionary_response, dict):
            if "dictionaries" in dictionary_response:
                result_dict = {
                    dict_entry["name"]: dict_entry["dictionary"]
                    for dict_entry in dictionary_response["dictionaries"]
                    if dict_entry.get("name") and "dictionary" in dict_entry
                }
                logger.info(
                    f"Successfully generated dictionaries for {len(result_dict)} datasets"
                )
                return result_dict
            else:
                logger.warning("Dictionary response missing 'dictionaries' key")
        else:
            logger.warning(
                f"Unexpected dictionary response format: {type(dictionary_response)}"
            )

        return {}

    except Exception as e:
        logger.error(f"Error generating dictionaries: {str(e)}", exc_info=True)
        return {}


def process_uploaded_file(file: UploadedFile) -> list[tuple[str, pd.DataFrame]]:
    """Process a single uploaded file and return a list of (dataset_name, dataframe) tuples

    Args:
        file: The uploaded file object
    Returns:
        list: List of (dataset_name, dataframe) tuples, or empty list if error
    """
    try:
        logger.info(f"Processing uploaded file: {file.name}")
        file_extension = os.path.splitext(file.name)[1].lower()
        results = []

        if file_extension == ".csv":
            df = pd.read_csv(file)
            dataset_name = os.path.splitext(file.name)[0]
            results.append((dataset_name, df))
            logger.info(
                f"Loaded CSV {dataset_name}: {len(df)} rows, {len(df.columns)} columns"
            )

        elif file_extension in [".xlsx", ".xls"]:
            # Read all sheets
            excel_file = pd.ExcelFile(file)
            base_name = os.path.splitext(file.name)[0]

            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                # Use sheet name as dataset name if multiple sheets, otherwise use file name
                dataset_name = (
                    f"{base_name}_{sheet_name}"
                    if len(excel_file.sheet_names) > 1
                    else base_name
                )
                results.append((dataset_name, df))
                logger.info(
                    f"Loaded Excel sheet {dataset_name}: {len(df)} rows, {len(df.columns)} columns"
                )
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        return results

    except Exception as e:
        logger.error(f"Error loading {file.name}: {str(e)}", exc_info=True)
        return []


# Add callback for AI Catalog dataset selection
def catalog_download_callback() -> None:
    """Callback function for AI Catalog dataset download"""
    if (
        "selected_catalog_datasets" in st.session_state
        and st.session_state.selected_catalog_datasets
    ):
        with st.sidebar:  # Use sidebar context
            with st.spinner("Loading selected datasets..."):
                selected_ids = [
                    ds["id"] for ds in st.session_state.selected_catalog_datasets
                ]
                dataframes = download_catalog_datasets(*selected_ids)

                # Add downloaded dataframes to session state
                for name, df in dataframes.items():
                    st.session_state.datasets[name] = pd.DataFrame(df)

                # Process the new data
                results = process_data_cached(st.session_state.datasets)

                if results["success"]:
                    st.session_state.cleansed_data = results["data"]
                    logger.info("Data processing successful, generating dictionaries")

                    # Generate data dictionaries
                    st.session_state.data_dictionaries = generate_dictionaries(
                        st.session_state.cleansed_data
                    )

                    if st.session_state.data_dictionaries:
                        st.success(
                            "✅ Data processed and dictionaries generated successfully!"
                        )
                    else:
                        st.warning(
                            "⚠️ Data processed but there were issues generating some dictionaries"
                        )
                else:
                    logger.error("Data processing failed")
                    st.error(
                        f"❌ Error processing data: {results.get('error', 'Unknown error')}"
                    )


def clear_data_callback() -> None:
    """Callback function to clear all data from session state and cache"""
    # Clear session state
    st.session_state.datasets = {}
    st.session_state.cleansed_data = {}
    st.session_state.data_dictionaries = {}
    st.session_state.selected_catalog_datasets = []  # Also clear catalog selection
    st.session_state.data_source = None  # Reset data source flag


def load_from_database_callback() -> None:
    """Callback function for Snowflake table download"""
    if (
        "selected_schema_tables" in st.session_state
        and st.session_state.selected_schema_tables
    ):
        with st.sidebar:
            with st.spinner("Loading selected tables..."):
                # Get data from Snowflake
                dataframes = Database.get_data(*st.session_state.selected_schema_tables)

                if not dataframes:
                    st.error(f"Failed to load data from {app_infra.database}")
                    return

                # Add downloaded dataframes to session state
                for name, df in dataframes.items():
                    st.session_state.datasets[name] = pd.DataFrame(df)
                    st.success(f"✓ {name}: {len(df)} rows, {len(df[0].keys())} columns")

                # Set flag to indicate data source is a database
                st.session_state.data_source = "database"

                # Process the new data
                results = process_data_cached(st.session_state.datasets)

                if results["success"]:
                    st.session_state.cleansed_data = results["data"]
                    logger.info("Data processing successful, generating dictionaries")

                    # Generate data dictionaries
                    st.session_state.data_dictionaries = generate_dictionaries(
                        st.session_state.cleansed_data
                    )

                    if st.session_state.data_dictionaries:
                        st.success(
                            "✅ Data processed and dictionaries generated successfully!"
                        )
                    else:
                        st.warning(
                            "⚠️ Data processed but there were issues generating some dictionaries"
                        )
                else:
                    logger.error("Data processing failed")
                    st.error(
                        f"❌ Error processing data: {results.get('error', 'Unknown error')}"
                    )


# Page config
st.set_page_config(page_title="Connect Data", page_icon=PAGE_ICON, layout="wide")


# Custom CSS
with open("./style.css") as f:
    css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Sidebar for data upload and processing
with st.sidebar:
    st.title("Connect")

    # Load Files expander containing file upload and AI Catalog
    with st.expander("Load Files", expanded=True):
        # File upload section
        col1, col2, col3 = st.columns([1, 4, 2])
        with col1:
            st.image("csv_File_Logo.svg", width=25)
        with col2:
            st.write("**Load Data Files**")
        uploaded_files = st.file_uploader(
            "Select 1 or multiple files",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            with st.spinner("Loading and processing files..."):
                # Process uploaded files
                for file in uploaded_files:
                    dataset_results = process_uploaded_file(file)
                    if dataset_results:
                        for dataset_name, df in dataset_results:
                            st.session_state.datasets[dataset_name] = df
                            st.success(
                                f"✓ {dataset_name}: {len(df)} rows, {len(df.columns)} columns"
                            )

                # Process data and generate dictionaries
                logger.info("Starting data processing")
                results = process_data_cached(st.session_state.datasets)

                if results["success"]:
                    st.session_state.cleansed_data = results["data"]
                    logger.info("Data processing successful, generating dictionaries")

                    # Generate new dictionaries
                    new_dictionaries = generate_dictionaries(
                        st.session_state.cleansed_data
                    )

                    # Update session state by merging existing and new dictionaries
                    existing_dicts = st.session_state.get("data_dictionaries", {})
                    st.session_state.data_dictionaries = {
                        **existing_dicts,
                        **new_dictionaries,
                    }

                    if st.session_state.data_dictionaries:
                        st.success(
                            "✅ Data processed and dictionaries generated successfully!"
                        )
                        st.info(
                            "View the generated data dictionaries in the [Data Dictionary](/Data_Dictionary) page"
                        )
                    else:
                        st.warning(
                            "⚠️ Data processed but there were issues generating some dictionaries"
                        )
                else:
                    logger.error("Data processing failed")
                    st.error(
                        f"❌ Error processing data: {results.get('error', 'Unknown error')}"
                    )

        # AI Catalog section
        st.subheader("☁️   DataRobot AI Catalog")

        # Get datasets from catalog
        with st.spinner("Loading datasets from AI Catalog..."):
            datasets = [i.model_dump() for i in list_catalog_datasets()]

        # Create form for dataset selection
        with st.form("catalog_selection_form", border=False):
            selected_catalog_datasets = st.multiselect(
                "Select datasets from AI Catalog",
                options=datasets,
                format_func=lambda x: f"{x['name']} ({x['size']})",
                help="You can select multiple datasets",
                key="selected_catalog_datasets",
            )

            # Form submit button
            submit_button = st.form_submit_button(
                "Load Datasets", on_click=catalog_download_callback
            )

            # Process form submission
            if submit_button and len(selected_catalog_datasets) > 0:
                # The callback will handle the download and processing
                pass
            elif submit_button:
                st.warning("Please select at least one dataset")

    # Database expander containing Snowflake section
    with st.expander("Database", expanded=False):
        st.image(get_database_logo(app_infra), width=100)

        schema_tables = Database.get_tables()

        # Create form for Snowflake table selection
        with st.form("table_selection_form", border=False):
            selected_schema_tables = st.multiselect(
                "Select datasets from Snowflake",
                options=schema_tables,
                help="You can select multiple tables",
                key="selected_schema_tables",
            )

            # Form submit button
            submit_button = st.form_submit_button(
                "Load Selected Tables",
                use_container_width=False,
                on_click=load_from_database_callback,
            )

            if submit_button and not selected_schema_tables:
                st.warning("Please select at least one table")

    # Add Clear Data button after the Database expander
    st.sidebar.button(
        "Clear Data",
        on_click=clear_data_callback,
        type="secondary",
        use_container_width=False,
    )

# Main content area
st.image(get_page_logo(), width=200)
st.title("Explore")

# Main content area - conditional rendering based on cleansed data
if not st.session_state.cleansed_data:
    st.info("Upload and process your data using the sidebar to get started")
else:
    for name, data in st.session_state.cleansed_data.items():
        st.subheader(f"{name}")

        # Display cleaning report in expander
        with st.expander("View Cleaning Report"):
            report = data["report"]
            if report.columns_cleaned:
                st.write("**Columns Cleaned:**")
                st.write(", ".join(report.columns_cleaned))

                if report.warnings:
                    st.write("**Warnings:**")
                    for warning in report.warnings:
                        st.write(f"- {warning}")

                if report.errors:
                    st.error("**Errors:**")
                    for error in report.errors:
                        st.write(f"- {error}")

        # Display dataframe with column filters
        df_display: pd.DataFrame = data["data"]

        # Create column filters
        col1, col2 = st.columns([3, 1])
        with col1:
            search = st.text_input(
                "Search columns", key=f"search_{name}", help="Filter columns by name"
            )
        with col2:
            n_rows = int(
                st.number_input(
                    "Rows to display",
                    min_value=1,
                    max_value=len(df_display),
                    value=min(10, len(df_display)),
                    step=1,
                    key=f"n_rows_{name}",
                )
            )

        # Filter columns based on search
        if search:
            cols = [col for col in df_display.columns if search.lower() in col.lower()]
        else:
            cols = df_display.columns.tolist()

        # Display filtered dataframe
        st.dataframe(df_display[cols].head(n_rows), use_container_width=True)

        # Download button
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            csv = df_display.to_csv(index=False)
            st.download_button(
                label="Download Cleansed Data",
                data=csv,
                file_name=f"{name}_cleansed.csv",
                mime="text/csv",
                key=f"download_{name}",
            )

        st.markdown("---")
