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
from typing import Any, Dict, List

sys.path.append("..")

import datarobot as dr
import pandas as pd
import snowflake.connector
import streamlit as st

from utils.api import cleanse_dataframes, get_catalog_datasets, get_dictionary
from utils.credentials import SnowflakeCredentials
from utils.schema import CleanseRequest, DatasetInput

SNOWFLAKE_CREDENTIALS = SnowflakeCredentials()

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
@st.cache_resource(show_spinner=False)
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

        # Create cleanse request with the list
        cleanse_request = CleanseRequest(datasets=datasets_list)

        # Cleanse the data
        cleansed_results = await cleanse_dataframes(cleanse_request)

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

        # Create the request with the datasets
        request_data = CleanseRequest(datasets=datasets)

        dictionary_response = await get_dictionary(request_data)
        dictionary_response = dictionary_response.model_dump()

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


@st.cache_data(show_spinner=False)
def process_uploaded_file(file):
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


# Add function to load selected datasets
@st.cache_data(show_spinner=False)
def get_datasets_as_df(_dataset_ids: List[str]) -> Dict[str, pd.DataFrame]:
    """Load selected datasets as pandas DataFrames"""
    dataframes = {}
    for _, id in enumerate(_dataset_ids, 1):
        dataset = dr.Dataset.get(id)
        try:
            dataframes[dataset.name] = dataset.get_as_dataframe()
            logger.info(f"Successfully downloaded {dataset.name}")
        except Exception as e:
            logger.error(f"Failed to read dataset {dataset.name}: {str(e)}")
            continue
    return dataframes


# Add callback for AI Catalog dataset selection
def catalog_download_callback():
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
                dataframes = get_datasets_as_df(selected_ids)

                # Add downloaded dataframes to session state
                for name, df in dataframes.items():
                    st.session_state.datasets[name] = df

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


def clear_data_callback():
    """Callback function to clear all data from session state and cache"""
    # Clear session state
    st.session_state.datasets = {}
    st.session_state.cleansed_data = {}
    st.session_state.data_dictionaries = {}
    st.session_state.selected_catalog_datasets = []  # Also clear catalog selection
    st.session_state.data_source = None  # Reset data source flag

    # Clear all Streamlit caches
    st.cache_data.clear()
    st.cache_resource.clear()


# Add after other initialization functions
def create_snowflake_connection() -> snowflake.connector.SnowflakeConnection:
    """Create a Snowflake connection with either private key or password authentication"""
    try:
        # Log all environment variables (excluding sensitive data)
        logger.info("=== Starting Snowflake Connection Attempt ===")
        logger.info("Environment variables check:")
        logger.info(f"USER: {SNOWFLAKE_CREDENTIALS.user}")
        logger.info(f"ACCOUNT: {SNOWFLAKE_CREDENTIALS.account}")
        logger.info(f"WAREHOUSE: {SNOWFLAKE_CREDENTIALS.warehouse}")
        logger.info(f"DATABASE: {SNOWFLAKE_CREDENTIALS.database}")
        logger.info(f"SCHEMA: {SNOWFLAKE_CREDENTIALS.db_schema}")
        logger.info(f"ROLE: {SNOWFLAKE_CREDENTIALS.role}")
        logger.info(f"KEY_PATH: {SNOWFLAKE_CREDENTIALS.snowflake_key_path}")

        # Initialize connection parameters with common values
        conn_params = {
            "user": SNOWFLAKE_CREDENTIALS.user,
            "account": SNOWFLAKE_CREDENTIALS.account,
            "warehouse": SNOWFLAKE_CREDENTIALS.warehouse,
            "database": SNOWFLAKE_CREDENTIALS.role,
            "schema": SNOWFLAKE_CREDENTIALS.db_schema,
            "role": SNOWFLAKE_CREDENTIALS.role,
        }

        # Check if private key path is set and valid
        key_path = SNOWFLAKE_CREDENTIALS.snowflake_key_path
        if key_path and os.path.exists(key_path):
            try:
                # Read and process private key
                with open(key_path, "rb") as key_file:
                    private_key_data = key_file.read()
                    logger.info("Successfully read private key file")

                # Load and convert key
                from cryptography.hazmat.backends import default_backend
                from cryptography.hazmat.primitives import serialization

                p_key = serialization.load_pem_private_key(
                    private_key_data, password=None, backend=default_backend()
                )
                logger.info("Successfully loaded PEM key")

                private_key = p_key.private_bytes(
                    encoding=serialization.Encoding.DER,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption(),
                )
                logger.info("Successfully converted key to DER format")

                # Add private key to connection parameters
                conn_params["private_key"] = private_key
                logger.info("Using private key authentication")

            except Exception as e:
                logger.warning(
                    f"Failed to process private key: {str(e)}, falling back to password authentication"
                )
                conn_params["password"] = SNOWFLAKE_CREDENTIALS.password
        else:
            # Use password authentication
            logger.info(
                "No valid private key path found, using password authentication"
            )
            conn_params["password"] = SNOWFLAKE_CREDENTIALS.password

        # Log password status (safely)
        logger.info(
            f"Password is {'set' if conn_params.get('password') else 'not set'}"
        )

        # Attempt connection
        conn = snowflake.connector.connect(**conn_params)
        logger.info(
            f"Successfully connected to Snowflake using role: {conn_params['role']}"
        )
        return conn

    except Exception as e:
        logger.error("=== Snowflake Connection Failed ===")
        logger.error(f"Final error: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        raise


@st.cache_data(show_spinner=False)
def get_snowflake_tables() -> List[str]:
    """Fetch list of tables from Snowflake schema"""
    try:
        conn = create_snowflake_connection()
        cursor = conn.cursor()

        # Log current session info
        logger.info("Checking current session settings...")
        cursor.execute(
            "SELECT CURRENT_DATABASE(), CURRENT_SCHEMA(), CURRENT_ROLE(), CURRENT_WAREHOUSE()"
        )
        current_settings = cursor.fetchone()
        logger.info(
            f"Current settings - Database: {current_settings[0]}, Schema: {current_settings[1]}, Role: {current_settings[2]}, Warehouse: {current_settings[3]}"
        )

        # Check if schema exists
        cursor.execute(
            f"""
            SELECT COUNT(*) 
            FROM {SNOWFLAKE_CREDENTIALS.database}.INFORMATION_SCHEMA.SCHEMATA 
            WHERE SCHEMA_NAME = '{SNOWFLAKE_CREDENTIALS.db_schema}'
        """
        )
        schema_exists = cursor.fetchone()[0]
        logger.info(f"Schema exists check: {schema_exists > 0}")

        # Get all objects (tables and views)
        cursor.execute(
            f"""
            SELECT table_name, table_type
            FROM {SNOWFLAKE_CREDENTIALS.database}.information_schema.tables 
            WHERE table_schema = '{SNOWFLAKE_CREDENTIALS.db_schema}'
            AND table_type IN ('BASE TABLE', 'VIEW')
            ORDER BY table_type, table_name
        """
        )
        results = cursor.fetchall()
        tables = [row[0] for row in results]

        # Log detailed results
        logger.info(f"Total objects found: {len(results)}")
        for table_name, table_type in results:
            logger.info(f"Found {table_type}: {table_name}")

        # Check schema privileges
        cursor.execute(
            f"""
            SHOW GRANTS ON SCHEMA {SNOWFLAKE_CREDENTIALS.database}.{SNOWFLAKE_CREDENTIALS.db_schema}
        """
        )
        privileges = cursor.fetchall()
        logger.info("Schema privileges:")
        for priv in privileges:
            logger.info(f"Privilege: {priv}")

        return tables

    except Exception as e:
        logger.error(f"Failed to fetch tables: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error details: {str(e)}")
        return []

    finally:
        try:
            if "cursor" in locals():
                cursor.close()
            if "conn" in locals():
                conn.close()
        except Exception:
            pass


@st.cache_data(show_spinner=False)
def get_snowflake_data(
    table_names: List[str], sample_size: int = 5000
) -> Dict[str, pd.DataFrame]:
    """Load selected tables from Snowflake as pandas DataFrames"""
    dataframes = {}

    try:
        conn = create_snowflake_connection()
        cursor = conn.cursor()

        for table in table_names:
            try:
                qualified_table = f"{SNOWFLAKE_CREDENTIALS.database}.{SNOWFLAKE_CREDENTIALS.db_schema}.{table}"
                logger.info(f"Fetching data from table: {qualified_table}")

                cursor.execute(
                    f"""
                    SELECT * FROM {qualified_table}
                    SAMPLE ({sample_size} ROWS)
                """
                )

                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                df = pd.DataFrame(data, columns=columns)

                # Convert date/datetime columns to string format
                for col in df.columns:
                    if pd.api.types.is_datetime64_any_dtype(df[col]) or isinstance(
                        df[col].dtype, pd.DatetimeTZDtype
                    ):
                        df[col] = df[col].dt.strftime("%Y-%m-%d %H:%M:%S")
                    elif df[col].dtype == "object":
                        try:
                            pd.to_datetime(df[col], errors="raise")
                            df[col] = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d")
                        except (ValueError, TypeError):
                            continue

                dataframes[table] = df
                logger.info(
                    f"Successfully loaded table {table}: {len(df)} rows, {len(df.columns)} columns"
                )

            except Exception as e:
                logger.error(f"Error loading table {table}: {str(e)}")
                logger.error(f"Error type: {type(e)}")
                logger.error(f"Error details: {str(e)}")
                continue

        return dataframes

    except Exception as e:
        logger.error(f"Error fetching Snowflake data: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error details: {str(e)}")
        return {}

    finally:
        try:
            if "cursor" in locals():
                cursor.close()
            if "conn" in locals():
                conn.close()
        except Exception:
            pass


# Modify the load_snowflake_data callback
def load_snowflake_data_callback():
    """Callback function for Snowflake table download"""
    if (
        "selected_snowflake_tables" in st.session_state
        and st.session_state.selected_snowflake_tables
    ):
        with st.sidebar:
            with st.spinner("Loading selected tables..."):
                # Get data from Snowflake
                dataframes = get_snowflake_data(
                    st.session_state.selected_snowflake_tables
                )

                if not dataframes:
                    st.error("Failed to load data from Snowflake")
                    return

                # Add downloaded dataframes to session state
                for name, df in dataframes.items():
                    st.session_state.datasets[name] = df
                    st.success(f"✓ {name}: {len(df)} rows, {len(df.columns)} columns")

                # Set flag to indicate data source is Snowflake
                st.session_state.data_source = "snowflake"

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
st.set_page_config(
    page_title="Connect Data", page_icon="datarobot icon.svg", layout="wide"
)


# Custom CSS
st.markdown(
    """
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1c83e1;
    }
    .stDownloadButton button {
        width: 100%;
    }
    </style>
""",
    unsafe_allow_html=True,
)

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
            datasets = [i.model_dump() for i in get_catalog_datasets()]

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
        st.image("Snowflake.svg", width=100)

        # Initialize Snowflake connection
        snowflake_tables = get_snowflake_tables()

        # Create form for Snowflake table selection
        with st.form("snowflake_selection_form", border=False):
            selected_snowflake_tables = st.multiselect(
                "Select datasets from Snowflake",
                options=snowflake_tables,
                help="You can select multiple tables",
                key="selected_snowflake_tables",
            )

            # Form submit button
            submit_button = st.form_submit_button(
                "Load Selected Tables",
                use_container_width=False,
                on_click=load_snowflake_data_callback,
            )

            if submit_button and not selected_snowflake_tables:
                st.warning("Please select at least one table")

    # Add Clear Data button after the Database expander
    st.sidebar.button(
        "Clear Data",
        on_click=clear_data_callback,
        type="secondary",
        use_container_width=False,
    )

# Main content area
st.image("datarobot logo.svg", width=200)
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
        df = data["data"]

        # Create column filters
        col1, col2 = st.columns([3, 1])
        with col1:
            search = st.text_input(
                "Search columns", key=f"search_{name}", help="Filter columns by name"
            )
        with col2:
            n_rows = st.number_input(
                "Rows to display",
                min_value=1,
                max_value=len(df),
                value=min(10, len(df)),
                key=f"n_rows_{name}",
            )

        # Filter columns based on search
        if search:
            cols = [col for col in df.columns if search.lower() in col.lower()]
        else:
            cols = df.columns

        # Display filtered dataframe
        st.dataframe(df[cols].head(n_rows), use_container_width=True)

        # Download button
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Cleansed Data",
                data=csv,
                file_name=f"{name}_cleansed.csv",
                mime="text/csv",
                key=f"download_{name}",
            )

        st.markdown("---")
