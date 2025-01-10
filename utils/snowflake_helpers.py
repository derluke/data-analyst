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

import functools
import logging
import os
import time
from typing import Any

import pandas as pd
import snowflake.connector
from fastapi import HTTPException

from utils.credentials import SnowflakeCredentials
from utils.schema import SnowflakeExecutionMetadata

logger = logging.getLogger("DataAnalystFrontend")

SNOWFLAKE_CREDENTIALS = SnowflakeCredentials()


def _get_snowflake_private_key() -> str | None:
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
            return private_key

        except Exception as e:
            logger.warning(
                f"Failed to process private key: {str(e)}, falling back to password authentication"
            )
    logger.info("No valid private key path found, using password authentication")


def create_snowflake_connection() -> snowflake.connector.SnowflakeConnection:
    """Create a connection to Snowflake using environment variables"""
    connect_params = {
        "user": SNOWFLAKE_CREDENTIALS.user,
        "account": SNOWFLAKE_CREDENTIALS.account,
        "warehouse": SNOWFLAKE_CREDENTIALS.warehouse,
        "database": SNOWFLAKE_CREDENTIALS.database,
        "schema": SNOWFLAKE_CREDENTIALS.db_schema,
        "role": SNOWFLAKE_CREDENTIALS.role,
    }
    if private_key := _get_snowflake_private_key():
        connect_params["private_key"] = private_key
    else:
        connect_params["password"] = SNOWFLAKE_CREDENTIALS.password

    try:
        return snowflake.connector.connect(**connect_params)

    except Exception as e:
        logger.error(f"Failed to connect to Snowflake: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to connect to Snowflake: {str(e)}"
        )


def execute_snowflake_query(
    conn: snowflake.connector.SnowflakeConnection, query: str, timeout: int = 300
) -> tuple[(list[tuple] | list[dict]), SnowflakeExecutionMetadata]:
    """Execute a Snowflake query with timeout and metadata capture

    Args:
        conn: Snowflake connection
        query: SQL query to execute
        timeout: Query timeout in seconds

    Returns:
        Tuple of (results, metadata)
    """
    try:
        cursor = conn.cursor(snowflake.connector.DictCursor)
        start_time = time.time()

        # Set query timeout at cursor level
        cursor.execute(f"ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = {timeout}")

        try:
            # Execute query
            cursor.execute(query)

            # Get results
            results = cursor.fetchall()

            # Get query ID from the cursor
            query_id = cursor.sfqid

            # Calculate execution time
            execution_time = time.time() - start_time

            # Prepare metadata
            metadata = SnowflakeExecutionMetadata(
                query_id=query_id,
                row_count=len(results),
                execution_time=execution_time,
                warehouse=conn.warehouse,
                database=conn.database,
                db_schema=conn.schema,
            )

            return results, metadata

        except snowflake.connector.errors.ProgrammingError as e:
            # Handle Snowflake-specific errors
            raise Exception(f"Snowflake error: {str(e)}")

    except Exception as e:
        raise Exception(f"Query execution failed: {str(e)}")
    finally:
        if cursor:
            cursor.close()


@functools.lru_cache(maxsize=1)
def get_snowflake_tables() -> list[str]:
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


@functools.lru_cache(maxsize=8)
def get_snowflake_data(
    *table_names, sample_size: int = 5000
) -> dict[str, list[dict[str, Any]]]:
    """Load selected tables from Snowflake as pandas DataFrames

    Args:
    - table_names: List of table names to fetch
    - sample_size: Number of rows to sample from each table

    Returns:
    - Dictionary of table names to list of records
    """
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
                logger.info(
                    f"Successfully loaded table {table}: {len(df)} rows, {len(df.columns)} columns"
                )
                dataframes[table] = df.to_dict(orient="records")

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
