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
from typing import cast

import duckdb
import polars as pl

from utils.logging_helper import get_logger
from utils.schema import (
    AnalystDataset,
    CleansedColumnReport,
    CleansedDataset,
    DataDictionary,
)

logger = get_logger(__name__)


class DuckDBHandler:
    def __init__(self, db_path: str = "chat.db"):
        self.conn = duckdb.connect(db_path)
        self.conn.execute("SET extension_directory = '/tmp/.duckdb/';")
        self.conn.install_extension("parquet")
        self.conn.load_extension("parquet")

        # Create metadata table for cleansing reports
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cleansing_reports (
                dataset_name VARCHAR,
                report JSON,
                PRIMARY KEY (dataset_name)
            )
        """)

    def register_dataframe(
        self, df: pl.DataFrame, name: str, is_cleansed: bool = False
    ) -> None:
        """Register a Polars DataFrame as a DuckDB table."""
        table_name = f"{name}_cleansed" if is_cleansed else name
        self.conn.register(table_name, df.to_arrow())

    def get_dataframe(self, name: str, cleansed: bool = False) -> pl.DataFrame:
        """Retrieve a registered table as a Polars DataFrame."""
        table_name = f"{name}_cleansed" if cleansed else name
        try:
            result = self.conn.execute(f'SELECT * FROM "{table_name}"')
            arrow_table = result.arrow()
            return cast(pl.DataFrame, pl.from_arrow(arrow_table))
        except duckdb.CatalogException as e:
            raise ValueError(f"Table {table_name} not found in DuckDB") from e

    def store_cleansing_report(
        self, dataset_name: str, reports: list[CleansedColumnReport]
    ) -> None:
        """Store cleansing reports in the metadata table."""
        report_json = json.dumps([report.model_dump() for report in reports])
        self.conn.execute(
            """
            INSERT OR REPLACE INTO cleansing_reports (dataset_name, report)
            VALUES (?, ?)
        """,
            [dataset_name, report_json],
        )

    def get_cleansing_report(self, dataset_name: str) -> list[CleansedColumnReport]:
        """Retrieve cleansing reports from the metadata table."""
        result = self.conn.execute(
            "SELECT report FROM cleansing_reports WHERE dataset_name = ?",
            [dataset_name],
        ).fetchone()

        if result:
            reports_data = json.loads(result[0])
            return [CleansedColumnReport(**report) for report in reports_data]
        return []


class AnalystDatasetDuckDB:
    def __init__(self, db_handler: DuckDBHandler):
        self.db = db_handler

    def register_dataset(self, dataset: AnalystDataset) -> None:
        """Register an AnalystDataset in DuckDB."""
        self.db.register_dataframe(dataset.data.df, dataset.name)

    def register_cleansed_dataset(self, cleansed: CleansedDataset) -> None:
        """Register a CleansedDataset and its reports."""
        # Store the cleansed dataframe
        self.db.register_dataframe(
            cleansed.dataset.data.df, cleansed.name, is_cleansed=True
        )
        # Store the cleansing reports
        self.db.store_cleansing_report(cleansed.name, cleansed.cleaning_report)

    def register_data_dictionary(self, dictionary: DataDictionary) -> None:
        """Register a DataDictionary."""
        # Store the data dictionary dataframe
        self.db.register_dataframe(
            dictionary.to_application_df(), f"{dictionary.name}_dict"
        )

    def get_data_dictionary(self, name: str) -> DataDictionary:
        """Retrieve a DataDictionary from DuckDB."""
        df = self.db.get_dataframe(f"{name}_dict")
        return DataDictionary.from_application_df(df, name)

    def get_dataset(
        self, name: str, cleansed: bool = False
    ) -> AnalystDataset | CleansedDataset:
        """Retrieve a dataset from DuckDB."""
        df = self.db.get_dataframe(name, cleansed)
        base_dataset = AnalystDataset(name=name, data=df)

        if cleansed:
            cleaning_report = self.db.get_cleansing_report(name)
            return CleansedDataset(
                dataset=base_dataset, cleaning_report=cleaning_report
            )
        return base_dataset
