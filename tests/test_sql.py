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

import time

import pytest
from dataanalyst.data_model import SnowflakeTableSpec
from dataanalyst.settings import App
from dataanalyst.sql import run_sql


@pytest.fixture()
def app_settings():
    return App()


@pytest.fixture
def simple_request():
    return "select count(*) from INSTACART.ORDER_PRODUCTS"


# TODO: parametrize for dbx
@pytest.fixture
def snowflake_spec():
    raw = {
        "warehouse": "DEMO_WH",
        "database": "DEMO",
        "schema_": "foobar",
        "table": "foobar",
    }
    return SnowflakeTableSpec(**raw)


@pytest.fixture()
def invalid_query_request():
    return "select foobar"


@pytest.fixture()
def long_result_request():
    return "select * from INSTACART.ORDER_PRODUCTS"


@pytest.fixture()
def long_running_request():
    return "CALL SYSTEM$WAIT(100);"


@pytest.fixture()
def row_limit(app_settings):
    return app_settings.sql_row_limit


@pytest.fixture()
def timeout(app_settings):
    return app_settings.sql_statement_timeout_secs


class TestSQL:
    def test_simple_request(self, simple_request, snowflake_spec):
        df = run_sql(query=simple_request, specs=[snowflake_spec], max_result_rows=10)
        assert not df.empty

    def test_invalid_query(self, invalid_query_request, snowflake_spec):
        with pytest.raises(BaseException):
            run_sql(
                query=invalid_query_request, specs=[snowflake_spec], max_result_rows=10
            )

    def test_row_limit(self, long_result_request, row_limit, snowflake_spec):
        df = run_sql(
            query=long_result_request, specs=[snowflake_spec], max_result_rows=row_limit
        )
        assert len(df) <= row_limit

    def test_timeout(self, long_running_request, timeout, snowflake_spec):
        start_time = time.time()
        with pytest.raises(BaseException):
            run_sql(
                query=long_running_request, specs=[snowflake_spec], max_result_rows=10
            )
        elapsed = time.time() - start_time

        assert elapsed < timeout + 5
