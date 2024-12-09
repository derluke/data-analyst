import base64

import pandas as pd
import pytest
import requests
from dataanalyst.api import app
from dataanalyst.data_model import (
    Analysis,
    AnalysisResult,
    AnalysisSummary,
    CodeSnippetResult,
    CSVSpec,
    DataDictionary,
    DataFrameJSON,
    DataOverview,
    DRCatalogSpec,
    ExceptionResult,
    PlotlyJSON,
    SnowflakeTableSpec,
    SQLSnippetResult,
    URLSpec,
)
from dataanalyst.settings import DRCredentials
from datarobotx.idp.datasets import get_or_create_dataset_from_df
from fastapi.testclient import TestClient

client = TestClient(app)


@pytest.fixture(scope="module")
def diabetes_dataset_url():
    return "https://s3.amazonaws.com/datarobot_public_datasets/10k_diabetes_20.csv"


@pytest.fixture(scope="module")
def dr_credentials():
    return DRCredentials()


@pytest.fixture(scope="module")
def diabetes_dataset_id(diabetes_dataset_url, dr_credentials):
    df = pd.read_csv(
        diabetes_dataset_url,
    )
    return get_or_create_dataset_from_df(
        dr_credentials.endpoint,
        dr_credentials.token,
        "10k_diabetes_20.csv",
        df,
    )


@pytest.fixture
def url_spec(diabetes_dataset_url):
    return URLSpec(url=diabetes_dataset_url)


@pytest.fixture
def csv_spec(diabetes_dataset_url):
    resp = requests.get(diabetes_dataset_url)
    return CSVSpec(
        filename="10k_diabetes_20.csv",
        raw_bytes=base64.b64encode(resp.content).decode("utf-8"),
    )


@pytest.fixture
def catalog_spec(diabetes_dataset_id, dr_credentials):
    return DRCatalogSpec(token=dr_credentials.token, dataset_id=diabetes_dataset_id)


@pytest.fixture
def db_params():
    return {"warehouse": "DEMO_WH", "database": "DEMO", "schema_": "INSTACART"}


@pytest.fixture
def snowflake_spec_orders(db_params):
    return SnowflakeTableSpec(**db_params, table="ORDERS")


@pytest.fixture
def snowflake_spec_order_products(db_params):
    return SnowflakeTableSpec(**db_params, table="ORDER_PRODUCTS")


@pytest.fixture(params=["url", "csv", "catalog", "snow"])
def spec_and_q(
    request,
    url_spec,
    csv_spec,
    catalog_spec,
    snowflake_spec_orders,
    snowflake_spec_order_products,
):
    diabetes_q = "Trends in readmission?"
    instacart_q = "Trends in order time?"
    if request.param == "url":
        return [url_spec.model_dump()], diabetes_q
    elif request.param == "csv":
        return [csv_spec.model_dump()], diabetes_q
    elif request.param == "catalog":
        return [catalog_spec.model_dump()], diabetes_q
    elif request.param == "snow":
        return [
            snowflake_spec_orders.model_dump(),
            snowflake_spec_order_products.model_dump(),
        ], instacart_q


@pytest.fixture(params=["chart", "table"])
def data_and_analysis(request, spec_and_q):
    spec, q = spec_and_q
    if spec[0]["data_type"] == "snowflake_table":
        analysis_name = "Distribution of orders by time"
    else:
        analysis_name = "Readmissions rate by gender"
    analysis = Analysis(
        name=analysis_name,
        output_type=request.param,
    ).model_dump()
    return {
        "data": spec,
        "analysis": analysis,
        "question": q,
    }


def test_overview(spec_and_q):
    spec, q = spec_and_q
    resp = client.post("/dataOverview", json=spec)
    assert resp.status_code == 200
    overviews = [DataOverview(**overview) for overview in resp.json()]
    for overview in overviews:
        assert len(overview.first_rows)
        assert len(overview.column_descriptions)


def test_data_dict(spec_and_q):
    spec, q = spec_and_q
    resp = client.post("/dataDictionary", json=spec)
    assert resp.status_code == 200
    dicts = [DataDictionary(**dict) for dict in resp.json()]
    for d in dicts:
        assert len(d.dictionary)
        assert len(d.dictionary[0].definition)


def test_potential_analyses(spec_and_q):
    spec, q = spec_and_q
    resp = client.post("/potentialAnalyses", json={"data": spec, "question": q})
    assert resp.status_code == 200
    analyses = [Analysis(**analysis) for analysis in resp.json()]
    for analysis in analyses:
        assert len(analysis.name)
        assert len(analysis.output_type)


def test_analysis_result(data_and_analysis):
    resp = client.post("/analysisResult", json=data_and_analysis)
    assert resp.status_code == 200
    analysis_result = AnalysisResult(**resp.json())
    assert analysis_result.success
    assert len(analysis_result.results)
    if any(
        [spec["data_type"] == "snowflake_table" for spec in data_and_analysis["data"]]
    ):
        assert any(
            [isinstance(result, DataFrameJSON) for result in analysis_result.results]
        )
        assert any(
            [isinstance(result, SQLSnippetResult) for result in analysis_result.results]
        )
    elif data_and_analysis["analysis"]["output_type"] == "table":
        assert any(
            [isinstance(result, DataFrameJSON) for result in analysis_result.results]
        )

    if data_and_analysis["analysis"]["output_type"] == "chart":
        assert any(
            [
                isinstance(result, CodeSnippetResult)
                for result in analysis_result.results
            ]
        )
        assert any(
            [isinstance(result, PlotlyJSON) for result in analysis_result.results]
        )


def test_summary(data_and_analysis):
    resp = client.post("/analysisResult", json=data_and_analysis)
    analysis_result = AnalysisResult(**resp.json())
    body = {
        "question": data_and_analysis["question"],
        "data": data_and_analysis["data"],
        "analysis_results": [analysis_result.model_dump()],
    }

    resp = client.post(
        "/summary",
        json=body,
    )
    assert resp.status_code == 200
    summary = AnalysisSummary(**resp.json())
    assert len(summary.summary)
    assert len(summary.next_questions)


@pytest.fixture()
def bad_python_runner(monkeypatch):
    import dataanalyst.base

    dataanalyst.base._MEMORY.clear(warn=False)

    def mock_runner(*args, **kwargs):
        raise ValueError("Bad python code")

    monkeypatch.setattr(dataanalyst.analyze, "run_python", mock_runner)


def test_failed_py_generation(data_and_analysis, bad_python_runner):
    if data_and_analysis["analysis"]["output_type"] == "table" and any(
        [spec["data_type"] == "snowflake_table" for spec in data_and_analysis["data"]]
    ):
        pytest.skip("Bad python generation N/A")
    resp = client.post(
        "/analysisResult",
        json=data_and_analysis,
    )
    assert resp.status_code == 200
    analysis_result = AnalysisResult(**resp.json())
    assert not analysis_result.success
    assert any(
        [isinstance(result, ExceptionResult) for result in analysis_result.results]
    )


@pytest.fixture()
def bad_sql_runner(monkeypatch):
    import dataanalyst.base

    dataanalyst.base._MEMORY.clear(warn=False)

    def mock_runner(*args, **kwargs):
        raise ValueError("Bad sql code")

    monkeypatch.setattr(dataanalyst.analyze, "run_sql", mock_runner)


def test_failed_sql_generation(data_and_analysis, bad_sql_runner):
    if not any(
        [spec["data_type"] == "snowflake_table" for spec in data_and_analysis["data"]]
    ):
        pytest.skip("Bad sql generation N/A")
    resp = client.post(
        "/analysisResult",
        json=data_and_analysis,
    )
    assert resp.status_code == 200
    analysis_result = AnalysisResult(**resp.json())
    assert not analysis_result.success
    assert any(
        [isinstance(result, ExceptionResult) for result in analysis_result.results]
    )
