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
import io
import json
import logging
import os
import sys
import time
import traceback
import warnings
from datetime import datetime
from typing import Any, Dict

import pandas as pd
import streamlit as st

sys.path.append("..")


# Import FastAPI functions directly
from utils.rest_api import (
    chat,
    get_business_analysis,
    run_analysis,
    run_charts,
    run_snowflake_analysis,
)
from utils.schema import (
    BusinessAnalysisRequest,
    ChatRequest,
    RunAnalysisRequest,
    RunChartsRequest,
    SnowflakeAnalysisRequest,
)

# Suppress warnings
warnings.filterwarnings("ignore")

# Add after imports, before session state initialization
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize session state variables at the very beginning of the file
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.datasets = {}
    st.session_state.cleansed_data = {}
    st.session_state.data_dictionaries = {}
    st.session_state.chat_messages = []
    st.session_state.chat_input_key = 0
    st.session_state.debug_mode = True
elif "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
elif "chat_input_key" not in st.session_state:
    st.session_state.chat_input_key = 0


# Page config
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="datarobot icon.svg",
    layout="wide",
    initial_sidebar_state="expanded",
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


def clear_chat():
    st.session_state.chat_messages = []
    st.session_state.chat_input_key += 1


# Sidebar with New Chat button only
with st.sidebar:
    st.title("Chat Controls")

    # Add New Chat button with callback
    st.button("New Chat", on_click=clear_chat, use_container_width=True)


# Configure logging with custom formatter
class CustomJsonFormatter(logging.Formatter):
    def format(self, record):
        if hasattr(record, "json_data"):
            try:
                if (
                    isinstance(record.json_data, dict)
                    and "messages" in record.json_data
                ):
                    formatted_messages = []
                    for msg in record.json_data["messages"]:
                        formatted_msg = {
                            "role": msg["role"],
                            "content": (
                                msg["content"].replace("\n", "\\n")[:100] + "..."
                                if len(msg["content"]) > 100
                                else msg["content"]
                            ),
                        }
                        formatted_messages.append(formatted_msg)

                    clean_payload = {
                        "model": record.json_data.get("model", ""),
                        "messages": formatted_messages,
                        "response_format": record.json_data.get("response_format", {}),
                        "stream": record.json_data.get("stream", False),
                    }
                    record.msg = f"\nOpenAI Request Payload:\n{json.dumps(clean_payload, indent=2)}"
                else:
                    record.msg = f"\n{json.dumps(record.json_data, indent=2)}"
            except (TypeError, ValueError) as e:
                record.msg = f"Error formatting JSON: {str(e)}\nOriginal data: {record.json_data}"
        return super().format(record)


# Configure logging
console_handler = logging.StreamHandler()
console_handler.setFormatter(CustomJsonFormatter())
root_logger = logging.getLogger()
root_logger.handlers = [console_handler]
logger = logging.getLogger("DataAnalystApp")


# Helper functions
def format_json(obj: Any) -> str:
    try:
        if hasattr(obj, "dict"):
            obj = obj.dict()
        if isinstance(obj, dict) and "messages" in obj:
            formatted_obj = obj.copy()
            for msg in formatted_obj["messages"]:
                if len(msg.get("content", "")) > 100:
                    msg["content"] = msg["content"][:100] + "..."
            return json.dumps(
                formatted_obj, indent=2, sort_keys=True, default=str, ensure_ascii=False
            )
        return json.dumps(
            obj, indent=2, sort_keys=True, default=str, ensure_ascii=False
        )
    except Exception as e:
        return f"Error formatting JSON: {str(e)}\nOriginal object: {str(obj)}"


def format_dataframe_info(df: pd.DataFrame, name: str) -> str:
    buffer = io.StringIO()
    df.info(buf=buffer)
    return f"""
DataFrame: {name}
Shape: {df.shape}
Columns: {', '.join(df.columns)}
Info:
{buffer.getvalue()}
Sample (first 5 rows):
{df.head().to_string()}
"""


# Add logging wrapper for API calls
def log_api_call(func):
    async def wrapper(*args, **kwargs):
        request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        separator = f"\n{'='*80}\n"

        logger.info(
            f"{separator}API CALL START: {func.__name__} [{request_id}]{separator}"
        )

        try:
            formatted_args = [
                arg.dict() if hasattr(arg, "dict") else arg for arg in args
            ]
            formatted_kwargs = {
                k: v.dict() if hasattr(v, "dict") else v for k, v in kwargs.items()
            }

            input_log = f"""
INPUT PARAMETERS [{request_id}]
------------------------
Function: {func.__name__}
Timestamp: {datetime.now().isoformat()}

Arguments:
{format_json(formatted_args)}

Keyword Arguments:
{format_json(formatted_kwargs)}
"""
            logger.debug(input_log)

            start_time = time.time()
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time

            if hasattr(result, "request_options"):
                request_options = result.request_options
                formatted_options = {
                    "method": request_options.get("method"),
                    "url": request_options.get("url"),
                    "files": request_options.get("files"),
                    "json_data": request_options.get("json_data", {}),
                }
                logger.debug(
                    f"""
Request options:
{json.dumps(formatted_options, indent=2, ensure_ascii=False)}
"""
                )

            output_log = f"""
OUTPUT RESULTS [{request_id}]
------------------------
Function: {func.__name__}
Execution Time: {execution_time:.2f} seconds

Response:
{format_json(result)}
"""
            logger.debug(output_log)

            logger.info(
                f"{separator}API CALL COMPLETE: {func.__name__} [{request_id}]{separator}"
            )
            return result

        except Exception as e:
            error_log = f"""
ERROR IN API CALL [{request_id}]
------------------------
Function: {func.__name__}
Error Type: {type(e).__name__}
Error Message: {str(e)}

Stack Trace:
"""
            logger.error(error_log, exc_info=True)
            raise

    return wrapper


# Wrap API functions with logging

chat = log_api_call(chat)
run_analysis = log_api_call(run_analysis)
run_charts = log_api_call(run_charts)
get_business_analysis = log_api_call(get_business_analysis)


# Add enhanced error logging function
def log_error_details(error: Exception, context: Dict[str, Any]) -> None:
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


# Update process_chat_and_analysis with enhanced error handling
async def process_chat_and_analysis(question: str, chat_messages: list) -> None:
    error_context = {"question": question, "chat_history_length": len(chat_messages)}

    try:
        # Create placeholder containers in desired order
        with st.chat_message("assistant", avatar="bot.jpg"):
            message_placeholder = st.empty()

            # Create containers for each section
            bottom_line_container = st.container()
            analysis_container = st.container()
            charts_container = st.container()
            insights_container = st.container()
            followup_container = st.container()

            # Initialize the assistant message structure
            assistant_message = {"role": "assistant", "content": "", "components": []}

            # Get initial chat response
            try:
                chat_request = ChatRequest(messages=chat_messages)
                chat_response = await chat(chat_request)
                message_content = chat_response.get("response", "")
                message_placeholder.markdown(message_content)
                assistant_message["content"] = message_content
            except Exception as e:
                message_content = "Something went wrong. Check your connection to the Internet and to DataRobot. The LLM could be out of quota. Try your question again or start a new chat."
                message_placeholder.markdown(message_content)
                assistant_message["content"] = message_content
                logger.error(f"Error in chat function: {str(e)}", exc_info=True)

            # Run analysis with enhanced error capture
            analysis_result = None
            with st.spinner("Running analysis..."):
                try:
                    if st.session_state.get("data_source") == "snowflake":
                        # Use Snowflake analysis
                        # Convert DataFrames to dictionary format
                        data_dict = {}
                        for name, df in st.session_state.datasets.items():
                            # Convert DataFrame to records format and ensure each record is a dictionary
                            records = df.to_dict("records")
                            data_dict[name] = records

                        # Convert data dictionary to proper format
                        dict_data = {}
                        for (
                            name,
                            dict_list,
                        ) in st.session_state.data_dictionaries.items():
                            if isinstance(dict_list, list):
                                dict_data[name] = {
                                    "columns": [d.get("column") for d in dict_list],
                                    "descriptions": [
                                        d.get("description") for d in dict_list
                                    ],
                                    "data_types": [
                                        d.get("data_type") for d in dict_list
                                    ],
                                }

                        analysis_request = SnowflakeAnalysisRequest(
                            data=data_dict,
                            dictionary=dict_data,
                            question=chat_response.get("enhanced_question", question),
                            warehouse=os.getenv("warehouse"),
                            database=os.getenv("database"),
                            schema=os.getenv("schema"),
                        )
                        analysis_result = await run_snowflake_analysis(analysis_request)
                    else:
                        # Use regular analysis
                        # Convert DataFrames to proper dictionary format
                        formatted_data = {}
                        for name, df in st.session_state.datasets.items():
                            # Convert DataFrame to list of dictionaries where each row is a dictionary
                            # with column names as keys
                            formatted_data[name] = df.to_dict("records")

                        analysis_request = RunAnalysisRequest(
                            data=formatted_data,
                            dictionary=st.session_state.data_dictionaries,
                            question=chat_response.get("enhanced_question", question),
                        )
                        analysis_result = await run_analysis(analysis_request)

                    # Store analysis results in components
                    assistant_message["components"].append(
                        {"type": "analysis", "data": analysis_result}
                    )

                    # Display analysis results
                    with analysis_container:
                        if "code" in analysis_result:
                            with st.expander("Analysis Code", expanded=False):
                                # Use SQL language highlighting for Snowflake mode
                                language = (
                                    "sql"
                                    if st.session_state.get("data_source")
                                    == "snowflake"
                                    else "python"
                                )
                                st.code(analysis_result["code"], language=language)
                        if "data" in analysis_result:
                            with st.expander("Analysis Results", expanded=True):
                                if isinstance(analysis_result["data"], list):
                                    df = pd.DataFrame(analysis_result["data"])
                                    st.dataframe(df, use_container_width=True)
                                else:
                                    st.write(analysis_result["data"])
                except Exception as e:
                    error_context.update({"component": "analysis"})
                    log_error_details(e, error_context)

            # Process charts and business analysis concurrently
            if analysis_result and "data" in analysis_result:
                try:
                    chart_df = (
                        pd.DataFrame(analysis_result["data"])
                        if isinstance(analysis_result["data"], list)
                        else pd.DataFrame([analysis_result["data"]])
                    )

                    # Prepare requests
                    chart_request = RunChartsRequest(
                        data=chart_df.to_dict("records"),
                        question=chat_response.get("enhanced_question", question),
                    )

                    business_request = BusinessAnalysisRequest(
                        data=chart_df.to_dict("records"),
                        dictionary=[
                            {
                                "column": col,
                                "description": "Analysis result column",
                                "data_type": str(chart_df[col].dtype),
                            }
                            for col in chart_df.columns
                        ],
                        question=chat_response.get("enhanced_question", question),
                    )

                    # Create and start tasks immediately
                    charts_task = asyncio.create_task(run_charts(chart_request))
                    business_task = asyncio.create_task(
                        get_business_analysis(business_request)
                    )

                    # Process both tasks as they complete
                    with st.spinner("Generating analysis..."):
                        # Create tasks list
                        tasks = [charts_task, business_task]

                        # Wait for each task to complete
                        for coro in asyncio.as_completed(tasks):
                            try:
                                result = await coro

                                # Determine which task completed by checking the result structure
                                if isinstance(result, dict) and (
                                    "fig1" in result or "fig2" in result
                                ):
                                    # Charts task completed
                                    assistant_message["components"].append(
                                        {"type": "charts", "data": result}
                                    )
                                    with charts_container:
                                        if "fig1" in result:
                                            st.plotly_chart(
                                                result["fig1"], use_container_width=True
                                            )
                                        if "fig2" in result:
                                            st.plotly_chart(
                                                result["fig2"], use_container_width=True
                                            )

                                else:
                                    # Business analysis task completed
                                    assistant_message["components"].append(
                                        {"type": "business_analysis", "data": result}
                                    )

                                    with bottom_line_container:
                                        with st.expander("Bottom Line", expanded=True):
                                            st.markdown(
                                                result.get("bottom_line", "").replace(
                                                    "$", "\\$"
                                                )
                                            )

                                    with insights_container:
                                        if result.get("additional_insights"):
                                            with st.expander(
                                                "Additional Insights", expanded=True
                                            ):
                                                st.markdown(
                                                    result[
                                                        "additional_insights"
                                                    ].replace("$", "\\$")
                                                )

                                    with followup_container:
                                        if result.get("follow_up_questions"):
                                            with st.expander(
                                                "Follow-up Questions", expanded=True
                                            ):
                                                for q in result["follow_up_questions"]:
                                                    st.markdown(
                                                        f"- {q}".replace("$", r"\$")
                                                    )

                            except Exception as e:
                                # Determine which task failed by checking remaining tasks
                                task_type = (
                                    "charts" if charts_task in tasks else "business"
                                )
                                error_context.update(
                                    {
                                        "component": f"concurrent_processing_{task_type}",
                                        "task_type": task_type,
                                    }
                                )
                                log_error_details(e, error_context)

                                # Display error for the specific component
                                if task_type == "charts":
                                    with charts_container:
                                        st.error(f"Error generating charts: {str(e)}")
                                else:
                                    with bottom_line_container:
                                        st.error(
                                            f"Error generating business analysis: {str(e)}"
                                        )

                except Exception as e:
                    error_context.update({"component": "concurrent_processing_setup"})
                    log_error_details(e, error_context)
                    st.error(f"Error setting up analysis: {str(e)}")

            # Store the complete message in session state
            st.session_state.chat_messages.append(assistant_message)

    except Exception as e:
        error_context["component"] = "main_process"
        log_error_details(e, error_context)
        st.error(f"Error processing chat and analysis: {str(e)}")


# Main page content (Chat Interface)
st.image("datarobot logo.svg", width=200)

if not st.session_state.cleansed_data:
    st.info("Please upload and process data using the sidebar before starting the chat")
else:
    # Display chat history
    for message in st.session_state.chat_messages:
        # Set avatar based on role
        avatar = "bot.jpg" if message["role"] == "assistant" else "you.jpg"

        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

            # Add components in the same structure as the original response
            if "components" in message:
                # Find business analysis component
                business_analysis = next(
                    (
                        comp
                        for comp in message["components"]
                        if comp["type"] == "business_analysis"
                    ),
                    {"data": {}},
                )

                # Display sections using empty containers
                with st.expander("Bottom Line", expanded=True):
                    bottom_line = business_analysis["data"].get("bottom_line", "")
                    st.markdown(
                        bottom_line.replace("$", "\\$")
                        if bottom_line
                        else "No bottom line available"
                    )

                # Display analysis results
                with st.expander("Analysis Results", expanded=True):
                    analysis_component = next(
                        (
                            comp
                            for comp in message["components"]
                            if comp["type"] == "analysis"
                        ),
                        {"data": {}},
                    )

                    if "code" in analysis_component["data"]:
                        st.code(analysis_component["data"]["code"], language="python")

                    analysis_data = analysis_component["data"].get("data", None)
                    if analysis_data:
                        if isinstance(analysis_data, list):
                            df = pd.DataFrame(analysis_data)
                            st.dataframe(df, use_container_width=True)
                        else:
                            st.write(analysis_data)
                    else:
                        st.markdown("No analysis results available")

                # Display charts
                with st.expander("Charts", expanded=True):
                    charts_component = next(
                        (
                            comp
                            for comp in message["components"]
                            if comp["type"] == "charts"
                        ),
                        {"data": {}},
                    )

                    if "fig1" in charts_component["data"]:
                        st.plotly_chart(
                            charts_component["data"]["fig1"], use_container_width=True
                        )
                    if "fig2" in charts_component["data"]:
                        st.plotly_chart(
                            charts_component["data"]["fig2"], use_container_width=True
                        )
                    if not charts_component["data"]:
                        st.markdown("No charts available")

                # Display additional insights
                with st.expander("Additional Insights", expanded=True):
                    insights = business_analysis["data"].get("additional_insights", "")
                    st.markdown(
                        insights.replace("$", "\\$")
                        if insights
                        else "No additional insights available"
                    )

                # Display follow-up questions
                with st.expander("Follow-up Questions", expanded=True):
                    questions = business_analysis["data"].get("follow_up_questions", [])
                    if questions:
                        for q in questions:
                            st.markdown(f"- {q}".replace("$", r"\$"))
                    else:
                        st.markdown("No follow-up questions available")

    # Chat input
    if question := st.chat_input("Ask a question about your data"):
        valid_messages = [
            msg
            for msg in st.session_state.chat_messages
            if isinstance(msg, dict)
            and msg.get("role") in ["user", "assistant", "system"]
            and msg.get("content", "").strip()
        ]

        valid_messages.append({"role": "user", "content": question})
        chat_request = ChatRequest(messages=valid_messages)
        chat_response = asyncio.run(chat(chat_request))

        enhanced_question = chat_response.get("enhanced_user_message", question)
        user_message = {"role": "user", "content": enhanced_question}
        st.session_state.chat_messages.append(user_message)

        # Display user message with custom avatar
        with st.chat_message("user", avatar="you.jpg"):
            st.markdown(enhanced_question)

        # Process chat and display assistant response
        asyncio.run(process_chat_and_analysis(enhanced_question, valid_messages))
