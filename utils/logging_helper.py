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
import time
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Coroutine, ParamSpec, TypeVar

logging.basicConfig(level=logging.INFO)


def get_logger(name: str = "DataAnalystBackend") -> logging.Logger:
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    consoleHandle = logging.StreamHandler()
    consoleHandle.setLevel(logging.INFO)
    consoleHandle.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.propagate = False  # Prevent propagation to root logger
    for handler in logger.handlers:
        logger.removeHandler(handler)
    logger.addHandler(consoleHandle)
    return logger


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


P = ParamSpec("P")
T = TypeVar("T")


def log_api_call(
    func: Callable[P, Coroutine[Any, Any, T]],
) -> Callable[P, Coroutine[Any, Any, T]]:
    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        logger = get_logger()
        request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        separator = f"\n{'=' * 80}\n"
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
            input_log = (
                f"INPUT PARAMETERS [{request_id}]\n"
                "------------------------\n"
                f"Function: {func.__name__}\n"
                f"Timestamp: {datetime.now().isoformat()}\n\n"
                "Arguments:\n"
                f"{format_json(formatted_args)}\n\n"
                "Keyword Arguments:\n"
                f"{format_json(formatted_kwargs)}\n"
            )
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
                logger.debug(f"Request options:\n{format_json(formatted_options)}\n")

            output_log = (
                f"OUTPUT RESULTS [{request_id}]\n"
                "------------------------\n"
                f"Function: {func.__name__}\n"
                f"Execution Time: {execution_time:.2f} seconds\n\n"
                "Response:\n"
                f"{format_json(result)}\n"
            )
            logger.debug(output_log)
            logger.info(
                f"{separator}API CALL COMPLETE: {func.__name__} [{request_id}]{separator}"
            )
            return result
        except Exception as e:
            error_log = (
                f"ERROR IN API CALL [{request_id}]\n"
                "------------------------\n"
                f"Function: {func.__name__}\n"
                f"Error Type: {type(e).__name__}\n"
                f"Error Message: {str(e)}\n\n"
                "Stack Trace:\n"
            )
            logger.error(error_log, exc_info=True)
            raise

    return wrapper
