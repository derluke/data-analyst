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

import logging
import os

from fastapi import APIRouter
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from utils.rest_api import app

# Configure logging to filter out the health check logs
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Filter out "GET /" health check logs
        return "GET / HTTP/1.1" not in record.getMessage()


logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

SCRIPT_NAME = os.environ.get("SCRIPT_NAME", "")
IS_PROD = os.environ.get("IS_PROD", "True")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")
base_router = APIRouter()


# # client side routes
if IS_PROD:

    @base_router.get(f"{SCRIPT_NAME}/data")
    @base_router.get(f"{SCRIPT_NAME}/chats")
    @base_router.get(f"{SCRIPT_NAME}/chats/{{chat_id}}")
    @base_router.get(f"{SCRIPT_NAME}/")
    async def serve_root() -> FileResponse:
        """Serve the React index.html for the root route."""
        print(SCRIPT_NAME)
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.include_router(base_router)

if IS_PROD:
    # Important to be last so that we fall back to the static files if the route is not found
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
