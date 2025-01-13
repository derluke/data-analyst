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

from pathlib import Path
from typing import List, Tuple

import datarobot as dr
import pulumi

from infra.common.schema import ApplicationSourceArgs

from .settings_main import project_name

application_path = Path("frontend/")

app_source_args = ApplicationSourceArgs(
    resource_name=f"Data Analyst App Source [{project_name}]",
    replicas=2,
).model_dump(mode="json", exclude_none=True)


def ensure_app_settings(app_id: str) -> None:
    try:
        dr.client.get_client().patch(
            f"customApplications/{app_id}/",
            json={"allowAutoStopping": True},
        )
    except Exception:
        pulumi.warn("Patching app unsuccessful.")
    return version_id


def ensure_app_source_settings(source_id: str, version_id: str):
    try:
        dr.client.get_client().patch(
            url=f"customApplicationSources/{source_id}/versions/{version_id}/",
            json={
                "resources": {
                    "sessionAffinity": True,
                    "resourceLabel": "cpu.xlarge",
                    "replicas": 2,
                }
            },
        )
    except dr.errors.ClientError:
        pulumi.warn("Patching app source unsuccessful.")
    return version_id


app_resource_name: str = f"Data Analyst Application [{project_name}]"


def get_app_files() -> List[Tuple[str, str]]:
    source_files = [
        (rf"{str(f)}", str(f.relative_to(application_path)).replace("\\", "/"))
        for f in application_path.glob("**/*")
        if f.is_file() and not f.name.endswith(".yaml")
    ]

    source_files.extend(
        [
            (r"utils/__init__.py", r"utils/__init__.py"),
            (r"utils/api.py", r"utils/api.py"),
            (r"utils/credentials.py", r"utils/credentials.py"),
            (r"utils/datetime_helpers.py", r"utils/datetime_helpers.py"),
            (r"utils/errors.py", r"utils/errors.py"),
            (r"utils/prompts.py", r"utils/prompts.py"),
            (r"utils/resources.py", r"utils/resources.py"),
            (r"utils/rest_api.py", r"utils/rest_api.py"),
            (r"utils/schema.py", r"utils/schema.py"),
            (r"utils/logging.py", r"utils/logging.py"),
            (r"utils/snowflake_helpers.py", r"utils/snowflake_helpers.py"),
            (str(application_path / "metadata.yaml"), "metadata.yaml"),
        ]
    )

    return source_files
