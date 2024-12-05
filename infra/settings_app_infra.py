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

import textwrap
from pathlib import Path
from typing import List, Sequence, Tuple

import datarobot as dr
import pulumi_datarobot as datarobot

from infra.common.schema import ApplicationSourceArgs

from .settings_main import project_name

application_path = Path("frontend/")

app_source_args = ApplicationSourceArgs(
    resource_name=f"Data Analyst App Source [{project_name}]",
).model_dump(mode="json", exclude_none=True)


def ensure_app_settings(app_id: str) -> None:
    dr.client.get_client().patch(
        f"customApplications/{app_id}/",
        json={"allowAutoStopping": True},
    )


app_resource_name: str = f"Data Analyst Application [{project_name}]"


def get_app_files() -> List[Tuple[str, str]]:
    source_files = [
        (str(f), str(f.relative_to(application_path)))
        for f in application_path.glob("**/*")
        if f.is_file() and not f.name.endswith(".yaml")
    ]

    source_files.extend(
        [
            ("application/__init__.py", "application/__init__.py"),
            ("application/api.py", "application/api.py"),
            ("application/credentials.py", "application/credentials.py"),
            ("application/resources.py", "application/resources.py"),
            ("application/schema.py", "application/schema.py"),
            (str(application_path / "metadata.yaml"), "metadata.yaml"),
        ]
    )

    return source_files
