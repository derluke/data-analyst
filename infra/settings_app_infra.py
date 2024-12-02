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


def _prep_metadata_yaml(
    runtime_parameter_values: Sequence[
        datarobot.ApplicationSourceRuntimeParameterValueArgs,
    ],
) -> None:
    from jinja2 import BaseLoader, Environment

    llm_runtime_parameter_specs = "\n".join(
        [
            textwrap.dedent(
                f"""\
            - fieldName: {param.key}
              type: {param.type}"""
            )
            for param in runtime_parameter_values
        ]
    )
    with open(application_path / "metadata.yaml.jinja") as f:
        template = Environment(loader=BaseLoader()).from_string(f.read())

    (application_path / "metadata.yaml").write_text(
        template.render(
            runtime_parameters=llm_runtime_parameter_specs,
        )
    )


def get_app_files(
    runtime_parameter_values: Sequence[
        datarobot.ApplicationSourceRuntimeParameterValueArgs
    ],
) -> List[Tuple[str, str]]:
    _prep_metadata_yaml(runtime_parameter_values)

    source_files = [
        (str(f), str(f.relative_to(application_path)))
        for f in application_path.glob("**/*")
        if f.is_file()
        and f.name != "metadata.yaml.jinja"
        and not f.name.endswith(".yaml")
    ]

    source_files.extend(
        [
            ("dataanalyst/__init__.py", "dataanalyst/__init__.py"),
            ("dataanalyst/analyze.py", "dataanalyst/analyze.py"),
            ("dataanalyst/api.py", "dataanalyst/api.py"),
            ("dataanalyst/base.py", "dataanalyst/base.py"),
            ("dataanalyst/credentials.py", "dataanalyst/credentials.py"),
            ("dataanalyst/data_model.py", "dataanalyst/data_model.py"),
            ("dataanalyst/profile.py", "dataanalyst/profile.py"),
            ("dataanalyst/resources.py", "dataanalyst/resources.py"),
            ("dataanalyst/sample.py", "dataanalyst/sample.py"),
            ("dataanalyst/schema.py", "dataanalyst/schema.py"),
            ("dataanalyst/settings.py", "dataanalyst/settings.py"),
            ("dataanalyst/sql.py", "dataanalyst/sql.py"),
            (str(application_path / "metadata.yaml"), "metadata.yaml"),
        ]
    )

    return source_files
