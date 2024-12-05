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

from functools import partial
import subprocess
import sys

import datarobot as dr
from openai import OpenAI
from pydantic import ValidationError

sys.path.append("..")

from application.resources import ChatAgentDeployment

try:
    chat_agent_deployment_id = ChatAgentDeployment().id
    deployment_chat_base_url = (
        dr.Client().endpoint + f"/deployments/{chat_agent_deployment_id}/"
    )

    client = OpenAI(api_key=dr.Client().token, base_url=deployment_chat_base_url)
    create_completion = partial(client.chat.completions.create, model="gpt-4o")


except ValidationError as e:
    raise ValueError(
        "Unable to load Deployment ID."
        "If running locally, verify you have selected the correct "
        "stack and that it is active using `pulumi stack output`. "
        "If running in DataRobot, verify your runtime parameters have been set correctly."
    ) from e
