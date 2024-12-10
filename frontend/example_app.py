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

# type: ignore
import sys

import streamlit as st

sys.path.append("..")

from utils.api import client

if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "message_log" not in st.session_state:
    st.session_state.message_log = st.container()

if prompt := st.chat_input("Talk to me."):
    input_message = [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    all_messages = st.session_state.message_history + input_message
    response = str(
        client.chat.completions.create(messages=all_messages, model="gpt-4o")
        .choices[0]
        .message.content
    )
    response_content = [{"role": "assistant", "content": response}]

    st.session_state.message_history += input_message + response_content
    print(st.session_state.message_history)
    for message in st.session_state.message_history:
        st.session_state.message_log.chat_message(message["role"]).write(
            str(message["content"])
        )
