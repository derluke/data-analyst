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


from streamlit_theme import st_theme

from utils.schema import AppInfra

PAGE_ICON = "./datarobot_favicon.png"


def get_page_logo() -> str:
    theme = st_theme()
    logo = "./DataRobot_white.svg"
    if theme and theme.get("base") == "light":
        logo = "./DataRobot_black.svg"
    return logo


def get_database_logo(app_infra: AppInfra) -> str:
    if app_infra.database == "snowflake":
        return "./Snowflake.svg"
    elif app_infra.database == "bigquery":
        return "./Google_Cloud.svg"
