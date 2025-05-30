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

from selenium.webdriver.common.by import By


class Selectors:
    ADD_DATA_MULTISELECT = (
        By.CSS_SELECTOR,
        '[data-testid="database-table-select"]',
    )
    SELECT_ITEMS_BUTTON = (
        By.CSS_SELECTOR,
        '[data-testid="multi-select-close"]',
    )
    SAVE_SELECTIONS_BUTTON = (
        By.CSS_SELECTOR,
        '[data-testid="add-data-modal-save-button"]',
    )
    DATASET_RAW_ROWS_TAB = (
        By.CSS_SELECTOR,
        '[data-testid="raw-tab"]',
    )
    DATA_PROCESSED_BADGE = (
        By.CSS_SELECTOR,
        '[data-testid="data-processed-badge"]',
    )
    ADD_DATA_BUTTON = (
        By.CSS_SELECTOR,
        '[data-testid="add-data-button"]',
    )
    DATA_BASE_LABEL = (
        By.XPATH,
        "//label[contains(text(), 'Database')]",
    )
    H1_TALK_TO_MY_DATA = (
        By.XPATH,
        "//h1[contains(text(), 'Talk to my data')]",
    )
    CLEAR_DATASETS_BUTTON = (
        By.CSS_SELECTOR,
        '[data-testid="clear-datasets-button"]',
    )
    MAIN_NAV_DATA = (
        By.CSS_SELECTOR,
        '[data-testid="data-menu-option"]',
    )
    MAIN_NAV_CHATS = (
        By.CSS_SELECTOR,
        '[data-testid="chats-menu-option"]',
    )
    INITIAL_PROMPT_INPUT = (
        By.CSS_SELECTOR,
        '[data-testid="initial-prompt-input"]',
    )
    INITIAL_PROMPT_SUBMIT_BUTTON = (
        By.CSS_SELECTOR,
        '[data-testid="send-message-button"]',
    )
    SUMMARY_TAB_SUCCESS = (
        By.CSS_SELECTOR,
        '[data-testid="summary-loading-success"]',
    )
    INSIGHTS_TAB_SUCCESS = (
        By.CSS_SELECTOR,
        '[data-testid="insights-loading-success"]',
    )
    CODE_TAB_SUCCESS = (
        By.CSS_SELECTOR,
        '[data-testid="code-loading-success"]',
    )
    DELETE_ALL_CHATS_BUTTON = (
        By.CSS_SELECTOR,
        '[data-testid="delete-all-chats-button"]',
    )
    WELCOME_MODAL_CLOSE_BUTTON = (
        By.CSS_SELECTOR,
        '[data-testid="welcome-modal-hide-button"]',
    )
