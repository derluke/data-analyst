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

import os

import pytest
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By

from tests.e2e.selectors import Selectors
from tests.e2e.utils import (
    PROCESSING_TIMEOUT,
    click_element,
    wait_for_element_to_be_visible,
    wait_for_element_to_disappear,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("FRONTEND_TYPE") != "react",
    reason="Skipping because FRONTEND_TYPE is not 'react'",
)


def assert_data_processed(browser: webdriver.Chrome) -> None:
    assert wait_for_element_to_be_visible(
        browser,
        *Selectors.DATA_PROCESSED_BADGE,
        PROCESSING_TIMEOUT,
    )


def dismiss_welcome_modal(browser: webdriver.Chrome) -> None:
    try:
        # Wait for the welcome modal to be visible
        welcome_modal = wait_for_element_to_be_visible(
            browser,
            *Selectors.WELCOME_MODAL_CLOSE_BUTTON,
        )

        if welcome_modal:
            welcome_modal.click()

    except TimeoutException:
        # Element not found, just proceed
        pass


def clear_all_data(browser: webdriver.Chrome) -> None:
    click_element(
        browser,
        *Selectors.CLEAR_DATASETS_BUTTON,
    )

    assert wait_for_element_to_disappear(
        browser,
        *Selectors.DATA_PROCESSED_BADGE,
        PROCESSING_TIMEOUT,
    )


def load_from_database(browser: webdriver.Chrome, dataset: str) -> None:
    click_element(
        browser,
        *Selectors.ADD_DATA_BUTTON,
    )

    click_element(
        browser,
        *Selectors.DATA_BASE_LABEL,
    )

    click_element(
        browser,
        *Selectors.ADD_DATA_MULTISELECT,
    )

    click_element(
        browser,
        By.CSS_SELECTOR,
        f'[data-testid="multi-select-option-{dataset}"]',
    )

    click_element(
        browser,
        *Selectors.SELECT_ITEMS_BUTTON,
    )

    click_element(
        browser,
        *Selectors.SAVE_SELECTIONS_BUTTON,
    )

    wait_for_element_to_disappear(
        browser,
        *Selectors.SAVE_SELECTIONS_BUTTON,
    )

    assert_data_processed(browser)


@pytest.mark.usefixtures("check_if_logged_in")
def test_app_loaded(browser: webdriver.Chrome, get_app_url: str) -> None:
    browser.get(get_app_url)

    dismiss_welcome_modal(browser)

    assert wait_for_element_to_be_visible(
        browser,
        *Selectors.H1_TALK_TO_MY_DATA,
        PROCESSING_TIMEOUT,
    )


@pytest.mark.usefixtures("check_if_logged_in")
def test_data_dictionary_loaded(browser: webdriver.Chrome, get_app_url: str) -> None:
    browser.get(get_app_url)

    dismiss_welcome_modal(browser)

    wait_for_element_to_be_visible(
        browser,
        *Selectors.H1_TALK_TO_MY_DATA,
    )

    # Load the dataset from the database
    load_from_database(browser, "LENDING_CLUB_PROFILE")

    # Remove all datasets
    clear_all_data(browser)


@pytest.mark.usefixtures("check_if_logged_in")
def test_chat_message(browser: webdriver.Chrome, get_app_url: str) -> None:
    browser.get(get_app_url)

    dismiss_welcome_modal(browser)

    # Wait for the main page to load
    wait_for_element_to_be_visible(
        browser,
        *Selectors.H1_TALK_TO_MY_DATA,
    )

    # Load the dataset from the database
    load_from_database(browser, "LENDING_CLUB_PROFILE")

    # Navigate to the chats page
    click_element(
        browser,
        *Selectors.MAIN_NAV_CHATS,
    )

    click_element(
        browser,
        *Selectors.INITIAL_PROMPT_INPUT,
    )

    # type a message in the chat input
    chat_input = wait_for_element_to_be_visible(
        browser,
        *Selectors.INITIAL_PROMPT_INPUT,
    )

    if chat_input:
        chat_input.send_keys(
            "What is the most common medical specialty of the physician?"
        )
    else:
        pytest.fail("Chat input element not found")

    click_element(
        browser,
        *Selectors.INITIAL_PROMPT_SUBMIT_BUTTON,
    )

    # wait for the response to be visible for all the tabs (Summary, Insights, Code)
    wait_for_element_to_be_visible(
        browser,
        *Selectors.SUMMARY_TAB_SUCCESS,
        PROCESSING_TIMEOUT,
    )
    wait_for_element_to_be_visible(
        browser,
        *Selectors.INSIGHTS_TAB_SUCCESS,
        PROCESSING_TIMEOUT,
    )
    wait_for_element_to_be_visible(
        browser,
        *Selectors.CODE_TAB_SUCCESS,
        PROCESSING_TIMEOUT,
    )

    # Delete all chats
    click_element(
        browser,
        *Selectors.DELETE_ALL_CHATS_BUTTON,
    )

    # Navigate back to the Data page
    click_element(
        browser,
        *Selectors.MAIN_NAV_DATA,
    )

    # Remove all datasets
    clear_all_data(browser)
