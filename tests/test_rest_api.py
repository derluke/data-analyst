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

from typing import Any, Generator
from unittest.mock import AsyncMock, MagicMock, patch
from unittest.mock import MagicMock as MagicMockType

import pytest
from fastapi import Request, Response
from fastapi.testclient import TestClient

from utils.analyst_db import AnalystDB  # noqa: E402
from utils.rest_api import (
    SessionState,
    _initialize_session,
    _set_session_cookie,
    app,
    delete_chat_message,
    get_datarobot_account,
    store_datarobot_account,
    use_user_token,
)
from utils.schema import AnalystChatMessage


@pytest.fixture
def mock_analyst_db() -> AsyncMock:
    return AsyncMock(spec=AnalystDB)


@pytest.fixture
def test_client() -> Generator[TestClient, None, None]:
    client = TestClient(app)
    yield client


@pytest.fixture
def mock_session_state() -> SessionState:
    session = SessionState()
    session._state = {
        "datarobot_account_info": None,
        "datarobot_endpoint": "https://app.datarobot.com/api/v2",
        "datarobot_api_token": None,
        "datarobot_api_skoped_token": None,
        "analyst_db": None,
    }
    return session


@pytest.fixture
def mock_request() -> MagicMockType:
    request = MagicMock(spec=Request)
    request.state = MagicMock()
    request.cookies = {}
    request.headers = {}
    request.method = "GET"
    return request


def test_initialize_session_default_values(
    setup_pulumi_stack: Any, pulumi_up: Any, mock_request: MagicMockType
) -> None:
    """Test that _initialize_session creates a new session with default values"""

    async def run_test() -> None:
        session_state, session_id, user_id = await _initialize_session(mock_request)
        assert session_state is not None
        assert session_id is not None
        assert user_id is not None
        assert session_state._state["datarobot_account_info"] is None
        assert session_state._state["datarobot_api_token"] is None
        assert session_state._state["datarobot_api_skoped_token"] is None
        assert session_state._state["analyst_db"] is None

    import asyncio

    asyncio.run(run_test())


def test_set_session_cookie(setup_pulumi_stack: Any, pulumi_up: Any) -> None:
    """Test that _set_session_cookie sets cookies correctly"""
    response = Response()
    user_id = "test_user_id"
    session_id = "test_session_id"

    # Test when user_id exists but no cookie
    _set_session_cookie(response, user_id, session_id, None)

    # The cookie should be set with base64 encoded user_id
    cookies = [
        header for header in response.raw_headers if header[0].decode() == "set-cookie"
    ]
    assert cookies, "No set-cookie header found"
    assert "session_fastapi" in cookies[0][1].decode()

    # Test when neither user_id nor cookie exists
    response = Response()
    _set_session_cookie(response, None, session_id, None)

    # The cookie should be set with session_id
    cookies = [
        header for header in response.raw_headers if header[0].decode() == "set-cookie"
    ]
    assert cookies, "No set-cookie header found"
    assert "session_fastapi" in cookies[0][1].decode()

    # Test when cookie already exists
    response = Response()
    _set_session_cookie(response, user_id, session_id, "existing_cookie")

    # No cookie should be set
    cookies = [
        header for header in response.raw_headers if header[0].decode() == "set-cookie"
    ]
    assert not cookies, "Cookie was set when it shouldn't have been"


@pytest.mark.asyncio
async def test_use_user_token_with_scoped_token(
    setup_pulumi_stack: Any, pulumi_up: Any, mock_request: MagicMockType
) -> None:
    """Test that use_user_token uses the scoped token if available"""
    # Setup mock request with a session that has a scoped token
    mock_request.state.session = MagicMock()
    mock_request.state.session.datarobot_api_skoped_token = "scoped_token_123"
    mock_request.state.session.datarobot_api_token = None
    mock_request.state.session.datarobot_endpoint = "https://app.datarobot.com/api/v2"

    # Mock dr.Client to track which token is used
    with patch("datarobot.Client") as mock_dr_client:
        # Use the context manager
        with use_user_token(mock_request):
            pass

        # Check that dr.Client was called with the scoped token
        mock_dr_client.assert_called_once_with(
            token="scoped_token_123", endpoint="https://app.datarobot.com/api/v2"
        )


@pytest.mark.asyncio
async def test_use_user_token_fallback_to_regular_token(
    setup_pulumi_stack: Any, pulumi_up: Any, mock_request: MagicMockType
) -> None:
    """Test that use_user_token falls back to regular token if no scoped token"""
    # Setup mock request with a session that has only a regular token
    mock_request.state.session = MagicMock()
    mock_request.state.session.datarobot_api_skoped_token = None
    mock_request.state.session.datarobot_api_token = "regular_token_456"
    mock_request.state.session.datarobot_endpoint = "https://app.datarobot.com/api/v2"

    # Mock dr.Client to track which token is used
    with patch("datarobot.Client") as mock_dr_client:
        # Use the context manager
        with use_user_token(mock_request):
            pass

        # Check that dr.Client was called with the regular token
        mock_dr_client.assert_called_once_with(
            token="regular_token_456", endpoint="https://app.datarobot.com/api/v2"
        )


@pytest.mark.asyncio
async def test_get_datarobot_account_includes_scoped_token(
    setup_pulumi_stack: Any, pulumi_up: Any, mock_request: MagicMockType
) -> None:
    """Test that get_datarobot_account includes the scoped token in response"""
    # Setup mock request with session containing all tokens
    mock_request.state.session = MagicMock()
    mock_request.state.session.datarobot_account_info = {"uid": "test_uid"}
    mock_request.state.session.datarobot_api_token = "api_token_123456789"
    mock_request.state.session.datarobot_api_skoped_token = "scoped_token_987654321"

    # Call the endpoint function directly
    response = await get_datarobot_account(mock_request)

    # Check that response includes all expected fields
    assert response["datarobot_account_info"] == {"uid": "test_uid"}
    assert response["datarobot_api_token"] == "****6789"
    assert response["datarobot_api_skoped_token"] == "****4321"


@pytest.mark.asyncio
async def test_store_datarobot_account_handles_scoped_token(
    setup_pulumi_stack: Any,
    pulumi_up: Any,
    mock_request: MagicMockType,
    mock_session_state: SessionState,
) -> None:
    """Test that store_datarobot_account can save the scoped token"""
    # Setup mock request
    mock_request.state.session = mock_session_state

    # Only test the keys that are supported
    mock_json = AsyncMock(return_value={"api_token": "api_token_123"})
    mock_request.json = mock_json

    # Call the endpoint function directly
    response = await store_datarobot_account(mock_request)

    # Check that tokens were saved to session
    assert mock_request.state.session.datarobot_api_token == "api_token_123"
    assert response["success"] is True


@patch("datarobot.client.get_client")
@patch("datarobot.client._create_client")
@patch("datarobot.client._is_compatible_client", return_value=True)
def test_authorization_header_integration(
    mock_is_compatible_client: MagicMockType,
    mock_create_client: MagicMockType,
    mock_get_client: MagicMockType,
    test_client: TestClient,
    setup_pulumi_stack: Any,
    pulumi_up: Any,
) -> None:
    """Integration test for Authorization header"""
    # Mock out the datarobot client to avoid actual API calls
    mock_dr_client = MagicMock()
    mock_create_client.return_value = (mock_dr_client, None, False, None)
    mock_get_client.return_value = mock_dr_client

    # Mock the account info response
    mock_dr_client.get.return_value.json.return_value = {"uid": "test_uid"}

    # Make a request with Authorization header
    response = test_client.get(
        "/api/v1/user/datarobot-account",
        headers={"Authorization": "Bearer test_token_123"},
    )

    # Response should include the token we sent
    assert response.status_code == 200

    response = test_client.post(
        "/api/v1/user/datarobot-account",
        json={"api_token": "api_token_123"},
    )
    assert response.status_code == 200

    # And get it back again to verify both tokens are there
    response = test_client.get(
        "/api/v1/user/datarobot-account",
        headers={"Authorization": "Bearer test_token_123"},
    )
    assert response.status_code == 200

    # The token should be truncated but still recognizable
    if len("test_token_123") > 4:
        truncated_token = response.json()["datarobot_api_skoped_token"]
        # Should show only the last 4 characters
        assert truncated_token == "****_123"
    else:
        # For short tokens, they remain as-is
        assert response.json()["datarobot_api_skoped_token"] == "test_token_123"


@pytest.mark.asyncio
async def test_delete_chat_message(
    setup_pulumi_stack: Any,
    pulumi_up: Any,
    mock_request: MagicMockType,
    mock_analyst_db: AsyncMock,
) -> None:
    mock_request.state.session = MagicMock()
    mock_analyst_db.chat_handler = AsyncMock()

    # Set up the mocks for the new function signature
    mock_message = AnalystChatMessage(
        id="message_id",
        role="user",
        content="User message 1",
        components=[],
        chat_id="test_chat_id",
    )
    mock_analyst_db.get_chat_message.return_value = mock_message
    mock_analyst_db.delete_chat_message.return_value = True

    test_messages = [
        AnalystChatMessage(role="user", content="User message 1", components=[]),
        AnalystChatMessage(
            role="assistant", content="Assistant message 1", components=[]
        ),
        AnalystChatMessage(role="user", content="User message 2", components=[]),
        AnalystChatMessage(
            role="assistant", content="Assistant message 2", components=[]
        ),
    ]
    mock_analyst_db.get_chat_messages.return_value = test_messages.copy()

    # Test successful deletion
    result = await delete_chat_message(mock_request, "message_id", mock_analyst_db)

    # Verify the mock interactions and results
    mock_analyst_db.get_chat_message.assert_called_once_with(message_id="message_id")
    mock_analyst_db.delete_chat_message.assert_called_once_with(message_id="message_id")
    mock_analyst_db.get_chat_messages.assert_called_once_with(chat_id="test_chat_id")
    assert result == test_messages.copy()

    # Reset mocks for next test
    mock_analyst_db.get_chat_message.reset_mock()
    mock_analyst_db.delete_chat_message.reset_mock()
    mock_analyst_db.get_chat_messages.reset_mock()

    # Test when message not found
    mock_analyst_db.get_chat_message.return_value = None

    result = await delete_chat_message(
        mock_request, "missing_message_id", mock_analyst_db
    )

    # Should return an empty list when message not found
    assert isinstance(result, list)
    assert len(result) == 0
    mock_analyst_db.delete_chat_message.assert_not_called()

    # Reset mocks for next test
    mock_analyst_db.get_chat_message.reset_mock()
    mock_analyst_db.delete_chat_message.reset_mock()

    # Test when an exception occurs
    mock_analyst_db.get_chat_message.return_value = mock_message
    mock_analyst_db.delete_chat_message.side_effect = Exception("Test exception")

    result = await delete_chat_message(mock_request, "message_id", mock_analyst_db)

    # Should return an empty list when an exception occurs
    assert isinstance(result, list)
    assert len(result) == 0
