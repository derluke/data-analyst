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

from __future__ import annotations

from typing import Any, Optional

import datarobot as dr  # Assuming you have the DataRobot Python SDK installed
import pulumi
from datarobotx.idp.credentials import get_replace_or_create_credential
from pulumi import dynamic


class AWSCredentialProvider(dynamic.ResourceProvider):
    def create(self, props: dict[str, Any]) -> dynamic.CreateResult:
        dr_client = dr.client.get_client()
        credential_id = get_replace_or_create_credential(
            endpoint=dr_client.endpoint,
            token=dr_client.token,
            name=str(props.get("resource_name")),
            credential_type="s3",
            aws_access_key_id=props.get("access_key_id"),
            aws_secret_access_key=props.get("secret_access_key"),
            aws_session_token=props.get("session_token"),
        )
        credential = dr.Credential.get(credential_id)  # type: ignore[attr-defined]

        # Include resource_name in the outputs to match future state
        return dynamic.CreateResult(
            id_=str(credential.credential_id),
            outs={
                "credential_id": credential.credential_id,
                "resource_name": props.get("resource_name"),  # Add this
                "access_key_id": props.get("access_key_id"),
                "secret_access_key": props.get("secret_access_key"),
                "session_token": props.get("session_token"),
            },
        )

    def delete(self, id: str, props: dict[str, Any]) -> None:
        # Delete the credential when the resource is destroyed
        dr.Credential.get(id).delete()  # type: ignore[attr-defined]

    def diff(
        self, id: str, olds: dict[str, Any], news: dict[str, Any]
    ) -> dynamic.DiffResult:
        relevant_keys = {"access_key_id", "secret_access_key", "session_token"}
        changes = {k: olds.get(k) != news.get(k) for k in news if k in relevant_keys}

        return dynamic.DiffResult(
            changes=any(changes.values()),
            replaces=[
                "access_key_id",
                "secret_access_key",
                "session_token",
            ],
        )


class AWSCredential(dynamic.Resource):
    credential_id: pulumi.Output[str]
    access_key_id: pulumi.Output[str]
    secret_access_key: pulumi.Output[str]
    session_token: pulumi.Output[Optional[str]]

    def __init__(
        self,
        resource_name: str,
        access_key_id: str,
        secret_access_key: str,
        session_token: str | None = None,
        opts: Any = None,
        **kwargs: Any,
    ):
        props = {
            "resource_name": resource_name,
            "access_key_id": pulumi.Output.secret(access_key_id),
            "secret_access_key": pulumi.Output.secret(secret_access_key),
            "session_token": (
                pulumi.Output.secret(session_token) if session_token else None
            ),
            "credential_id": None,
        }
        super().__init__(
            AWSCredentialProvider(),
            resource_name,
            props,
            opts,
        )
