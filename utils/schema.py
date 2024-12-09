from __future__ import annotations

from pydantic import BaseModel


class ChatAgentDeploymentSettings(BaseModel):
    target_feature_name: str = "content"
    prompt_feature_name: str = "promptText"
    request_timeout: int = 60
    max_retries: int = 1
    temperature: int = 0


class AppInfraSettings(BaseModel):
    registered_model_name: str
    registered_model_version_id: str
