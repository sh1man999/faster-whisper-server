from functools import lru_cache
import logging
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from faster_whisper_server.config import Config
from faster_whisper_server.model_manager import PiperModelManager, WhisperModelManager

logger = logging.getLogger(__name__)

# NOTE: `get_config` is called directly instead of using sub-dependencies so that these functions could be used outside of `FastAPI`  # noqa: E501


# https://fastapi.tiangolo.com/advanced/settings/?h=setti#creating-the-settings-only-once-with-lru_cache
# WARN: Any new module that ends up calling this function directly (not through `FastAPI` dependency injection) should be patched in `tests/conftest.py`  # noqa: E501
@lru_cache
def get_config() -> Config:
    return Config()


ConfigDependency = Annotated[Config, Depends(get_config)]


@lru_cache
def get_model_manager() -> WhisperModelManager:
    config = get_config()
    return WhisperModelManager(config.whisper)


ModelManagerDependency = Annotated[WhisperModelManager, Depends(get_model_manager)]


@lru_cache
def get_piper_model_manager() -> PiperModelManager:
    config = get_config()
    return PiperModelManager(config.whisper.ttl)  # HACK: should have its own config


PiperModelManagerDependency = Annotated[PiperModelManager, Depends(get_piper_model_manager)]


security = HTTPBearer()


async def verify_api_key(
    config: ConfigDependency, credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)]
) -> None:
    if credentials.credentials != config.api_key:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)


ApiKeyDependency = Depends(verify_api_key)

