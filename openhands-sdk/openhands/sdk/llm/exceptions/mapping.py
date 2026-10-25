from __future__ import annotations

from litellm.exceptions import (
    APIConnectionError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout as LiteLLMTimeout,
)

from .classifier import (
    is_content_policy_violation,
    is_context_window_exceeded,
    looks_like_auth_error,
    looks_like_malformed_conversation_history_error,
)
from .types import (
    LLMAuthenticationError,
    LLMBadRequestError,
    LLMContentPolicyViolationError,
    LLMContextWindowExceedError,
    LLMHardTimeoutError,
    LLMMalformedConversationHistoryError,
    LLMRateLimitError,
    LLMServiceUnavailableError,
    LLMTimeoutError,
)


class SDKHardTimeout(LiteLLMTimeout):
    """A timeout raised by the SDK's own per-attempt deadline."""

    def __init__(
        self, message: str, model: str, llm_provider: str, max_attempts: int
    ) -> None:
        super().__init__(message=message, model=model, llm_provider=llm_provider)
        self.retry_max_attempts = max_attempts


def map_provider_exception(exception: Exception) -> Exception:
    """
    Map provider/LiteLLM exceptions to SDK-typed exceptions.

    Returns original exception if no mapping applies.
    """
    # Context window exceeded first (highest priority among normal retries)
    if is_context_window_exceeded(exception):
        return LLMContextWindowExceedError(str(exception))

    # Malformed prompt history is distinct from context-window exhaustion even
    # though the recovery path still uses condensation.
    if looks_like_malformed_conversation_history_error(exception):
        return LLMMalformedConversationHistoryError(str(exception))

    # Auth-like errors often appear as BadRequest/OpenAIError with specific text
    if looks_like_auth_error(exception):
        return LLMAuthenticationError(str(exception))

    if isinstance(exception, RateLimitError):
        return LLMRateLimitError(str(exception))

    if isinstance(exception, SDKHardTimeout):
        return LLMHardTimeoutError(
            f"{exception}. Try a shorter or chunked response, or increase this "
            "model profile's timeout or hard timeout attempt limit in Advanced "
            "settings."
        )

    if isinstance(exception, LiteLLMTimeout):
        return LLMTimeoutError(str(exception))

    # Connectivity and service-side availability issues → service unavailable
    if isinstance(
        exception, (APIConnectionError, ServiceUnavailableError, InternalServerError)
    ):
        return LLMServiceUnavailableError(str(exception))

    # Content-policy blocks are deterministic 4xx; distinguish them from generic
    # bad requests so the agent can recover softly instead of hard-erroring.
    if is_content_policy_violation(exception):
        return LLMContentPolicyViolationError(str(exception))

    # Generic client-side 4xx errors
    if isinstance(exception, BadRequestError):
        return LLMBadRequestError(str(exception))

    # Unknown: let caller re-raise original
    return exception
