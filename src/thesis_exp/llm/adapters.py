"""Provider adapters for LLM inference."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from thesis_exp.llm.base import BaseLLMAdapter, DiagnosisRequest, ModelConfig, RawLLMResponse


class OpenAICompatibleChatAdapter(BaseLLMAdapter):
    """Adapter for OpenAI-compatible chat completion APIs."""

    provider_name = "openai_compatible"

    def supports_json_output(self) -> bool:
        return True

    def generate(
        self,
        request: DiagnosisRequest,
        model_config: ModelConfig,
    ) -> RawLLMResponse:
        api_key = model_config.resolve_api_key()
        if not api_key:
            raise RuntimeError("Missing API key for OpenAI-compatible adapter.")
        if not model_config.api_base_url:
            raise RuntimeError("Missing api_base_url for OpenAI-compatible adapter.")

        request_payload = self.build_request_payload(request, model_config)
        response_payload = self._post_json(
            url=model_config.api_base_url.rstrip("/") + "/chat/completions",
            payload=request_payload,
            api_key=api_key,
            timeout_seconds=model_config.timeout_seconds,
        )
        response_text = self.extract_response_text(response_payload)

        return RawLLMResponse(
            provider_name=model_config.provider_name,
            model_name=model_config.model_name,
            sample_id=request.sample_id,
            prompt_template_name=request.prompt_template_name,
            response_text=response_text,
            request_payload=request_payload,
            response_payload=response_payload,
            response_metadata={
                "json_output_requested": request.use_json_output,
                "response_schema_name": request.response_schema_name,
            },
        )

    def build_request_payload(
        self,
        request: DiagnosisRequest,
        model_config: ModelConfig,
    ) -> dict[str, Any]:
        """Build one OpenAI-compatible request payload."""

        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.user_prompt})

        payload: dict[str, Any] = {
            "model": model_config.model_name,
            "messages": messages,
            self._token_limit_parameter(model_config): model_config.max_output_tokens,
        }
        if self._should_send_temperature(model_config):
            payload["temperature"] = model_config.temperature
        if request.use_json_output:
            payload["response_format"] = {"type": "json_object"}
        payload.update(model_config.extra_settings)
        return payload

    def _token_limit_parameter(self, model_config: ModelConfig) -> str:
        """Select the provider token-limit field for the target model."""

        configured_name = str(model_config.extra_settings.get("token_limit_parameter", "")).strip()
        if configured_name:
            return configured_name
        if model_config.model_name.startswith("gpt-5"):
            return "max_completion_tokens"
        return "max_tokens"

    def _should_send_temperature(self, model_config: ModelConfig) -> bool:
        """Return whether an explicit temperature field should be sent."""

        if "temperature" in model_config.extra_settings:
            return False
        return not model_config.model_name.startswith("gpt-5")

    def extract_response_text(self, response_payload: dict[str, Any]) -> str:
        """Extract model text from a provider response payload."""

        choices = response_payload.get("choices", [])
        if not choices:
            raise RuntimeError("Response payload does not contain choices.")

        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            return "".join(text_parts)
        return str(content)

    def _post_json(
        self,
        *,
        url: str,
        payload: dict[str, Any],
        api_key: str,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=url,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP request failed with status {exc.code}: {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Network request failed: {exc.reason}") from exc


class LocalMockAdapter(BaseLLMAdapter):
    """Local adapter for deterministic testing and dry runs."""

    provider_name = "mock"

    def supports_json_output(self) -> bool:
        return True

    def generate(
        self,
        request: DiagnosisRequest,
        model_config: ModelConfig,
    ) -> RawLLMResponse:
        response_text = self._build_mock_response_text(request, model_config)
        return RawLLMResponse(
            provider_name=model_config.provider_name,
            model_name=model_config.model_name,
            sample_id=request.sample_id,
            prompt_template_name=request.prompt_template_name,
            response_text=response_text,
            request_payload={
                "system_prompt": request.system_prompt,
                "user_prompt": request.user_prompt,
                "use_json_output": request.use_json_output,
            },
            response_payload={"mock": True, "response_text": response_text},
            response_metadata={"response_schema_name": request.response_schema_name},
        )

    def _build_mock_response_text(
        self,
        request: DiagnosisRequest,
        model_config: ModelConfig,
    ) -> str:
        mock_response_text = str(model_config.extra_settings.get("mock_response_text", "")).strip()
        if mock_response_text:
            return mock_response_text

        if request.use_json_output:
            payload = {
                "bug_type": "mock_bug_type",
                "bug_location": "line 1",
                "explanation": f"Mock diagnosis for sample {request.sample_id}.",
                "repaired_code": "",
            }
            return json.dumps(payload, ensure_ascii=False)

        return f"Mock diagnosis response for sample {request.sample_id}."
