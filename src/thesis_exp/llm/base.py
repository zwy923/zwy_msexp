"""Core abstractions for model-agnostic LLM inference."""

from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _sample_to_dict(sample: Any) -> dict[str, Any]:
    if hasattr(sample, "to_dict"):
        return sample.to_dict()
    if isinstance(sample, dict):
        return sample
    raise TypeError("Sample must be a dict-like object or expose to_dict().")


def _sample_identifier(sample: Any) -> str:
    for attribute_name in ("transformed_sample_id", "sample_id", "problem_id"):
        identifier = getattr(sample, attribute_name, None)
        if identifier:
            return str(identifier)
    if isinstance(sample, dict):
        for key in ("transformed_sample_id", "sample_id", "problem_id"):
            identifier = sample.get(key)
            if identifier:
                return str(identifier)
    raise ValueError("Could not resolve a stable sample identifier.")


@dataclass(slots=True)
class PromptTemplate:
    """Structured prompt template for diagnosis requests."""

    template_name: str
    system_prompt: str = ""
    user_prompt_template: str = "{sample_json}"
    response_schema_name: str = ""
    is_fully_rendered: bool = False

    def render(self, sample: Any) -> dict[str, str]:
        """Render system and user messages for one sample."""

        if self.is_fully_rendered:
            return {
                "system_prompt": self.system_prompt,
                "user_prompt": self.user_prompt_template,
            }

        sample_payload = _sample_to_dict(sample)
        sample_json = json.dumps(sample_payload, ensure_ascii=False, indent=2, sort_keys=True)
        user_prompt = self.user_prompt_template.format(
            sample=sample_payload,
            sample_json=sample_json,
            sample_id=_sample_identifier(sample),
        )
        return {
            "system_prompt": self.system_prompt,
            "user_prompt": user_prompt,
        }


@dataclass(slots=True)
class ModelConfig:
    """Provider-neutral model configuration."""

    provider_name: str
    model_name: str
    api_base_url: str = ""
    api_key_env_var: str = ""
    api_key: str = ""
    temperature: float = 0.0
    max_output_tokens: int = 2048
    timeout_seconds: float = 60.0
    max_retries: int = 3
    backoff_initial_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    enable_json_output: bool = False
    log_raw_response: bool = True
    artifact_root_dir: str = ""
    extra_settings: dict[str, Any] = field(default_factory=dict)

    def resolve_api_key(self) -> str:
        """Resolve the API key from config or environment."""

        if self.api_key:
            return self.api_key
        if self.api_key_env_var:
            return os.environ.get(self.api_key_env_var, "")
        return ""

    @classmethod
    def from_env(
        cls,
        provider_name: str,
        model_name: str,
        *,
        api_base_url: str = "",
        api_key_env_var: str = "",
        **kwargs: Any,
    ) -> "ModelConfig":
        """Create a config using environment-backed credentials."""

        api_key = os.environ.get(api_key_env_var, "") if api_key_env_var else ""
        return cls(
            provider_name=provider_name,
            model_name=model_name,
            api_base_url=api_base_url,
            api_key_env_var=api_key_env_var,
            api_key=api_key,
            **kwargs,
        )


@dataclass(slots=True)
class DiagnosisRequest:
    """Prepared diagnosis request passed to an adapter."""

    sample_id: str
    prompt_template_name: str
    response_schema_name: str
    system_prompt: str
    user_prompt: str
    use_json_output: bool
    request_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RawLLMResponse:
    """Raw response returned by an adapter or resumed from artifacts."""

    provider_name: str
    model_name: str
    sample_id: str
    prompt_template_name: str
    response_text: str
    request_payload: dict[str, Any] = field(default_factory=dict)
    response_payload: dict[str, Any] = field(default_factory=dict)
    response_metadata: dict[str, Any] = field(default_factory=dict)
    attempt_count: int = 1
    artifact_directory: str = ""
    resumed_from_artifact: bool = False


class BaseLLMAdapter(ABC):
    """Adapter interface for one provider family."""

    provider_name: str

    @abstractmethod
    def supports_json_output(self) -> bool:
        """Return whether native JSON mode is supported."""

    @abstractmethod
    def generate(
        self,
        request: DiagnosisRequest,
        model_config: ModelConfig,
    ) -> RawLLMResponse:
        """Execute one provider-specific request."""


class BaseLLMClient(ABC):
    """Backward-compatible alias for legacy client naming."""

    @abstractmethod
    def generate(
        self,
        request: DiagnosisRequest,
        model_config: ModelConfig,
    ) -> RawLLMResponse:
        """Execute one model request."""


class DiagnosisArtifactStore:
    """Persist request and response artifacts for reproducible runs."""

    def __init__(self, artifact_root_dir: str) -> None:
        self.artifact_root_dir = Path(artifact_root_dir)

    def _safe_path_component(self, value: str) -> str:
        safe_value = re.sub(r'[<>:"/\\|?*]+', "_", value)
        return safe_value.strip() or "default"

    def _artifact_dir(self, run_id: str, sample_id: str, model_config: ModelConfig) -> Path:
        return (
            self.artifact_root_dir
            / self._safe_path_component(run_id)
            / self._safe_path_component(model_config.provider_name)
            / self._safe_path_component(model_config.model_name)
            / self._safe_path_component(sample_id)
        )

    def load_response(
        self,
        run_id: str,
        sample_id: str,
        model_config: ModelConfig,
    ) -> RawLLMResponse | None:
        """Load a saved response if it exists."""

        artifact_dir = self._artifact_dir(run_id, sample_id, model_config)
        response_path = artifact_dir / "response.json"
        if not response_path.exists():
            return None

        payload = json.loads(response_path.read_text(encoding="utf-8"))
        raw_response = RawLLMResponse(
            provider_name=payload["provider_name"],
            model_name=payload["model_name"],
            sample_id=payload["sample_id"],
            prompt_template_name=payload["prompt_template_name"],
            response_text=payload["response_text"],
            request_payload=payload.get("request_payload", {}),
            response_payload=payload.get("response_payload", {}),
            response_metadata=payload.get("response_metadata", {}),
            attempt_count=payload.get("attempt_count", 1),
            artifact_directory=str(artifact_dir),
            resumed_from_artifact=True,
        )
        return raw_response

    def save_request(
        self,
        run_id: str,
        request: DiagnosisRequest,
        model_config: ModelConfig,
    ) -> str:
        """Save one rendered request artifact."""

        artifact_dir = self._artifact_dir(run_id, request.sample_id, model_config)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        request_payload = {
            "provider_name": model_config.provider_name,
            "model_name": model_config.model_name,
            "sample_id": request.sample_id,
            "prompt_template_name": request.prompt_template_name,
            "response_schema_name": request.response_schema_name,
            "system_prompt": request.system_prompt,
            "user_prompt": request.user_prompt,
            "use_json_output": request.use_json_output,
            "request_metadata": request.request_metadata,
        }
        (artifact_dir / "request.json").write_text(
            json.dumps(request_payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return str(artifact_dir)

    def save_response(
        self,
        run_id: str,
        response: RawLLMResponse,
        model_config: ModelConfig,
    ) -> str:
        """Save one raw response artifact."""

        artifact_dir = self._artifact_dir(run_id, response.sample_id, model_config)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        response.artifact_directory = str(artifact_dir)

        response_payload = {
            "provider_name": response.provider_name,
            "model_name": response.model_name,
            "sample_id": response.sample_id,
            "prompt_template_name": response.prompt_template_name,
            "response_text": response.response_text,
            "request_payload": response.request_payload,
            "response_payload": response.response_payload,
            "response_metadata": response.response_metadata,
            "attempt_count": response.attempt_count,
            "artifact_directory": response.artifact_directory,
        }
        (artifact_dir / "response.json").write_text(
            json.dumps(response_payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if model_config.log_raw_response:
            (artifact_dir / "raw_response.txt").write_text(response.response_text, encoding="utf-8")
        return str(artifact_dir)


class DiagnosisInferenceEngine:
    """Model-agnostic orchestration for LLM diagnosis calls."""

    def __init__(
        self,
        adapters: dict[str, BaseLLMAdapter],
        artifact_store: DiagnosisArtifactStore | None = None,
        sleep_fn: Any = time.sleep,
    ) -> None:
        self.adapters = adapters
        self.artifact_store = artifact_store
        self.sleep_fn = sleep_fn

    def diagnose(
        self,
        sample: Any,
        prompt_template: PromptTemplate | str,
        model_config: ModelConfig,
        *,
        run_id: str = "default_run",
    ) -> RawLLMResponse:
        """Diagnose one sample and return the raw response."""

        adapter = self._resolve_adapter(model_config.provider_name)
        prompt = self._normalize_prompt_template(prompt_template)
        sample_id = _sample_identifier(sample)

        rendered_prompt = prompt.render(sample)
        request = DiagnosisRequest(
            sample_id=sample_id,
            prompt_template_name=prompt.template_name,
            response_schema_name=prompt.response_schema_name,
            system_prompt=rendered_prompt["system_prompt"],
            user_prompt=rendered_prompt["user_prompt"],
            use_json_output=model_config.enable_json_output and adapter.supports_json_output(),
            request_metadata={"provider_name": model_config.provider_name},
        )

        if self.artifact_store is not None:
            self.artifact_store.save_request(run_id, request, model_config)

        response = self._generate_with_retry(adapter, request, model_config)

        if self.artifact_store is not None:
            self.artifact_store.save_response(run_id, response, model_config)

        return response

    def diagnose_batch(
        self,
        samples: list[Any],
        prompt_template: PromptTemplate | str,
        model_config: ModelConfig,
        *,
        run_id: str = "default_run",
    ) -> list[RawLLMResponse]:
        """Diagnose a batch of samples sequentially."""

        return [
            self.diagnose(sample, prompt_template, model_config, run_id=run_id)
            for sample in samples
        ]

    def _generate_with_retry(
        self,
        adapter: BaseLLMAdapter,
        request: DiagnosisRequest,
        model_config: ModelConfig,
    ) -> RawLLMResponse:
        last_error: Exception | None = None

        for attempt in range(1, model_config.max_retries + 1):
            try:
                response = adapter.generate(request, model_config)
                response.attempt_count = attempt
                return response
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt == model_config.max_retries:
                    break
                delay_seconds = model_config.backoff_initial_seconds * (
                    model_config.backoff_multiplier ** (attempt - 1)
                )
                self.sleep_fn(delay_seconds)

        raise RuntimeError(
            f"Diagnosis request failed after {model_config.max_retries} attempts."
        ) from last_error

    def _resolve_adapter(self, provider_name: str) -> BaseLLMAdapter:
        adapter = self.adapters.get(provider_name)
        if adapter is None:
            raise KeyError(f"No adapter registered for provider: {provider_name}")
        return adapter

    def _normalize_prompt_template(self, prompt_template: PromptTemplate | str) -> PromptTemplate:
        if isinstance(prompt_template, PromptTemplate):
            return prompt_template
        return PromptTemplate(template_name="inline_template", user_prompt_template=prompt_template)
