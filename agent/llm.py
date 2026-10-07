"""LLM client — OpenAI-compatible chat completions, one class for several providers.

Providers (``PROVIDERS``): ``zhipu`` (open.bigmodel.cn, glm-*) and ``deepseek``
(api.deepseek.com, deepseek-*). Same protocol; they differ only in base URL,
key variable and how deterministic decoding is requested (decision D6).

Experiment requirements (docs/PROJECT_POSITIONING.md §7):
* greedy decoding (``do_sample=False``): the same prompt must give the same answer,
  otherwise differences between experiment arms cannot be attributed;
* every call records tokens and latency (cost per recovered case);
* hard call / token budgets so a loop or batch cannot run away;
* optional on-disk cache keyed by (model, params, prompt) so re-running an
  evaluation replays identical responses without new API calls.

Lessons carried over from gmv-rca-agent: retry 408/429/5xx with backoff on the
*same* prompt; reject API keys containing control characters (a Ctrl+V pasted
into a hidden prompt stores ``\\x16`` and the gateway answers HTML 400).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import requests

log = logging.getLogger(__name__)

ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "glm-4-flash"
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class LlmError(RuntimeError):
    pass


class LlmBudgetExceeded(LlmError):
    pass


@dataclass(frozen=True)
class LlmResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    finish_reason: str | None = None
    cached: bool = False

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class UsageMeter:
    max_calls: int | None = None
    max_tokens: int | None = None
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def check(self) -> None:
        if self.max_calls is not None and self.calls >= self.max_calls:
            raise LlmBudgetExceeded(f"call budget exhausted ({self.calls}/{self.max_calls})")
        if self.max_tokens is not None and self.input_tokens + self.output_tokens >= self.max_tokens:
            raise LlmBudgetExceeded(f"token budget exhausted ({self.input_tokens + self.output_tokens}/{self.max_tokens})")

    def record(self, r: LlmResponse) -> None:
        with self._lock:  # safe under a thread pool
            if r.cached:
                self.cached_calls += 1
                return
            self.calls += 1
            self.input_tokens += r.input_tokens
            self.output_tokens += r.output_tokens

    def snapshot(self) -> dict[str, int]:
        return {"calls": self.calls, "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "cached_calls": self.cached_calls}


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    key_env: str                    # .env variable holding the API key
    default_model: str
    greedy: dict[str, Any]          # request fields for deterministic decoding
    model_prefixes: tuple[str, ...]


PROVIDERS = {
    # do_sample=False is true greedy decoding. Kept identical to the original client so
    # existing cache entries (the fingerprint includes params) still hit.
    "zhipu": Provider("zhipu", ZHIPU_BASE_URL, "ZHIPUAI_API_KEY", "glm-4-flash", {"do_sample": False}, ("glm",)),
    # Pinned id, not the "deepseek-chat" alias (the platform re-points aliases; in 2026-09 it
    # served deepseek-flash = DeepSeek-V4.1-Flash). Thinking is ON by default and ignores
    # temperature, so it is disabled explicitly; temperature=0 is then the closest to greedy
    # (not guaranteed bit-exact: the on-disk cache is what makes arms share attempt 1).
    "deepseek": Provider("deepseek", DEEPSEEK_BASE_URL, "DEEPSEEK_API_KEY", "deepseek-flash",
                         {"temperature": 0.0, "thinking": {"type": "disabled"}}, ("deepseek",)),
}


class ChatClient(Protocol):
    model: str

    def complete(self, prompt: str, system: str | None = None) -> LlmResponse: ...


def provider_for_model(model: str) -> str:
    for p in PROVIDERS.values():
        if model.lower().startswith(p.model_prefixes):
            return p.name
    raise LlmError(f"cannot tell the provider of model {model!r}; set llm.provider / LLM_PROVIDER")


def check_api_key(key: str, env_name: str = "ZHIPUAI_API_KEY") -> str:
    bad = [f"U+{ord(c):04X}" for c in key if ord(c) < 32 or ord(c) == 127]
    if bad:
        raise LlmError(f"API key contains {len(bad)} invisible control character(s) ({', '.join(bad)}); "
                       "re-enter it without Ctrl+V into a hidden prompt")
    if not key.strip():
        raise LlmError(f"API key is empty (set {env_name} in .env)")
    return key.strip()


@dataclass
class OpenAICompatibleChatClient:
    api_key: str = field(repr=False)
    model: str = DEFAULT_MODEL
    provider: str = "zhipu"
    base_url: str | None = None           # None = the provider's
    max_output_tokens: int = 2048
    timeout_s: float = 120.0
    max_retries: int = 5
    meter: UsageMeter = field(default_factory=UsageMeter)

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS:
            raise LlmError(f"unknown provider {self.provider!r}; known: {', '.join(PROVIDERS)}")
        self.base_url = self.base_url or PROVIDERS[self.provider].base_url
        self.api_key = check_api_key(self.api_key, PROVIDERS[self.provider].key_env)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None, provider: str = "zhipu", model: str | None = None,
                 **kw: Any) -> "OpenAICompatibleChatClient":
        """Model: LLM_MODEL (or legacy ZHIPU_MODEL for zhipu) > ``model`` > the provider default."""
        env = dict(os.environ) if env is None else env
        p = PROVIDERS[provider]
        legacy = env.get("ZHIPU_MODEL") if provider == "zhipu" else None
        chosen = (env.get("LLM_MODEL") or legacy or model or p.default_model).strip()
        return cls(api_key=env.get(p.key_env, ""), model=chosen, provider=provider, **kw)

    @property
    def params(self) -> dict[str, Any]:
        return {**PROVIDERS[self.provider].greedy, "max_tokens": self.max_output_tokens}

    # LLM 客户端
    def complete(self, prompt: str, system: str | None = None) -> LlmResponse:
        self.meter.check()
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        payload = {"model": self.model, "messages": messages, **self.params}
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                # 向大模型发送请求
                resp = requests.post(f"{self.base_url}/chat/completions", json=payload, timeout=self.timeout_s,
                                     headers={"Authorization": f"Bearer {self.api_key}"})
            except requests.RequestException as e:
                if attempt == self.max_retries:
                    raise LlmError(f"transport error after {attempt + 1} attempts: {e}") from e
                _backoff(attempt)
                continue
            latency = int((time.perf_counter() - t0) * 1000)
            if resp.status_code in RETRYABLE_STATUS and attempt < self.max_retries:
                log.warning("%s HTTP %s, retry %d", self.provider, resp.status_code, attempt + 1)
                _backoff(attempt)
                continue
            if resp.status_code != 200:
                raise LlmError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
            choice = data["choices"][0]
            usage = data.get("usage") or {}
            r = LlmResponse(text=choice["message"].get("content") or "", model=data.get("model", self.model),
                            input_tokens=int(usage.get("prompt_tokens", 0)),
                            output_tokens=int(usage.get("completion_tokens", 0)),
                            latency_ms=latency, finish_reason=choice.get("finish_reason"))
            self.meter.record(r)
            return r
        raise LlmError("unreachable")


ZhipuChatClient = OpenAICompatibleChatClient  # backwards-compatible name


def make_client(llm_cfg: dict[str, Any] | None = None, model: str | None = None,
                env: dict[str, str] | None = None, **kw: Any) -> OpenAICompatibleChatClient:
    """Build the client from config or from a pinned model.

    * ``model`` given (replaying a recorded run): exactly that model; provider inferred from its name.
    * otherwise ``LLM_PROVIDER`` overrides ``llm.provider``; model as in ``from_env``.
    """
    env = dict(os.environ) if env is None else env
    if model:
        p = PROVIDERS[provider_for_model(model)]
        return OpenAICompatibleChatClient(api_key=env.get(p.key_env, ""), model=model, provider=p.name, **kw)
    cfg = llm_cfg or {}
    provider = (env.get("LLM_PROVIDER") or cfg.get("provider") or "zhipu").strip()
    if provider not in PROVIDERS:
        raise LlmError(f"unknown provider {provider!r}; known: {', '.join(PROVIDERS)}")
    # llm.model belongs to llm.provider: when LLM_PROVIDER switches provider, use that provider's default
    cfg_model = cfg.get("model") if cfg.get("provider", "zhipu") == provider else None
    return OpenAICompatibleChatClient.from_env(env, provider=provider, model=cfg_model, **kw)


def _backoff(attempt: int, base: float = 1.0, cap: float = 30.0) -> None:
    time.sleep(min(cap, base * 2 ** attempt))


def prompt_fingerprint(model: str, params: dict[str, Any], system: str | None, prompt: str) -> str:
    blob = json.dumps({"model": model, "params": params, "system": system, "prompt": prompt},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class CachingChatClient:
    """Replays responses for identical (model, params, system, prompt)."""

    inner: OpenAICompatibleChatClient
    path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cache: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._cache[rec["key"]] = rec["response"]

    @property
    def model(self) -> str:
        return self.inner.model

    @property
    def meter(self) -> UsageMeter:
        return self.inner.meter

    def complete(self, prompt: str, system: str | None = None) -> LlmResponse:
        key = prompt_fingerprint(self.inner.model, self.inner.params, system, prompt)
        if key in self._cache:
            r = LlmResponse(**{**self._cache[key], "cached": True})
            self.inner.meter.record(r)
            return r
        r = self.inner.complete(prompt, system)
        with self._lock:  # safe under a thread pool
            self._cache[key] = asdict(r)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "response": asdict(r)}, ensure_ascii=False) + "\n")
        return r
