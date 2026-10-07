import pytest

from agent.llm import LlmError, make_client, prompt_fingerprint, provider_for_model

ENV = {"ZHIPUAI_API_KEY": "z" * 49, "DEEPSEEK_API_KEY": "d" * 35}
CFG = {"provider": "zhipu", "model": "glm-4-flash"}


def test_zhipu_params_unchanged_so_old_cache_still_hits():
    c = make_client(CFG, env=ENV, max_output_tokens=2048)
    assert (c.provider, c.model) == ("zhipu", "glm-4-flash")
    assert c.params == {"do_sample": False, "max_tokens": 2048}
    assert c.base_url == "https://open.bigmodel.cn/api/paas/v4"


def test_env_switches_provider_without_leaking_the_other_providers_model():
    c = make_client(CFG, env={**ENV, "LLM_PROVIDER": "deepseek", "ZHIPU_MODEL": "glm-4-flash"})
    assert (c.provider, c.model, c.base_url) == ("deepseek", "deepseek-flash", "https://api.deepseek.com")
    assert c.params["temperature"] == 0.0 and "do_sample" not in c.params
    assert c.params["thinking"] == {"type": "disabled"}  # otherwise temperature is ignored


def test_config_provider_and_model_override():
    c = make_client({"provider": "deepseek", "model": "deepseek-chat"}, env={**ENV, "LLM_MODEL": "deepseek-x"})
    assert c.model == "deepseek-x"


def test_replay_pins_recorded_model_and_infers_provider():
    assert make_client(model="deepseek-chat", env=ENV).provider == "deepseek"
    assert make_client(model="glm-4-flash", env=ENV).provider == "zhipu"
    assert provider_for_model("GLM-4-Plus") == "zhipu"
    with pytest.raises(LlmError):
        provider_for_model("some-other-model")


def test_missing_key_names_the_right_variable():
    with pytest.raises(LlmError, match="DEEPSEEK_API_KEY"):
        make_client({"provider": "deepseek"}, env={"ZHIPUAI_API_KEY": "z" * 49})


def test_cache_keys_differ_between_providers():
    z = make_client(CFG, env=ENV)
    d = make_client(model="deepseek-chat", env=ENV)
    assert prompt_fingerprint(z.model, z.params, "s", "p") != prompt_fingerprint(d.model, d.params, "s", "p")
