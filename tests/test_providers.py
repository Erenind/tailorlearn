from __future__ import annotations

import os
import stat

from tailorlearn.providers import PROVIDER_PRESETS, ProviderStore


def test_builtin_provider_presets_match_common_ai_accesses():
    assert [
        (
            preset.vendor_name,
            preset.base_url,
            preset.model,
        )
        for preset in PROVIDER_PRESETS
    ] == [
        (
            "DeepSeek",
            "https://api.deepseek.com",
            "deepseek-flash",
        ),
        (
            "千问",
            "https://cn-beijing.maas.qianwenaiapi.com/compatible-mode/v1",
            "qwen3.8-flash",
        ),
        (
            "智谱",
            "https://open.bigmodel.cn/api/paas/v4",
            "glm-4.7-flash",
        ),
        (
            "豆包",
            "https://ark.cn-beijing.volces.com/api/v3",
            "",
        ),
        (
            "Kimi",
            "https://api.moonshot.cn/v1",
            "",
        ),
    ]


def test_provider_store_keeps_multiple_vendors_and_protects_file(tmp_path):
    path = tmp_path / ".local" / "ai_profiles.json"
    store = ProviderStore(path)
    first = store.upsert(
        profile_id=None,
        vendor_name="OpenAI",
        base_url="https://api.openai.com/v1/",
        model="model-a",
        api_key="sk-1234567890abcd",
    )
    second = store.upsert(
        profile_id=None,
        vendor_name="Other",
        base_url="https://example.test/v1",
        model="model-b",
        api_key="secret-b",
    )
    store.set_active(second.id)

    config = store.load()
    assert {profile.vendor_name for profile in config.profiles} == {
        "OpenAI",
        "Other",
    }
    assert config.active_profile_id == second.id
    assert first.base_url == "https://api.openai.com/v1"
    # Windows does not expose POSIX permission bits through stat().
    # The atomic write and secret-masking behaviour are platform-neutral;
    # retain the stricter mode assertion where the OS supports it.
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    else:
        assert path.is_file()
    assert first.masked_api_key == "sk-••••••••abcd"
    assert first.public_dict()["api_key"] == first.masked_api_key
    assert first.api_key not in repr(first.public_dict())


def test_provider_edit_without_key_preserves_existing_key(tmp_path):
    store = ProviderStore(tmp_path / "ai_profiles.json")
    profile = store.upsert(
        profile_id=None,
        vendor_name="Vendor",
        base_url="https://example.test/v1",
        model="one",
        api_key="keep-me",
    )
    store.upsert(
        profile_id=profile.id,
        vendor_name="Vendor",
        base_url="https://example.test/v1",
        model="two",
        api_key=None,
    )
    loaded = store.load().profiles[0]
    assert loaded.model == "two"
    assert loaded.api_key == "keep-me"
