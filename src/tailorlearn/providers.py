from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path

from tailorlearn.models import ProviderConfig, ProviderProfile


class ProviderStore:
    """Local-only provider profiles; never store this file under db/."""

    def __init__(self, path: Path):
        self.path = path

    def load(self) -> ProviderConfig:
        if not self.path.exists():
            return ProviderConfig()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return ProviderConfig.model_validate(data)
        except (OSError, json.JSONDecodeError, ValueError):
            return ProviderConfig()

    def save(self, config: ProviderConfig) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.path.parent.chmod(0o700)
        except OSError:
            pass
        fd, temp_name = tempfile.mkstemp(
            prefix=".ai_profiles-",
            suffix=".json",
            dir=self.path.parent,
            text=True,
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(
                    config.model_dump(),
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            temp_path.chmod(0o600)
            os.replace(temp_path, self.path)
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    def upsert(
        self,
        *,
        profile_id: str | None,
        vendor_name: str,
        base_url: str,
        model: str,
        api_key: str | None,
    ) -> ProviderProfile:
        config = self.load()
        vendor_name = vendor_name.strip()
        base_url = base_url.strip().rstrip("/")
        model = model.strip()
        if not vendor_name or not base_url or not model:
            raise ValueError("厂商名称、base_url 和模型均不能为空")

        current = next(
            (profile for profile in config.profiles if profile.id == profile_id),
            None,
        )
        if profile_id and current is None:
            raise ValueError("找不到要更新的厂商配置")
        for profile in config.profiles:
            if (
                profile.id != profile_id
                and profile.vendor_name.casefold() == vendor_name.casefold()
            ):
                raise ValueError("厂商名称已存在，请直接编辑该配置")

        resolved_key = current.api_key if api_key is None and current else (api_key or "")
        profile = ProviderProfile(
            id=current.id if current else f"provider_{uuid.uuid4().hex[:12]}",
            vendor_name=vendor_name,
            base_url=base_url,
            model=model,
            api_key=resolved_key,
        )
        profiles = [item for item in config.profiles if item.id != profile.id]
        profiles.append(profile)
        profiles.sort(key=lambda item: item.vendor_name.casefold())
        config = ProviderConfig(
            active_profile_id=config.active_profile_id or profile.id,
            profiles=profiles,
        )
        self.save(config)
        return profile

    def set_active(self, profile_id: str) -> None:
        config = self.load()
        if not any(profile.id == profile_id for profile in config.profiles):
            raise ValueError("找不到该厂商配置")
        config.active_profile_id = profile_id
        self.save(config)

    def active(self) -> ProviderProfile | None:
        config = self.load()
        if config.active_profile_id:
            for profile in config.profiles:
                if profile.id == config.active_profile_id:
                    return profile
        return config.profiles[0] if config.profiles else None

    def clear_key(self, profile_id: str) -> None:
        config = self.load()
        found = False
        for profile in config.profiles:
            if profile.id == profile_id:
                profile.api_key = ""
                found = True
        if not found:
            raise ValueError("找不到该厂商配置")
        self.save(config)

