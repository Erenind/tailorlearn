from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppPaths:
    root: Path

    @classmethod
    def default(cls) -> "AppPaths":
        current = Path.cwd().resolve()
        if (current / "pyproject.toml").exists():
            return cls(current)
        return cls(Path(__file__).resolve().parents[2])

    @property
    def db_dir(self) -> Path:
        return self.root / "db"

    @property
    def database_file(self) -> Path:
        return self.db_dir / "tailorlearn.sqlite3"

    @property
    def materials_dir(self) -> Path:
        return self.db_dir / "materials"

    @property
    def exports_dir(self) -> Path:
        return self.db_dir / "exports"

    @property
    def local_dir(self) -> Path:
        return self.root / ".local"

    @property
    def provider_file(self) -> Path:
        return self.local_dir / "ai_profiles.json"

    def ensure(self) -> None:
        for path in (
            self.db_dir,
            self.materials_dir,
            self.exports_dir,
            self.local_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        try:
            self.local_dir.chmod(0o700)
        except OSError:
            pass
