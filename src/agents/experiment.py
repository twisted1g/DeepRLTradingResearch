from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from uuid import uuid4

from stable_baselines3.common.logger import configure


class Experiment:
    def __init__(
        self,
        name: str,
        base_dir: str | Path | None = None,
        run_id: str | None = None,
        outputs: Iterable[str] = ("stdout", "csv", "tensorboard"),
    ) -> None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        suffix = uuid4().hex[:6]
        self.run_id = run_id or f"{timestamp}-{suffix}"
        if base_dir is None:
            project_root = Path(__file__).resolve().parents[2]
            base_dir = project_root / "experiments"
        self.dir = Path(base_dir) / name / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "checkpoints").mkdir(exist_ok=True)
        self.outputs = tuple(outputs)

    def save_config(self, config: object) -> Path:
        data = asdict(config) if is_dataclass(config) else getattr(config, "__dict__", {})
        path = self.dir / "config.json"
        path.write_text(
            __import__("json").dumps(data, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path

    def setup_logger(self):
        return configure(str(self.dir), list(self.outputs))

    def save_model(self, model, name: str = "model"):
        path = self.dir / name
        model.save(str(path))
        return path
