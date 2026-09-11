import importlib.metadata
import json
import logging
import platform
import re
import subprocess
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf
LOGGER_NAME = "kaggle_titanic"
_LOG_CONTEXT: ContextVar[str] = ContextVar("log_context", default="")


class _LogContextFilter(logging.Filter):
    """Add the active execution context to every project log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.context = _LOG_CONTEXT.get()
        return True


def get_logger() -> logging.Logger:
    """Return the project's shared logger."""

    return logging.getLogger(LOGGER_NAME)


@contextmanager
def log_context(context: str):
    """Temporarily label related log messages, for example a CV fold."""

    token = _LOG_CONTEXT.set(f"[{context}] ")
    try:
        yield
    finally:
        _LOG_CONTEXT.reset(token)


def configure_logging(run_dir: Path) -> logging.Logger:
    """Configure console and per-run file logging."""

    logger = get_logger()
    warning_logger = logging.getLogger("py.warnings")
    logging.captureWarnings(True)

    logger.setLevel(logging.INFO)
    logger.propagate = False
    warning_logger.setLevel(logging.WARNING)
    warning_logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(context)s%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for configured_logger in (logger, warning_logger):
        for handler in configured_logger.handlers[:]:
            handler.close()
            configured_logger.removeHandler(handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.addFilter(_LogContextFilter())
        configured_logger.addHandler(console_handler)

        file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.addFilter(_LogContextFilter())
        configured_logger.addHandler(file_handler)

    return logger


class ExperimentTracker:
    """Persist configuration, metadata, features, logs, and metrics per run."""

    def __init__(self, experiments_dir: str | Path, experiment_name: str) -> None:
        self.experiments_dir = Path(experiments_dir)
        self.experiment_name = experiment_name
        self.project_root = Path(__file__).resolve().parent.parent
        self.run_id = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
        self.run_dir = self.experiments_dir / experiment_name / self.run_id
        self.metadata: dict = {}
        self.logger = get_logger()

    def start(self, config) -> Path:
        """Create a run directory and save the resolved experiment configuration."""

        self.run_dir.mkdir(parents=True, exist_ok=False)

        resolved_config = OmegaConf.create(
            OmegaConf.to_container(config, resolve=True),
        )
        OmegaConf.save(resolved_config, self.run_dir / "config.yaml")

        self.metadata = {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "status": "running",
            "started_at": self._timestamp(),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "package_versions": self._package_versions(
                self.project_root / "requirements.txt",
            ),
            "git": self._git_metadata(),
            "datasets": {},
        }
        self._save_metadata()
        self.logger = configure_logging(self.run_dir)
        self.log(f"Run started: {self.run_id}")

        return self.run_dir

    def log_dataset(self, name: str, path: str | Path, shape: tuple[int, int]) -> None:
        """Record the source path and dimensions of a dataset used in this run."""

        self.metadata["datasets"][name] = {
            "path": str(path),
            "shape": list(shape),
        }
        self._save_metadata()

    def log_features(
        self,
        target: str,
        numerical: list[str],
        categorical: list[str],
    ) -> None:
        """Save the target and feature columns used for training."""

        features = OmegaConf.create({
            "target": target,
            "numerical": numerical,
            "categorical": categorical,
            "all": [*numerical, *categorical],
        })
        OmegaConf.save(features, self.run_dir / "features.yaml")
        self.log(f"Saved {len(features.all)} feature columns")

    def log_results(self, results: pd.DataFrame) -> None:
        """Save the final cross-validation table."""

        results.to_csv(self.run_dir / "results.csv", index=False)
        self.log("Saved cross-validation results")

    def finish(self) -> None:
        """Mark a run as completed."""

        self.metadata["status"] = "completed"
        self.metadata["finished_at"] = self._timestamp()
        self._save_metadata()
        self.log("Run completed")

    def fail(self, error: Exception) -> None:
        """Record a Python exception before propagating it to the caller."""

        self.metadata["status"] = "failed"
        self.metadata["finished_at"] = self._timestamp()
        self.metadata["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        self._save_metadata()
        self.log(f"Run failed: {type(error).__name__}: {error}")

    def log(self, message: str) -> None:
        """Write a message to both the console and the current run log."""

        self.logger.info(message)

    def _save_metadata(self) -> None:
        metadata_path = self.run_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().astimezone().isoformat(timespec="seconds")

    @staticmethod
    def _package_versions(requirements_path: Path) -> dict[str, str | None]:
        """Return installed versions for dependencies declared in requirements.txt."""

        if not requirements_path.exists():
            return {}

        versions = {}
        for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.split("#", maxsplit=1)[0].strip()

            if not line or line.startswith(("-", "http://", "https://", "git+")):
                continue

            package = re.split(r"[<>=!~;\[\s]", line, maxsplit=1)[0]

            if not package:
                continue

            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None

        return versions

    @staticmethod
    def _git_metadata() -> dict[str, str | bool | None]:
        def run_git(*args: str) -> str | None:
            result = subprocess.run(
                ["git", *args],
                capture_output=True,
                text=True,
                check=False,
            )
            return result.stdout.strip() if result.returncode == 0 else None

        commit = run_git("rev-parse", "HEAD")
        status = run_git("status", "--porcelain")

        return {
            "commit": commit,
            "is_dirty": bool(status) if status is not None else None,
        }
