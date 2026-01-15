from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
import os


class Settings(BaseSettings):
    """Configuraciones de la aplicación."""

    HF_TOKEN: str = os.getenv("HF_TOKEN", "")

    FILES_ORIGIN_URL: str = "jeffangel/demo-logistic-regression-no-black-box"
    DATA_FOLDER: Path = Path("/tmp")
    SAMPLE_DATA_FILENAME: str = "sample_data.pkl"
    TRAINING_OBJECT_FILENAME: str = "logistic_regression_model.pkl"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
