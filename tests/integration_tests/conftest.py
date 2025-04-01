import os
from pathlib import Path

import pytest
from dotenv import load_dotenv


@pytest.fixture(scope="session", autouse=True)
def load_env():
    load_dotenv(dotenv_path=Path(__file__).parent / ".env.integration")
    os.environ["CHROMA_DIR"] = str(Path(__file__).parent / ".data")
