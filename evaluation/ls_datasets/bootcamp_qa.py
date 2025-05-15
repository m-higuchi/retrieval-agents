# noqa: D100
from langchain_core.documents import Document
from langsmith.schemas import Dataset


def create_dataset() -> Dataset:  # noqa: D103
    import langsmith

    client = langsmith.Client()
    return client.clone_public_dataset(
        "https://smith.langchain.com/public/56fe54cd-b7d7-4d3b-aaa0-88d7a2d30931/d"
    )


def get_source_documents(metadata) -> list[Document]:  # type: ignore[no-untyped-def]  # noqa: D103
    import io
    import os
    import zipfile

    import requests

    # Fetch the source documents
    url = "https://storage.googleapis.com/benchmarks-artifacts/basecamp-data/basecamp-data.zip"

    response = requests.get(url)
    docs = []
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        for filename in z.namelist():
            if filename.endswith("/") or not filename.lower().endswith(".md"):
                continue
            with z.open(filename) as f:
                docs.append(
                    Document(
                        page_content=f.read().decode("utf-8"),
                        metadata={**metadata, "source": os.path.basename(filename)},
                    )
                )
    return docs
