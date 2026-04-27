"""Dagster resources: Azure Blob Storage client wrapper."""
from pathlib import Path

from azure.storage.blob import BlobServiceClient, ContainerClient
from dagster import ConfigurableResource


class AzureBlobResource(ConfigurableResource):
    """Lists and downloads PDFs from a single Azure Storage container.

    Connection string + container name are injected from env vars in
    `definitions.py`, so the same code works locally and in Dagster+ Cloud
    (which supplies env vars as workspace secrets).
    """

    connection_string: str
    container: str

    def _client(self) -> ContainerClient:
        return BlobServiceClient.from_connection_string(
            self.connection_string
        ).get_container_client(self.container)

    def list_pdfs(self) -> list[str]:
        """Return PDF blob names in deterministic order so the sensor sees stable diffs."""
        return sorted(
            b.name
            for b in self._client().list_blobs()
            if b.name.lower().endswith(".pdf")
        )

    def download_to(self, blob_name: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob = self._client().get_blob_client(blob_name)
        with open(dest, "wb") as f:
            f.write(blob.download_blob().readall())
