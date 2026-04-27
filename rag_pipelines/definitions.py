"""Dagster entrypoint. `dagster dev` discovers this via [tool.dagster] in pyproject.toml."""
from dagster import Definitions, EnvVar

from rag_pipelines.assets import ingested_pdf
from rag_pipelines.resources import AzureBlobResource
from rag_pipelines.sensors import new_pdf_blob_sensor

defs = Definitions(
    assets=[ingested_pdf],
    sensors=[new_pdf_blob_sensor],
    resources={
        "azure_blob": AzureBlobResource(
            connection_string=EnvVar("AZURE_BLOB_CONNECTION_STRING"),
            container=EnvVar("AZURE_BLOB_CONTAINER"),
        ),
    },
)
