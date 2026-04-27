"""Dagster assets: each PDF blob becomes a partition of `ingested_pdf`."""
import tempfile
from pathlib import Path

from dagster import (
    AssetExecutionContext,
    DynamicPartitionsDefinition,
    MaterializeResult,
    MetadataValue,
    asset,
)

from rag_pipelines.resources import AzureBlobResource
from src.ingest.pipeline import ingest_path

# Each blob name is one partition. The sensor populates this set as new
# files arrive in the container; once a partition is materialized, Dagster
# won't re-run it unless explicitly asked — giving us idempotency for free.
pdf_partitions = DynamicPartitionsDefinition(name="azure_pdf_blobs")


@asset(partitions_def=pdf_partitions, group_name="ingest")
def ingested_pdf(
    context: AssetExecutionContext,
    azure_blob: AzureBlobResource,
) -> MaterializeResult:
    blob_name = context.partition_key

    with tempfile.TemporaryDirectory() as tmp_dir:
        local = Path(tmp_dir) / Path(blob_name).name
        azure_blob.download_to(blob_name, local)
        result = ingest_path(local)

    if result["chunks"] == 0:
        # Don't raise — a scanned PDF will fail forever otherwise. Surface it
        # in the UI as a successful-but-empty materialization so it's visible.
        context.log.warning(
            f"{blob_name}: no extractable text — likely a scanned PDF, needs OCR."
        )

    return MaterializeResult(
        metadata={
            "source_blob": blob_name,
            "pages": result["pages"],
            "chunks": result["chunks"],
            "qdrant_id_sample": MetadataValue.json(result["ids"][:5]),
        }
    )
