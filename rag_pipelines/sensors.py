"""Sensor that polls Azure Blob Storage for new PDFs and triggers ingestion."""
from dagster import (
    AssetSelection,
    DefaultSensorStatus,
    RunRequest,
    SensorEvaluationContext,
    SensorResult,
    sensor,
)

from rag_pipelines.assets import ingested_pdf, pdf_partitions
from rag_pipelines.resources import AzureBlobResource


@sensor(
    asset_selection=AssetSelection.assets(ingested_pdf),
    minimum_interval_seconds=60,
    # Stopped by default so tests/CI don't accidentally start polling a real
    # container. Enable in the Dagster UI when you're ready.
    default_status=DefaultSensorStatus.STOPPED,
)
def new_pdf_blob_sensor(
    context: SensorEvaluationContext,
    azure_blob: AzureBlobResource,
) -> SensorResult:
    current = set(azure_blob.list_pdfs())
    known = set(context.instance.get_dynamic_partitions(pdf_partitions.name))
    new_blobs = sorted(current - known)

    if not new_blobs:
        return SensorResult(run_requests=[], skip_reason="No new PDFs in container.")

    return SensorResult(
        run_requests=[
            RunRequest(partition_key=b, run_key=f"ingest:{b}") for b in new_blobs
        ],
        dynamic_partitions_requests=[pdf_partitions.build_add_request(new_blobs)],
    )
