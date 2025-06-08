"""Helpers for running the SWE-bench evaluation harness."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable

from swebench.harness.run_evaluation import main as swe_run_main

__all__ = ["SWE_BENCH_LITE_DATASET", "run_swe_benchmark"]

# Default dataset names
SWE_BENCH_LITE_DATASET: str = "SWE-bench/SWE-bench_Lite"

_ALIASES: dict[str, str] = {
    "lite": SWE_BENCH_LITE_DATASET,
    "swe-bench-lite": SWE_BENCH_LITE_DATASET,
    "swebench-lite": SWE_BENCH_LITE_DATASET,
}


def run_swe_benchmark(
    *,
    dataset_name: str = "lite",
    split: str = "test",
    instance_ids: Iterable[str] | None = None,
    predictions_path: str,
    max_workers: int = 4,
    force_rebuild: bool = False,
    cache_level: str = "env",
    clean: bool = False,
    open_file_limit: int = 4096,
    run_id: str | None = None,
    timeout: int = 1800,
    namespace: str | None = "swebench",
    rewrite_reports: bool = False,
    modal: bool = False,
    instance_image_tag: str = "latest",
    report_dir: str = ".",
) -> dict:
    """Run the SWE-bench evaluation harness.

    This is a thin wrapper around :func:`swebench.harness.run_evaluation.main`.
    All arguments map directly to the underlying harness.
    """

    if dataset_name in _ALIASES:
        dataset_name = _ALIASES[dataset_name]

    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    ids = list(instance_ids) if instance_ids is not None else []
    return swe_run_main(
        dataset_name=dataset_name,
        split=split,
        instance_ids=ids,
        predictions_path=predictions_path,
        max_workers=max_workers,
        force_rebuild=force_rebuild,
        cache_level=cache_level,
        clean=clean,
        open_file_limit=open_file_limit,
        run_id=run_id,
        timeout=timeout,
        namespace=namespace,
        rewrite_reports=rewrite_reports,
        modal=modal,
        instance_image_tag=instance_image_tag,
        report_dir=report_dir,
    )

