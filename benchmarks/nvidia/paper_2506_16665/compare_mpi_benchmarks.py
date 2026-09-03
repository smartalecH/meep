#!/usr/bin/env python3
"""Create a hash-bound staged-versus-direct comparison artifact."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import sys
import tempfile
from typing import Any, Mapping, Sequence

import benchmark_manifest as bm
import run_mpi_benchmark as runner


SCHEMA = pathlib.Path(__file__).with_name("mpi_route_comparison.schema.json")


def _snapshot(path: pathlib.Path) -> tuple[dict[str, Any], str]:
    payload = path.resolve().read_bytes()
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise runner.RunnerError("route result must be a JSON object")
    return value, hashlib.sha256(payload).hexdigest()


def _counters(result: Mapping[str, Any]) -> dict[str, int]:
    return {item["name"]: item["value"] for item in result["counter_aggregates"]}


def compare(staged: Mapping[str, Any], direct: Mapping[str, Any],
            staged_path: pathlib.Path, direct_path: pathlib.Path,
            staged_hash: str, direct_hash: str) -> dict[str, Any]:
    runner.validate_result(staged); runner.validate_result(direct)
    if staged["run_manifest"] != direct["run_manifest"] or staged["physics_reference"] != direct["physics_reference"]:
        raise runner.RunnerError("route results do not bind the same manifest and physics reference")
    if staged["requested_execution"]["route"] != "staged" or staged["resolved_execution"]["route"] != "staged":
        raise runner.RunnerError("staged artifact did not request and resolve staged transport")
    if direct["requested_execution"]["route"] != "direct" or direct["resolved_execution"]["route"] != "direct":
        raise runner.RunnerError("direct artifact did not request and resolve direct transport")
    for key in ("backend", "precision", "overlap", "graph", "ranks"):
        if staged["requested_execution"][key] != direct["requested_execution"][key]:
            raise runner.RunnerError(f"route comparison changed requested {key}")
    if staged["provenance"] != direct["provenance"]:
        raise runner.RunnerError("route comparison changed build provenance")
    for key in ("backend", "precision", "overlap", "graph"):
        if staged["resolved_execution"][key] != direct["resolved_execution"][key]:
            raise runner.RunnerError(f"route comparison changed resolved {key}")
    if staged["simulation"] != direct["simulation"]:
        raise runner.RunnerError("route comparison changed simulation dimensions or steps")
    staged_devices = [(r["rank"], r["device"]["uuid"] if r["device"] else None) for r in staged["rank_records"]]
    direct_devices = [(r["rank"], r["device"]["uuid"] if r["device"] else None) for r in direct["rank_records"]]
    if staged_devices != direct_devices:
        raise runner.RunnerError("route comparison changed rank/GPU mapping")
    staged_modules = [(record["module_paths"], record["module_sha256"])
                      for record in staged["rank_records"]]
    direct_modules = [(record["module_paths"], record["module_sha256"])
                      for record in direct["rank_records"]]
    if staged_modules != direct_modules:
        raise runner.RunnerError("route comparison changed loaded module/library paths or hashes")
    if ([record["cpu_affinity"] for record in staged["rank_records"]] !=
            [record["cpu_affinity"] for record in direct["rank_records"]]):
        raise runner.RunnerError("route comparison changed observed CPU affinity")
    for key in ("mpi_version", "ucx_version"):
        if staged["launch"][key] != direct["launch"][key]:
            raise runner.RunnerError(f"route comparison changed {key}")
    staged_monitors = [[rep["monitors"] for rep in rank["repetitions"]] for rank in staged["rank_records"]]
    direct_monitors = [[rep["monitors"] for rep in rank["repetitions"]] for rank in direct["rank_records"]]
    if staged_monitors != direct_monitors:
        raise runner.RunnerError("staged and direct raw monitor arrays are not bitwise identical")
    if staged["physics_observables"] != direct["physics_observables"]:
        raise runner.RunnerError("staged and direct derived observables are not identical")
    sc, dc = _counters(staged), _counters(direct)
    for name in ("messages_sent", "messages_received", "bytes_sent", "bytes_received"):
        if sc[name] != dc[name]:
            raise runner.RunnerError(f"route wire counter {name} differs")
    if sc["direct_bytes"] != 0 or sc["device_to_host_bytes"] != sc["bytes_sent"] or sc["host_to_device_bytes"] != sc["bytes_received"] or sc["transport_pinned_bytes"] <= 0:
        raise runner.RunnerError("staged transport counters violate the route contract")
    if dc["device_to_host_calls"] or dc["device_to_host_bytes"] or dc["host_to_device_calls"] or dc["host_to_device_bytes"] or dc["transport_pinned_bytes"] or dc["direct_bytes"] != dc["bytes_sent"] + dc["bytes_received"]:
        raise runner.RunnerError("direct transport counters violate the route contract")
    return {"schema_version": 1, "kind": "paper_2506_16665_staged_direct_comparison",
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
            "manifest_sha256": staged["run_manifest"]["sha256"],
            "physics_reference_sha256": staged["physics_reference"]["sha256"],
            "staged": {"path": str(staged_path.resolve()), "sha256": staged_hash},
            "direct": {"path": str(direct_path.resolve()), "sha256": direct_hash},
            "bitwise_monitor_match": True, "physics_match": True, "wire_totals_match": True}


def validate(value: Mapping[str, Any]) -> None:
    schema = bm.load_json_object(SCHEMA, "route comparison schema")
    bm._validate_schema_structure(value, schema, schema, "route comparison")


def atomic_write(value: Mapping[str, Any], path: pathlib.Path) -> None:
    validate(value); path = path.resolve(); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temporary = pathlib.Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(directory_fd)
        finally: os.close(directory_fd)
    finally:
        if temporary is not None and temporary.exists(): temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", required=True, type=pathlib.Path)
    parser.add_argument("--direct", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)
    try:
        staged, staged_hash = _snapshot(args.staged); direct, direct_hash = _snapshot(args.direct)
        runner.authenticate_result_files(staged)
        runner.authenticate_result_files(direct)
        result = compare(staged, direct, args.staged, args.direct, staged_hash, direct_hash)
        atomic_write(result, args.output)
        return 0
    except (OSError, ValueError, bm.ValidationError, runner.RunnerError) as error:
        print(f"compare_mpi_benchmarks.py: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
