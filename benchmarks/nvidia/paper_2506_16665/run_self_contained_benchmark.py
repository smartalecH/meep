#!/usr/bin/env python3
"""Run a dependency-free fixed-step CPU/HIP benchmark.

This complements the external-GDS paper runner.  It deliberately uses a tiny
built-in vacuum problem so accelerator and MPI scaling can be measured before
the external corpus is available.  Every result retains raw rank timings,
runtime counters, physical device identity, and the exact launch environment.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import math
import os
import pathlib
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import benchmark_manifest as bm
import run_mpi_benchmark as mpi_runner


HERE = pathlib.Path(__file__).resolve().parent
SCHEMA = HERE / "self_contained_benchmark.schema.json"
CASE = {
    "name": "builtin_3d_vacuum_fixed_step",
    "cell": [6.0, 4.0, 4.0],
    "resolution": 8,
    "pml": 0.5,
    "component": "Ez",
    "frequency": 0.35,
    "fwidth": 0.14,
    "amplitude": 0.37,
    "source_center": [-1.5, 0.0, 0.0],
    "sample_points": [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    "warmup_steps": 100,
    "measured_steps": 100,
    "measured_repetitions": 5,
    "repetition_mode": "one_initialization_sequential_steady_windows",
}
FORBIDDEN_STEADY_COUNTERS = (
    "steady_allocation_count",
    "graph_recapture_count",
    "full_field_copy_count",
    "host_fallback_count",
    "host_fallback_device_to_host_bytes",
    "host_fallback_host_to_device_bytes",
    "host_fallback_steady_capacity_growths",
    "material_fallback_warning_count",
)


class SelfContainedError(RuntimeError):
    pass


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def source_tree_state(source: pathlib.Path) -> dict[str, Any]:
    source = source.resolve()

    def git_bytes(*arguments: str) -> bytes:
        try:
            return subprocess.run(
                ["git", "-C", str(source), *arguments],
                check=True,
                capture_output=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as error:
            raise SelfContainedError(
                f"cannot authenticate source tree: {error}"
            ) from error

    commit = git_bytes("rev-parse", "HEAD").decode("ascii").strip()
    status = git_bytes("status", "--porcelain=v1", "--untracked-files=all", "-z")
    clean = not status
    dirty_diff_sha256 = None
    if not clean:
        digest = hashlib.sha256()
        digest.update(b"tracked-diff\0")
        digest.update(git_bytes("diff", "--binary", "HEAD", "--", "."))
        untracked = git_bytes("ls-files", "--others", "--exclude-standard", "-z")
        for encoded_path in sorted(item for item in untracked.split(b"\0") if item):
            path = source / encoded_path.decode("utf-8", errors="surrogateescape")
            digest.update(b"\0untracked-path\0")
            digest.update(encoded_path)
            digest.update(b"\0untracked-content\0")
            if path.is_symlink():
                digest.update(
                    os.readlink(path).encode("utf-8", errors="surrogateescape")
                )
            else:
                digest.update(path.read_bytes())
        dirty_diff_sha256 = digest.hexdigest()
    return {
        "commit": commit,
        "source_clean": clean,
        "source_diff_sha256": dirty_diff_sha256,
    }


def benchmark_source_provenance(source: pathlib.Path) -> dict[str, Any]:
    runner = pathlib.Path(__file__).resolve()
    state = source_tree_state(source)
    return {
        **state,
        "dirty": not state["source_clean"],
        "runner_source": {"path": str(runner), "sha256": bm.sha256_file(runner)},
    }


def validate_benchmark_source_provenance(provenance: Mapping[str, Any]) -> None:
    source = pathlib.Path(str(provenance.get("source", ""))).resolve()
    runner = pathlib.Path(__file__).resolve()
    recorded_runner = provenance.get("runner_source")
    if not isinstance(recorded_runner, Mapping):
        raise SelfContainedError("benchmark provenance lacks its runner source hash")
    runner_hash = str(recorded_runner.get("sha256", ""))
    if (
        pathlib.Path(str(recorded_runner.get("path", ""))).resolve() != runner
        or runner_hash == "0" * 64
        or runner_hash != bm.sha256_file(runner)
    ):
        raise SelfContainedError("benchmark runner source hash is stale")
    current = source_tree_state(source)
    if (
        provenance.get("commit") != current["commit"]
        or provenance.get("source_clean") is not current["source_clean"]
        or provenance.get("dirty") != (not current["source_clean"])
        or provenance.get("source_diff_sha256") != current["source_diff_sha256"]
        or (
            not current["source_clean"]
            and provenance.get("source_diff_sha256") in (None, "", "0" * 64)
        )
    ):
        raise SelfContainedError("benchmark source-tree state is stale")


def _atomic(value: Mapping[str, Any], path: pathlib.Path) -> None:
    schema = bm.load_json_object(SCHEMA, "self-contained benchmark schema")
    bm._validate_schema_structure(value, schema, schema, "self-contained benchmark")
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            temporary = pathlib.Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _complex_samples(mp: Any, simulation: Any) -> list[list[float]]:
    component = getattr(mp, CASE["component"])
    result = []
    for point in CASE["sample_points"]:
        value = complex(simulation.get_field_point(component, mp.Vector3(*point)))
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise SelfContainedError("field sample is not finite")
        result.append([float(value.real), float(value.imag)])
    return result


def _build_simulation(mp: Any, backend: str, device_id: int) -> Any:
    kwargs: dict[str, Any] = {
        "cell_size": mp.Vector3(*CASE["cell"]),
        "resolution": CASE["resolution"],
        "boundary_layers": [mp.PML(CASE["pml"])],
        "geometry": [],
        "sources": [
            mp.Source(
                mp.GaussianSource(CASE["frequency"], fwidth=CASE["fwidth"]),
                component=getattr(mp, CASE["component"]),
                center=mp.Vector3(*CASE["source_center"]),
                amplitude=CASE["amplitude"],
            )
        ],
        "backend": backend,
        "accelerator_strict": backend != "cpu",
    }
    if backend != "cpu":
        kwargs["device_id"] = device_id
    return mp.Simulation(**kwargs)


def _counter_snapshot(runtime: Mapping[str, Any]) -> dict[str, int]:
    return {name: int(runtime.get(name, 0)) for name in mpi_runner.COUNTER_AGGREGATIONS}


def _worker(args: argparse.Namespace) -> int:
    mp = importlib.import_module("meep")
    mp.verbosity(0)
    # Accelerator compilation requires the backend communicator context even
    # for a singleton run; CPU singleton references do not.
    divided = args.backend != "cpu" or mp.count_processors() > 1
    if divided:
        mp.begin_global_communications()
        mp.divide_parallel_processes(1)
    gather = getattr(mp, "active_communicator_allgather_json", None)
    if gather is None:
        raise SelfContainedError("Meep lacks active_communicator_allgather_json")
    rank = int(mp.my_rank())
    size = int(mp.count_processors())
    if size != args.ranks:
        raise SelfContainedError(
            f"communicator has {size} ranks, expected {args.ranks}"
        )
    local_rank = mpi_runner._local_rank()
    repetitions = []
    started = time.perf_counter()
    simulation = _build_simulation(mp, args.backend, local_rank)
    simulation.init_sim()
    initialization_seconds = time.perf_counter() - started
    warmup_started = time.perf_counter()
    simulation.fields.advance(CASE["warmup_steps"])
    mp.all_wait()
    warmup_seconds = time.perf_counter() - warmup_started
    final_runtime = None
    for repetition_index in range(CASE["measured_repetitions"]):
        before = simulation.get_execution_runtime_report()
        started = time.perf_counter()
        simulation.fields.advance(CASE["measured_steps"])
        mp.all_wait()
        steady_seconds = time.perf_counter() - started
        after = simulation.get_execution_runtime_report()
        samples = _complex_samples(mp, simulation)
        start_counters = _counter_snapshot(before)
        end_counters = _counter_snapshot(after)
        deltas = {
            name: end_counters[name] - start_counters[name] for name in start_counters
        }
        if any(value < 0 for value in deltas.values()):
            raise SelfContainedError("runtime counter decreased during measured work")
        repetitions.append(
            {
                "initialization_seconds": (
                    initialization_seconds if repetition_index == 0 else 0.0
                ),
                "warmup_seconds": warmup_seconds if repetition_index == 0 else 0.0,
                "steady_seconds": steady_seconds,
                "steps": CASE["measured_steps"],
                "dt_meep": float(simulation.fields.dt),
                "grid_shape": mpi_runner.legacy._grid_shape(simulation),
                "counter_start": start_counters,
                "counter_end": end_counters,
                "counter_deltas": deltas,
                "samples": samples,
            }
        )
        final_runtime = after
    simulation.reset_meep()
    if final_runtime is None:
        raise SelfContainedError("no repetition ran")

    affinity = mpi_runner.observed_cpu_affinity()
    selectors = [item for item in args.visible_devices.split(",") if item]
    device = None
    if args.backend != "cpu":
        device = mpi_runner.accelerator_device_provenance(
            args.accelerator,
            int(final_runtime["device_id"]),
            str(final_runtime["device_uuid"]),
            selectors,
            args.rocm_smi,
        )
    launch_payload = args.launch_record_file.read_bytes()
    provenance_payload = args.provenance_record_file.read_bytes()
    module = pathlib.Path(str(mp.__file__)).resolve()
    extension = pathlib.Path(
        str(getattr(getattr(mp, "_meep", None), "__file__", ""))
    ).resolve()
    local = {
        "rank": rank,
        "communicator_size": size,
        "local_rank": local_rank,
        "hostname": socket.gethostname(),
        "role": "cpu" if args.backend == "cpu" else "owner",
        "device": device,
        "cpu_affinity": affinity,
        "cpu_numa_nodes": mpi_runner.observed_cpu_numa_nodes(affinity),
        "runtime": final_runtime,
        "repetitions": repetitions,
        "module_paths": {"meep": str(module), "extension": str(extension)},
        "module_sha256": {
            "meep": bm.sha256_file(module),
            "extension": bm.sha256_file(extension),
        },
        "launch_sha256": hashlib.sha256(launch_payload).hexdigest(),
        "provenance_sha256": hashlib.sha256(provenance_payload).hexdigest(),
    }
    gathered = [
        json.loads(value) for value in gather(json.dumps(local, sort_keys=True))
    ]
    if rank == 0:
        critical = [
            max(
                float(record["repetitions"][index]["steady_seconds"])
                for record in gathered
            )
            for index in range(CASE["measured_repetitions"])
        ]
        shape = gathered[0]["repetitions"][0]["grid_shape"]
        result = {
            "schema_version": 1,
            "kind": "meep_self_contained_fixed_step_benchmark",
            "generated_at_utc": dt.datetime.now(dt.timezone.utc)
            .replace(microsecond=0)
            .isoformat(),
            "case": CASE,
            "case_sha256": canonical_hash(CASE),
            "launch": json.loads(launch_payload),
            "provenance": json.loads(provenance_payload),
            "rank_records": gathered,
            "timing": {
                "critical_path_samples_seconds": critical,
                "minimum_seconds": min(critical),
                "median_seconds": statistics.median(critical),
                "maximum_seconds": max(critical),
                "grid_timesteps_per_second": (
                    math.prod(shape)
                    * CASE["measured_steps"]
                    / statistics.median(critical)
                ),
            },
            "validation": {
                "fixed_step_protocol": True,
                "steady_state_clean": True,
                "transport_accounting": True,
                "topology_attested": args.backend == "cpu" or args.accelerator == "hip",
                "cpu_reference": None,
                "cpu_tolerance_passed": None,
                "peer_route": None,
                "peer_route_bitwise": None,
            },
        }
        validate_result(result, allow_unbound_references=True)
        _atomic(result, args.output)
    if divided:
        mp.end_divide_parallel()
    return 0


def _load_result(
    path: pathlib.Path, *, allow_unbound_references: bool = False
) -> dict[str, Any]:
    value = bm.load_json_object(path.resolve(), "self-contained benchmark result")
    validate_result(value, allow_unbound_references=allow_unbound_references)
    return value


def _finite_number(
    value: Any, label: str, *, positive: bool = False, allow_negative: bool = False
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise SelfContainedError(f"{label} is not finite")
    number = float(value)
    if (not allow_negative and number < 0) or (positive and number <= 0):
        raise SelfContainedError(f"{label} has an invalid sign")
    return number


def _load_linked_result(reference: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(reference, Mapping):
        raise SelfContainedError(f"{label} reference is missing")
    digest = str(reference.get("sha256", ""))
    if len(digest) != 64 or digest == "0" * 64:
        raise SelfContainedError(f"{label} reference hash is invalid")
    path = pathlib.Path(str(reference.get("path", ""))).resolve()
    try:
        actual_digest = bm.sha256_file(path)
    except OSError as error:
        raise SelfContainedError(f"cannot read {label} reference: {path}") from error
    if actual_digest != digest:
        raise SelfContainedError(f"{label} reference hash does not match its content")
    value = bm.load_json_object(path, f"{label} reference")
    validate_result(value, allow_unbound_references=True)
    return value


def _validate_cpu_samples(
    result: Mapping[str, Any], reference: Mapping[str, Any]
) -> None:
    if (
        reference["launch"]["backend"] != "cpu"
        or reference["case_sha256"] != result["case_sha256"]
    ):
        raise SelfContainedError("CPU reference is not the same self-contained case")
    expected_repetitions = reference["rank_records"][0]["repetitions"]
    for record in result["rank_records"]:
        for repetition_index, repetition in enumerate(record["repetitions"]):
            expected = expected_repetitions[repetition_index]
            if (
                repetition["steps"] != expected["steps"]
                or repetition["dt_meep"] != expected["dt_meep"]
                or repetition["grid_shape"] != expected["grid_shape"]
            ):
                raise SelfContainedError("GPU and CPU fixed-step work differs")
            for got_pair, want_pair in zip(repetition["samples"], expected["samples"]):
                got, want = complex(*got_pair), complex(*want_pair)
                if abs(got - want) > 1e-11 + 1e-8 * abs(want):
                    raise SelfContainedError(
                        "GPU field sample failed CPU tolerance in a measured window"
                    )


def _validate_peer_samples(result: Mapping[str, Any], peer: Mapping[str, Any]) -> None:
    matching_launch_fields = (
        "backend",
        "accelerator",
        "ranks",
        "omp_threads",
        "visible_devices",
        "map_by",
        "rank_by",
        "bind_to",
    )
    if (
        peer["case_sha256"] != result["case_sha256"]
        or peer["launch"]["route"] != "staged"
        or any(
            peer["launch"][name] != result["launch"][name]
            for name in matching_launch_fields
        )
        or peer["validation"]["cpu_reference"] != result["validation"]["cpu_reference"]
        or peer["validation"]["cpu_tolerance_passed"] is not True
    ):
        raise SelfContainedError(
            "peer route did not use the same source, case, and rank/device mapping"
        )
    peer_modules = [record["module_sha256"] for record in peer["rank_records"]]
    result_modules = [record["module_sha256"] for record in result["rank_records"]]
    if peer_modules != result_modules:
        raise SelfContainedError("peer route loaded different Meep modules")
    if [
        [rep["samples"] for rep in record["repetitions"]]
        for record in peer["rank_records"]
    ] != [
        [rep["samples"] for rep in record["repetitions"]]
        for record in result["rank_records"]
    ]:
        raise SelfContainedError(
            "staged/direct field samples are not bitwise identical"
        )


def validate_result(
    value: Mapping[str, Any], *, allow_unbound_references: bool = False
) -> None:
    schema = bm.load_json_object(SCHEMA, "self-contained benchmark schema")
    bm._validate_schema_structure(value, schema, schema, "self-contained benchmark")
    if value["case"] != CASE or value["case_sha256"] != canonical_hash(CASE):
        raise SelfContainedError(
            "self-contained case does not match the canonical definition"
        )
    validate_benchmark_source_provenance(value["provenance"])
    launch = value["launch"]
    ranks = int(launch["ranks"])
    records = value["rank_records"]
    if ranks not in {1, 2, 4, 8} or len(records) != ranks:
        raise SelfContainedError("result does not contain the requested 1/2/4/8 ranks")
    if [record["rank"] for record in records] != list(range(ranks)):
        raise SelfContainedError("rank records are not communicator ordered")
    canonical = lambda item: json.dumps(item, sort_keys=True).encode("utf-8")
    launch_hash = hashlib.sha256(canonical(launch)).hexdigest()
    provenance_hash = hashlib.sha256(canonical(value["provenance"])).hexdigest()
    expected_samples_by_repetition: list[Any] = []
    devices: list[Mapping[str, Any]] = []
    shapes, timesteps = set(), set()
    total_messages_sent = total_messages_received = 0
    total_bytes_sent = total_bytes_received = 0
    counter_names = set(mpi_runner.COUNTER_AGGREGATIONS)
    for record in records:
        if (
            record["communicator_size"] != ranks
            or record["launch_sha256"] != launch_hash
            or record["provenance_sha256"] != provenance_hash
        ):
            raise SelfContainedError(
                "rank identity or launch/provenance binding is inconsistent"
            )
        if len(record["repetitions"]) != CASE["measured_repetitions"]:
            raise SelfContainedError("result does not retain all measured repetitions")
        for repetition_index, repetition in enumerate(record["repetitions"]):
            if repetition["steps"] != CASE["measured_steps"]:
                raise SelfContainedError("measured step count is not canonical")
            initialization = _finite_number(
                repetition["initialization_seconds"], "initialization time"
            )
            warmup = _finite_number(repetition["warmup_seconds"], "warmup time")
            _finite_number(repetition["steady_seconds"], "steady time", positive=True)
            timestep = _finite_number(
                repetition["dt_meep"], "Meep timestep", positive=True
            )
            shape = repetition["grid_shape"]
            if (
                not isinstance(shape, list)
                or len(shape) != 3
                or any(
                    isinstance(item, bool) or not isinstance(item, int) or item <= 0
                    for item in shape
                )
            ):
                raise SelfContainedError("measured grid shape is invalid")
            if (repetition_index == 0 and (initialization <= 0 or warmup <= 0)) or (
                repetition_index > 0 and (initialization != 0 or warmup != 0)
            ):
                raise SelfContainedError(
                    "result does not describe one initialization and warmup"
                )
            shapes.add(tuple(shape))
            timesteps.add(timestep)
            for snapshot_name in ("counter_start", "counter_end", "counter_deltas"):
                snapshot = repetition[snapshot_name]
                if set(snapshot) != counter_names:
                    raise SelfContainedError(
                        "counter snapshot does not contain the declared set"
                    )
                if any(
                    isinstance(item, bool) or not isinstance(item, int) or item < 0
                    for item in snapshot.values()
                ):
                    raise SelfContainedError(
                        "counter snapshot contains an invalid value"
                    )
            for name in counter_names:
                if (
                    repetition["counter_end"][name] - repetition["counter_start"][name]
                    != repetition["counter_deltas"][name]
                ):
                    raise SelfContainedError(
                        f"counter delta {name} disagrees with its snapshots"
                    )
            if any(
                repetition["counter_deltas"][name] for name in FORBIDDEN_STEADY_COUNTERS
            ):
                raise SelfContainedError(
                    "measured work allocated, recopied, recaptured, or fell back"
                )
            samples = repetition["samples"]
            if (
                not isinstance(samples, list)
                or len(samples) != len(CASE["sample_points"])
                or any(
                    not isinstance(sample, list)
                    or len(sample) != 2
                    or any(
                        not math.isfinite(
                            _finite_number(
                                component, "field sample", allow_negative=True
                            )
                        )
                        for component in sample
                    )
                    for sample in samples
                )
            ):
                raise SelfContainedError("measured field sample set is invalid")
            if len(expected_samples_by_repetition) <= repetition_index:
                expected_samples_by_repetition.append(samples)
            elif samples != expected_samples_by_repetition[repetition_index]:
                raise SelfContainedError(
                    "field samples disagree across ranks within a measured window"
                )
        if launch["backend"] == "cpu":
            if record["device"] is not None:
                raise SelfContainedError("CPU result unexpectedly records a GPU")
            continue
        runtime = record["runtime"]
        device = record["device"]
        if not runtime["device_owner"] or runtime["resolved_backend"] != "nvidia":
            raise SelfContainedError("GPU rank did not resolve the accelerator backend")
        if (
            device["visible_devices"] != launch["visible_devices"]
            or str(device["physical_selector"])
            != str(launch["visible_devices"][record["local_rank"]])
            or mpi_runner.normalize_gpu_uuid(device["uuid"])
            != mpi_runner.normalize_gpu_uuid(runtime["device_uuid"])
        ):
            raise SelfContainedError("GPU rank identity disagrees with its launch")
        if record["cpu_numa_nodes"] != [device["numa_node"]]:
            raise SelfContainedError("GPU rank is not bound within its GPU NUMA node")
        devices.append(device)
        deltas = {
            name: sum(rep["counter_deltas"][name] for rep in record["repetitions"])
            for name in mpi_runner.COUNTER_AGGREGATIONS
        }
        if ranks == 1:
            if any(
                deltas[name]
                for name in (
                    "messages_sent",
                    "messages_received",
                    "bytes_sent",
                    "bytes_received",
                    "direct_bytes",
                )
            ):
                raise SelfContainedError(
                    "single-rank result unexpectedly recorded MPI traffic"
                )
        elif launch["route"] == "direct":
            if (
                not runtime["mpi_query_available"]
                or not runtime["mpi_cuda_aware"]
                or runtime["resolved_transport"] != "direct"
                or deltas["direct_bytes"] <= 0
                or deltas["direct_bytes"]
                != deltas["bytes_sent"] + deltas["bytes_received"]
                or any(
                    deltas[name]
                    for name in (
                        "device_to_host_calls",
                        "device_to_host_bytes",
                        "host_to_device_calls",
                        "host_to_device_bytes",
                    )
                )
                or runtime["transport_pinned_bytes"] != 0
            ):
                raise SelfContainedError(
                    "direct transport accounting is not fail-closed"
                )
        elif launch["route"] == "staged":
            if (
                runtime["resolved_transport"] != "staged"
                or deltas["direct_bytes"] != 0
                or deltas["device_to_host_bytes"] != deltas["bytes_sent"]
                or deltas["host_to_device_bytes"] != deltas["bytes_received"]
                or deltas["device_to_host_calls"] <= 0
                or deltas["host_to_device_calls"] <= 0
                or runtime["transport_pinned_bytes"] <= 0
            ):
                raise SelfContainedError(
                    "staged transport accounting is not fail-closed"
                )
        total_messages_sent += deltas["messages_sent"]
        total_messages_received += deltas["messages_received"]
        total_bytes_sent += deltas["bytes_sent"]
        total_bytes_received += deltas["bytes_received"]
    if len(shapes) != 1 or len(timesteps) != 1:
        raise SelfContainedError("grid shape or timestep differs across ranks/windows")
    if (
        total_messages_sent != total_messages_received
        or total_bytes_sent != total_bytes_received
    ):
        raise SelfContainedError("global MPI send/receive accounting is unbalanced")
    if devices:
        for name, normalized in (
            ("uuid", lambda item: mpi_runner.normalize_gpu_uuid(item)),
            ("pci_bus_id", lambda item: mpi_runner.normalize_pci_bus_id(item)),
            ("physical_unique_id", str),
        ):
            identities = [normalized(device[name]) for device in devices]
            if len(set(identities)) != len(identities):
                raise SelfContainedError(f"GPU ranks have duplicate physical {name}")
    expected_critical = [
        max(float(record["repetitions"][index]["steady_seconds"]) for record in records)
        for index in range(CASE["measured_repetitions"])
    ]
    shape = next(iter(shapes))
    expected_timing = {
        "critical_path_samples_seconds": expected_critical,
        "minimum_seconds": min(expected_critical),
        "median_seconds": statistics.median(expected_critical),
        "maximum_seconds": max(expected_critical),
        "grid_timesteps_per_second": (
            math.prod(shape)
            * CASE["measured_steps"]
            / statistics.median(expected_critical)
        ),
    }
    if value["timing"] != expected_timing:
        raise SelfContainedError(
            "published timing/rate is not derived from slowest-rank samples"
        )

    validation = value["validation"]
    if launch["backend"] == "cpu":
        if (
            validation["cpu_reference"] is not None
            or validation["cpu_tolerance_passed"] is not None
            or validation["peer_route"] is not None
            or validation["peer_route_bitwise"] is not None
        ):
            raise SelfContainedError("CPU result contains accelerator reference claims")
    elif not allow_unbound_references:
        if validation["cpu_tolerance_passed"] is not True:
            raise SelfContainedError("GPU result lacks a validated CPU reference")
        reference = _load_linked_result(validation["cpu_reference"], "CPU")
        _validate_cpu_samples(value, reference)
        if ranks > 1 and launch["route"] == "direct":
            if validation["peer_route_bitwise"] is not True:
                raise SelfContainedError("direct result lacks a bitwise staged peer")
            peer = _load_linked_result(validation["peer_route"], "staged peer")
            _validate_peer_samples(value, peer)
        elif (
            validation["peer_route"] is not None
            or validation["peer_route_bitwise"] is not None
        ):
            raise SelfContainedError("result contains an inapplicable peer-route claim")


def _compare_reference(result: dict[str, Any], reference_path: pathlib.Path) -> None:
    reference = _load_result(reference_path)
    _validate_cpu_samples(result, reference)
    result["validation"]["cpu_reference"] = {
        "path": str(reference_path.resolve()),
        "sha256": bm.sha256_file(reference_path),
    }
    result["validation"]["cpu_tolerance_passed"] = True


def _compare_peer(result: dict[str, Any], peer_path: pathlib.Path) -> None:
    peer = _load_result(peer_path)
    _validate_peer_samples(result, peer)
    result["validation"]["peer_route"] = {
        "path": str(peer_path.resolve()),
        "sha256": bm.sha256_file(peer_path),
    }
    result["validation"]["peer_route_bitwise"] = True


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--output", required=True, type=pathlib.Path)
    value.add_argument("--backend", required=True, choices=("cpu", "nvidia"))
    value.add_argument("--accelerator", choices=("hip", "cuda"), default="hip")
    value.add_argument("--route", required=True, choices=("host", "staged", "direct"))
    value.add_argument("--ranks", required=True, type=int, choices=(1, 2, 4, 8))
    value.add_argument("--visible-devices", default="")
    value.add_argument("--omp-threads", type=int, default=1)
    value.add_argument("--python", required=True, type=pathlib.Path)
    value.add_argument("--mpiexec", required=True, type=pathlib.Path)
    value.add_argument("--build-directory", required=True, type=pathlib.Path)
    value.add_argument("--library-path", required=True, type=pathlib.Path)
    value.add_argument(
        "--runtime-library-path",
        action="append",
        required=True,
        type=pathlib.Path,
        help="runtime library directory; repeat in intended loader order",
    )
    value.add_argument("--toolkit-compiler", type=pathlib.Path)
    value.add_argument("--rocm-smi", type=pathlib.Path)
    value.add_argument("--ucx-info", type=pathlib.Path)
    value.add_argument("--map-by")
    value.add_argument("--rank-by")
    value.add_argument("--bind-to")
    value.add_argument("--reference", type=pathlib.Path)
    value.add_argument("--peer-route-result", type=pathlib.Path)
    value.add_argument("--timeout", type=float, default=600.0)
    value.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    value.add_argument(
        "--launch-record-file", type=pathlib.Path, help=argparse.SUPPRESS
    )
    value.add_argument(
        "--provenance-record-file", type=pathlib.Path, help=argparse.SUPPRESS
    )
    return value


def _required_output(command: Sequence[str], environment: Mapping[str, str]) -> str:
    try:
        return subprocess.run(
            list(command),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=dict(environment),
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise SelfContainedError(
            f"cannot record {' '.join(command)}: {error}"
        ) from error


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.worker:
            return _worker(args)
        if not args.map_by or not args.bind_to:
            raise SelfContainedError(
                "outer launch requires explicit --map-by and --bind-to"
            )
        if args.backend == "cpu":
            if args.route != "host" or args.ranks != 1 or args.visible_devices:
                raise SelfContainedError(
                    "CPU references require route=host, ranks=1, no devices"
                )
        else:
            selectors = [
                item.strip() for item in args.visible_devices.split(",") if item.strip()
            ]
            if len(selectors) != args.ranks or len(set(selectors)) != len(selectors):
                raise SelfContainedError(
                    "GPU run requires one distinct visible selector per rank"
                )
            if args.accelerator == "hip" and args.rocm_smi is None:
                raise SelfContainedError("HIP run requires --rocm-smi")
            if args.reference is None:
                raise SelfContainedError("GPU run requires a CPU --reference artifact")
            if (
                args.route == "direct"
                and args.ranks > 1
                and args.peer_route_result is None
            ):
                raise SelfContainedError(
                    "multi-rank direct run requires a staged peer result"
                )
        for path, label in (
            (args.python, "Python"),
            (args.mpiexec, "MPI launcher"),
            (args.build_directory, "build directory"),
            (args.library_path, "Python library path"),
        ):
            if not path.resolve().exists():
                raise SelfContainedError(f"{label} does not exist: {path.resolve()}")

        env = dict(os.environ)
        env["PYTHONPATH"] = str(args.library_path.resolve())
        runtime_library_paths = list(
            dict.fromkeys(path.resolve() for path in args.runtime_library_path)
        )
        missing_runtime_paths = [
            path for path in runtime_library_paths if not path.is_dir()
        ]
        if missing_runtime_paths:
            raise SelfContainedError(
                "runtime library directories do not exist: "
                + ", ".join(str(path) for path in missing_runtime_paths)
            )
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            str(path) for path in runtime_library_paths
        )
        env["OMP_NUM_THREADS"] = str(args.omp_threads)
        env["OMP_PROC_BIND"] = "close"
        env["OMP_PLACES"] = "cores"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["MEEP_PRECISION"] = "native"
        env["MEEP_ACCELERATOR_RUNTIME"] = args.accelerator
        mpiexec = args.mpiexec.absolute()
        python = args.python.absolute()
        selectors = [
            item.strip() for item in args.visible_devices.split(",") if item.strip()
        ]
        if args.backend != "cpu":
            env = mpi_runner.launch_environment(
                env, args.route, "off", "eager", "native", args.accelerator
            )
            env.pop("CUDA_VISIBLE_DEVICES", None)
            env.pop("HIP_VISIBLE_DEVICES", None)
            env.pop("ROCR_VISIBLE_DEVICES", None)
            env[
                (
                    "ROCR_VISIBLE_DEVICES"
                    if args.accelerator == "hip"
                    else "CUDA_VISIBLE_DEVICES"
                )
            ] = ",".join(selectors)
        else:
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "HIP_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES",
                "MEEP_GPU_AWARE_MPI",
                "OMPI_MCA_opal_cuda_support",
                "OMPI_MCA_pml",
                "UCX_TLS",
            ):
                env.pop(name, None)

        command = [str(mpiexec), "-np", str(args.ranks), "--map-by", args.map_by]
        if args.rank_by:
            command += ["--rank-by", args.rank_by]
        if args.bind_to != "map-by":
            command += ["--bind-to", args.bind_to]
        command += [
            str(python),
            str(pathlib.Path(__file__).resolve()),
            "--worker",
            "--output",
            str(args.output.resolve()),
            "--backend",
            args.backend,
            "--accelerator",
            args.accelerator,
            "--route",
            args.route,
            "--ranks",
            str(args.ranks),
            "--visible-devices",
            ",".join(selectors),
            "--omp-threads",
            str(args.omp_threads),
            "--python",
            str(python),
            "--mpiexec",
            str(mpiexec),
            "--build-directory",
            str(args.build_directory.resolve()),
            "--library-path",
            str(args.library_path.resolve()),
        ]
        for runtime_library_path in runtime_library_paths:
            command += ["--runtime-library-path", str(runtime_library_path)]
        if args.rocm_smi:
            command += ["--rocm-smi", str(args.rocm_smi.resolve())]
        if args.toolkit_compiler:
            command += ["--toolkit-compiler", str(args.toolkit_compiler.resolve())]
        if args.ucx_info:
            command += ["--ucx-info", str(args.ucx_info.resolve())]

        environment_names = [
            "OMP_NUM_THREADS",
            "OMP_PROC_BIND",
            "OMP_PLACES",
            "OPENBLAS_NUM_THREADS",
            "MEEP_PRECISION",
            "MEEP_ACCELERATOR_RUNTIME",
            "PYTHONPATH",
            "LD_LIBRARY_PATH",
        ]
        if args.backend != "cpu":
            environment_names += [
                "MEEP_GPU_AWARE_MPI",
                "MEEP_NVIDIA_MPI_OVERLAP",
                "MEEP_NVIDIA_GRAPH_MODE",
                "OMPI_MCA_pml",
                "UCX_TLS",
                (
                    "ROCR_VISIBLE_DEVICES"
                    if args.accelerator == "hip"
                    else "CUDA_VISIBLE_DEVICES"
                ),
            ]
            if args.accelerator == "cuda":
                environment_names.append("OMPI_MCA_opal_cuda_support")
        launch = {
            "argv": command,
            "cwd": str(pathlib.Path.cwd()),
            "environment": {name: env[name] for name in environment_names},
            "timeout_seconds": args.timeout,
            "mpiexec": str(mpiexec),
            "python": str(python),
            "backend": args.backend,
            "accelerator": args.accelerator,
            "route": args.route,
            "ranks": args.ranks,
            "omp_threads": args.omp_threads,
            "visible_devices": selectors,
            "map_by": args.map_by,
            "rank_by": args.rank_by,
            "bind_to": args.bind_to,
            "mpi_version": _required_output([str(mpiexec), "--version"], env),
            "ucx_version": _required_output(
                (
                    [str(args.ucx_info.resolve()), "-v"]
                    if args.ucx_info
                    else ["ucx_info", "-v"]
                ),
                env,
            ),
        }
        source = pathlib.Path(__file__).resolve().parents[3]
        provenance = {
            "source": str(source),
            **benchmark_source_provenance(source),
            "build_directory": str(args.build_directory.resolve()),
            **mpi_runner.configured_build_provenance(
                args.build_directory.resolve(),
                mpiexec,
                python,
                args.accelerator,
                args.toolkit_compiler,
                args.rocm_smi if args.accelerator == "hip" else None,
            ),
        }
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=args.output.resolve().parent,
            prefix=".fixed-launch.",
        ) as launch_file, tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=args.output.resolve().parent,
            prefix=".fixed-provenance.",
        ) as provenance_file:
            json.dump(launch, launch_file, sort_keys=True)
            launch_file.flush()
            os.fsync(launch_file.fileno())
            json.dump(provenance, provenance_file, sort_keys=True)
            provenance_file.flush()
            os.fsync(provenance_file.fileno())
            command += [
                "--launch-record-file",
                launch_file.name,
                "--provenance-record-file",
                provenance_file.name,
            ]
            # The worker hashes the exact final argv, so refresh the launch record.
            launch["argv"] = command
            launch_file.seek(0)
            launch_file.truncate()
            json.dump(launch, launch_file, sort_keys=True)
            launch_file.flush()
            os.fsync(launch_file.fileno())
            subprocess.run(command, env=env, timeout=args.timeout, check=True)
        result = _load_result(args.output, allow_unbound_references=True)
        if args.reference:
            _compare_reference(result, args.reference)
        if args.peer_route_result:
            _compare_peer(result, args.peer_route_result)
        validate_result(result)
        _atomic(result, args.output)
        return 0
    except Exception as error:
        print(f"run_self_contained_benchmark.py: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
