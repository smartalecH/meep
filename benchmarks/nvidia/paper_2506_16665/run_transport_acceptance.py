#!/usr/bin/env python3
"""Run a small built-in CPU/GPU transport end-to-end acceptance suite."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import tempfile
from typing import Any, Mapping

import benchmark_manifest as bm
import run_mpi_benchmark as mpi_runner

CASE = {
    "name": "builtin_d1_transport_acceptance",
    "cell_z": 4.0,
    "dimensions": 1,
    "resolution": 10,
    "pml": 0.5,
    "medium": "vacuum",
    "component": "Ex",
    "source_z": 0.0,
    "frequency": 0.35,
    "fwidth": 0.14,
    "amplitude": 0.37,
    "decay_dt": 2.0,
    "decay_by": 1e-4,
    "max_steps": 2000,
    "sample_z": [-0.5, -0.25, 0.25, 0.5],
}
SCHEMA = pathlib.Path(__file__).with_name("transport_acceptance.schema.json")
COMPARISON_SCHEMA = pathlib.Path(__file__).with_name(
    "transport_acceptance_comparison.schema.json"
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def provider_zero_copy_evidence(path: pathlib.Path, memory_type: str) -> dict[str, Any]:
    if memory_type not in {"cuda", "rocm"}:
        raise mpi_runner.RunnerError("provider memory type must be cuda or rocm")
    resolved = path.resolve()
    try:
        payload = resolved.read_bytes()
    except OSError as error:
        raise mpi_runner.RunnerError(
            f"cannot read provider zero-copy log {resolved}: {error}"
        ) from error
    text = payload.decode("utf-8", errors="replace")
    transport = f"{memory_type}_ipc"
    device_protocol = re.compile(
        rf"from {memory_type}/GPU([0-9]+)"
        rf"(?:(?!from {memory_type}/GPU).)*?"
        rf"zero-copy[^\n]*{transport}/{transport}",
        re.DOTALL,
    )
    devices = set(device_protocol.findall(text))
    if len(devices) < 2:
        raise mpi_runner.RunnerError(
            f"provider log does not prove two-device {transport} zero-copy"
        )
    return {
        "provider": "ucx",
        "memory_type": memory_type,
        "transport": transport,
        "evidence": {
            "path": str(resolved),
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
    }


def _cpu_placement() -> tuple[list[int], list[int]]:
    affinity = sorted(os.sched_getaffinity(0))
    nodes: set[int] = set()
    for cpu in affinity:
        for node in pathlib.Path(f"/sys/devices/system/cpu/cpu{cpu}").glob(
            "node[0-9]*"
        ):
            nodes.add(int(node.name[4:]))
    return affinity, sorted(nodes)


def _normalize_pci_bus_id(value: str) -> str:
    normalized = value.strip().lower()
    if re.fullmatch(r"[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", normalized):
        normalized = "0000:" + normalized
    if re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", normalized) is None:
        raise mpi_runner.RunnerError(f"invalid PCI BDF {value!r}")
    return normalized


def _hip_pci_bus_id(logical_device_id: int) -> str:
    library = os.environ.get("MEEP_HIP_RUNTIME_LIBRARY", "libamdhip64.so")
    try:
        hip = ctypes.CDLL(library)
    except OSError as error:
        raise mpi_runner.RunnerError(
            f"cannot load HIP runtime {library}: {error}"
        ) from error
    get_bus_id = hip.hipDeviceGetPCIBusId
    get_bus_id.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    get_bus_id.restype = ctypes.c_int
    value = ctypes.create_string_buffer(32)
    status = get_bus_id(value, len(value), logical_device_id)
    if status != 0:
        raise mpi_runner.RunnerError(
            f"hipDeviceGetPCIBusId({logical_device_id}) failed with status {status}"
        )
    try:
        return _normalize_pci_bus_id(value.value.decode("ascii"))
    except UnicodeDecodeError as error:
        raise mpi_runner.RunnerError("HIP returned a non-ASCII PCI BDF") from error


def _device_provenance(runtime: Mapping[str, Any]) -> dict[str, Any]:
    local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0"))
    selectors = [
        item.strip()
        for item in os.environ.get("MEEP_ACCEPTANCE_VISIBLE_DEVICES", "").split(",")
        if item.strip()
    ]
    if local_rank >= len(selectors):
        raise mpi_runner.RunnerError(
            "GPU visibility selectors do not cover every local rank"
        )
    selector = selectors[local_rank]
    rocm_smi = os.environ.get("MEEP_ROCM_SMI")
    if rocm_smi:
        runtime_bus_id = _hip_pci_bus_id(int(runtime["device_id"]))
        completed = subprocess.run(
            [
                rocm_smi,
                "--showuniqueid",
                "--showbus",
                "--showproductname",
                "--showtoponuma",
                "--json",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        inventory = json.loads(completed.stdout)
        matches = [
            (name, entry)
            for name, entry in inventory.items()
            if isinstance(entry, dict)
            and _normalize_pci_bus_id(str(entry.get("PCI Bus", ""))) == runtime_bus_id
        ]
        if len(matches) != 1:
            raise mpi_runner.RunnerError(
                f"HIP PCI BDF {runtime_bus_id} matched {len(matches)} rocm-smi devices"
            )
        physical_name, entry = matches[0]
        if not physical_name.startswith("card") or not physical_name[4:].isdigit():
            raise mpi_runner.RunnerError(
                f"rocm-smi returned invalid physical selector {physical_name!r}"
            )
        return {
            "logical_device_id": int(runtime["device_id"]),
            "visibility_selector": selector,
            "physical_selector": physical_name[4:],
            "runtime_uuid": str(runtime["device_uuid"]),
            "physical_unique_id": str(entry.get("Unique ID", "")),
            "pci_bus_id": runtime_bus_id,
            "numa_node": int(entry.get("(Topology) Numa Node", -1)),
            "name": str(entry.get("Card Series", "")),
            "architecture": str(entry.get("GFX Version", "")),
        }
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={runtime['device_uuid']}",
            "--query-gpu=index,uuid,pci.bus_id,name",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    fields = [item.strip() for item in completed.stdout.strip().split(",", 3)]
    if len(fields) != 4:
        raise mpi_runner.RunnerError("nvidia-smi returned malformed device provenance")
    pci_bus_id = _normalize_pci_bus_id(fields[2])
    numa_path = pathlib.Path("/sys/bus/pci/devices") / pci_bus_id / "numa_node"
    numa_node = int(numa_path.read_text(encoding="utf-8").strip())
    return {
        "logical_device_id": int(runtime["device_id"]),
        "visibility_selector": selector,
        "physical_selector": fields[0],
        "runtime_uuid": str(runtime["device_uuid"]),
        "physical_unique_id": fields[1],
        "pci_bus_id": pci_bus_id,
        "numa_node": numa_node,
        "name": fields[3],
        "architecture": "",
    }


def validate_acceptance_artifact(
    value: Mapping[str, Any],
    expected_route: str | None = None,
    expected_visibility_selectors: list[str] | None = None,
) -> None:
    schema = bm.load_json_object(SCHEMA, "transport acceptance schema")
    bm._validate_schema_structure(value, schema, schema, "transport acceptance")
    route = value["route"]
    if expected_route is not None and route != expected_route:
        raise mpi_runner.RunnerError(
            f"expected {expected_route} acceptance artifact, got {route}"
        )
    if set(value["case"]) != set(CASE) or value["case"] != CASE:
        raise mpi_runner.RunnerError(
            "acceptance case does not exactly match the built-in case"
        )
    if value["case_sha256"] != canonical_hash(CASE):
        raise mpi_runner.RunnerError(
            "acceptance case hash does not match the built-in case"
        )

    expected_ranks = 1 if route == "cpu" else 2
    records = value["rank_records"]
    if len(records) != expected_ranks or [record["rank"] for record in records] != list(
        range(expected_ranks)
    ):
        raise mpi_runner.RunnerError(
            f"{route} acceptance requires {expected_ranks} ordered rank record(s)"
        )
    required_runtime = {
        "resolved_backend",
        "requested_transport",
        "resolved_transport",
        "captured_requested_transport",
        "mpi_provider",
        "mpi_query_available",
        "mpi_cuda_aware",
        "communicator_rank",
        "communicator_size",
        "device_owner",
        "captured_transport_epoch_active",
        "captured_transport_epoch_fresh",
        "transport_pinned_bytes",
        "device_id",
        "device_uuid",
        "host_fallback_count",
        "host_fallback_device_to_host_bytes",
        "host_fallback_host_to_device_bytes",
        "host_fallback_steady_capacity_growths",
        "material_fallback_warning_count",
    }
    expected_transport = "none" if route == "cpu" else route
    gpu_uuids: set[str] = set()
    physical_selectors: set[str] = set()
    physical_ids: set[str] = set()
    pci_bus_ids: set[str] = set()
    observed_selectors: list[str] = []
    for index, record in enumerate(records):
        runtime = record["runtime"]
        missing_runtime = required_runtime - set(runtime)
        if missing_runtime:
            raise mpi_runner.RunnerError(
                f"{route} rank {index} runtime is missing {sorted(missing_runtime)}"
            )
        if not isinstance(runtime["mpi_provider"], str) or not runtime["mpi_provider"]:
            raise mpi_runner.RunnerError(
                f"{route} rank {index} MPI provider is invalid"
            )
        for name in (
            "mpi_query_available",
            "mpi_cuda_aware",
            "device_owner",
            "captured_transport_epoch_active",
            "captured_transport_epoch_fresh",
        ):
            if not isinstance(runtime[name], bool):
                raise mpi_runner.RunnerError(
                    f"{route} rank {index} runtime {name} is invalid"
                )
        for name in (
            "communicator_rank",
            "communicator_size",
            "transport_pinned_bytes",
        ):
            if (
                isinstance(runtime[name], bool)
                or not isinstance(runtime[name], int)
                or runtime[name] < 0
            ):
                raise mpi_runner.RunnerError(
                    f"{route} rank {index} runtime {name} is invalid"
                )
        if (
            runtime["communicator_rank"] != index
            or runtime["communicator_size"] != expected_ranks
        ):
            raise mpi_runner.RunnerError(
                f"{route} runtime communicator identity is inconsistent"
            )
        affinity = record["cpu_affinity"]
        cpu_numa_nodes = record["cpu_numa_nodes"]
        if (
            not affinity
            or any(
                isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0
                for cpu in affinity
            )
            or len(set(affinity)) != len(affinity)
        ):
            raise mpi_runner.RunnerError(
                f"{route} rank {index} CPU affinity is invalid"
            )
        if (
            not cpu_numa_nodes
            or any(
                isinstance(node, bool) or not isinstance(node, int) or node < 0
                for node in cpu_numa_nodes
            )
            or cpu_numa_nodes != sorted(set(cpu_numa_nodes))
        ):
            raise mpi_runner.RunnerError(
                f"{route} rank {index} CPU NUMA placement is invalid"
            )
        expected_backend = "cpu" if route == "cpu" else "nvidia"
        if (
            runtime["resolved_backend"] != expected_backend
            or runtime["resolved_transport"] != expected_transport
        ):
            raise mpi_runner.RunnerError(
                f"{route} runtime resolved route/backend is inconsistent"
            )
        if route == "cpu":
            if (
                runtime["device_owner"]
                or runtime["captured_transport_epoch_active"]
                or runtime["transport_pinned_bytes"] != 0
            ):
                raise mpi_runner.RunnerError(
                    "CPU acceptance records unexpected GPU transport state"
                )
            if record["device"] is not None:
                raise mpi_runner.RunnerError(
                    "CPU acceptance record unexpectedly has GPU provenance"
                )
        else:
            if (
                runtime["requested_transport"] != route
                or runtime["captured_requested_transport"] != route
                or not runtime["device_owner"]
                or not runtime["captured_transport_epoch_active"]
                or not runtime["captured_transport_epoch_fresh"]
            ):
                raise mpi_runner.RunnerError(
                    f"{route} runtime transport state is inconsistent"
                )
            if any(
                runtime[name]
                for name in (
                    "host_fallback_count",
                    "host_fallback_device_to_host_bytes",
                    "host_fallback_host_to_device_bytes",
                    "host_fallback_steady_capacity_growths",
                    "material_fallback_warning_count",
                )
            ):
                raise mpi_runner.RunnerError(
                    f"{route} runtime reports an unexpected host fallback"
                )
            if route == "staged" and runtime["transport_pinned_bytes"] <= 0:
                raise mpi_runner.RunnerError(
                    "staged runtime has no pinned transport storage"
                )
            if route == "direct" and (
                not runtime["mpi_query_available"]
                or not runtime["mpi_cuda_aware"]
                or runtime["transport_pinned_bytes"] != 0
            ):
                raise mpi_runner.RunnerError(
                    "direct runtime lacks positive GPU-aware MPI state"
                )
            device = record["device"]
            if not isinstance(device, dict):
                raise mpi_runner.RunnerError(
                    f"{route} rank {index} lacks GPU provenance"
                )
            required_device = {
                "logical_device_id",
                "visibility_selector",
                "physical_selector",
                "runtime_uuid",
                "physical_unique_id",
                "pci_bus_id",
                "numa_node",
                "name",
                "architecture",
            }
            if set(device) != required_device or not all(
                isinstance(device[name], str) and device[name]
                for name in (
                    "visibility_selector",
                    "physical_selector",
                    "runtime_uuid",
                    "physical_unique_id",
                    "pci_bus_id",
                    "name",
                )
            ):
                raise mpi_runner.RunnerError(
                    f"{route} rank {index} GPU provenance is incomplete"
                )
            if (
                isinstance(device["logical_device_id"], bool)
                or not isinstance(device["logical_device_id"], int)
                or device["logical_device_id"] < 0
                or isinstance(device["numa_node"], bool)
                or not isinstance(device["numa_node"], int)
                or device["numa_node"] < 0
                or _normalize_pci_bus_id(device["pci_bus_id"]) != device["pci_bus_id"]
            ):
                raise mpi_runner.RunnerError(
                    f"{route} rank {index} GPU provenance is invalid"
                )
            if (
                device["logical_device_id"] != runtime["device_id"]
                or device["runtime_uuid"] != runtime["device_uuid"]
            ):
                raise mpi_runner.RunnerError(
                    f"{route} rank {index} GPU runtime identity differs"
                )
            if cpu_numa_nodes != [device["numa_node"]]:
                raise mpi_runner.RunnerError(
                    f"{route} rank {index} CPU affinity is not exclusively local to its GPU NUMA node"
                )
            gpu_uuids.add(device["runtime_uuid"])
            physical_selectors.add(device["physical_selector"])
            physical_ids.add(device["physical_unique_id"])
            pci_bus_ids.add(device["pci_bus_id"])
            observed_selectors.append(device["visibility_selector"])

        counters = record["counter_deltas"]
        if set(counters) != set(mpi_runner.COUNTER_AGGREGATIONS):
            raise mpi_runner.RunnerError(
                f"{route} rank {index} counters do not match the required counter set"
            )
        for name, counter in counters.items():
            if isinstance(counter, bool) or not isinstance(counter, int) or counter < 0:
                raise mpi_runner.RunnerError(
                    f"{route} rank {index} counter {name} is invalid"
                )
        for name in (
            "steady_allocation_count",
            "graph_recapture_count",
            "full_field_copy_count",
            "host_fallback_count",
            "host_fallback_device_to_host_bytes",
            "host_fallback_host_to_device_bytes",
            "host_fallback_steady_capacity_growths",
            "material_fallback_warning_count",
        ):
            if counters[name] != 0:
                raise mpi_runner.RunnerError(
                    f"{route} rank {index} steady-state counter {name} is nonzero"
                )
        if route == "staged" and (
            counters["messages_sent"] <= 0
            or counters["messages_received"] <= 0
            or counters["bytes_sent"] <= 0
            or counters["bytes_received"] <= 0
            or counters["messages_sent"] != counters["messages_received"]
            or counters["bytes_sent"] != counters["bytes_received"]
            or counters["direct_bytes"] != 0
            or counters["device_to_host_calls"] <= 0
            or counters["host_to_device_calls"] <= 0
            or counters["device_to_host_bytes"] != counters["bytes_sent"]
            or counters["host_to_device_bytes"] != counters["bytes_received"]
            or counters["slot_reuses"] <= 0
        ):
            raise mpi_runner.RunnerError(
                f"staged rank {index} counters do not prove host-staged transport"
            )
        if route == "direct" and (
            counters["messages_sent"] <= 0
            or counters["messages_received"] <= 0
            or counters["bytes_sent"] <= 0
            or counters["bytes_received"] <= 0
            or counters["direct_bytes"]
            != counters["bytes_sent"] + counters["bytes_received"]
            or counters["direct_bytes"] <= 0
            or any(
                counters[name]
                for name in (
                    "device_to_host_calls",
                    "device_to_host_bytes",
                    "host_to_device_calls",
                    "host_to_device_bytes",
                )
            )
        ):
            raise mpi_runner.RunnerError(
                f"direct rank {index} counters do not prove GPU-aware MPI transport"
            )
        samples = record["observables"].get("samples")
        if not isinstance(samples, list) or len(samples) != len(CASE["sample_z"]):
            raise mpi_runner.RunnerError(
                f"{route} rank {index} does not contain the fixed observable samples"
            )
        for sample in samples:
            if (
                not isinstance(sample, list)
                or len(sample) != 2
                or any(
                    isinstance(component, bool)
                    or not isinstance(component, (int, float))
                    or not math.isfinite(component)
                    for component in sample
                )
            ):
                raise mpi_runner.RunnerError(
                    f"{route} rank {index} contains an invalid observable sample"
                )
    if route != "cpu":
        if not (
            len(gpu_uuids)
            == len(physical_selectors)
            == len(physical_ids)
            == len(pci_bus_ids)
            == expected_ranks
        ):
            raise mpi_runner.RunnerError(
                f"{route} ranks do not own unique physical GPUs"
            )
        if (
            expected_visibility_selectors is not None
            and observed_selectors != expected_visibility_selectors
        ):
            raise mpi_runner.RunnerError(
                f"{route} GPU visibility selectors {observed_selectors} do not match "
                f"{expected_visibility_selectors}"
            )


def load_acceptance_artifact(
    path: pathlib.Path,
    expected_route: str,
    expected_visibility_selectors: list[str] | None = None,
) -> dict[str, Any]:
    value = bm.load_json_object(path, f"{expected_route} acceptance")
    validate_acceptance_artifact(value, expected_route, expected_visibility_selectors)
    return value


def validate_comparison_artifact(value: Mapping[str, Any]) -> None:
    schema = bm.load_json_object(
        COMPARISON_SCHEMA, "transport acceptance comparison schema"
    )
    bm._validate_schema_structure(
        value, schema, schema, "transport acceptance comparison"
    )
    direct_fields = {"direct", "staged_direct_bitwise", "device_buffer_mpi_positive"}
    present = direct_fields.intersection(value)
    if present and present != direct_fields:
        raise mpi_runner.RunnerError(
            "direct device-buffer evidence must be entirely present or absent"
        )
    if "provider_zero_copy" in value and present != direct_fields:
        raise mpi_runner.RunnerError(
            "provider zero-copy evidence requires complete device-buffer MPI evidence"
        )
    provider = value.get("provider_zero_copy")
    if provider is not None:
        expected_transport = {"cuda": "cuda_ipc", "rocm": "rocm_ipc"}
        if provider["transport"] != expected_transport[provider["memory_type"]]:
            raise mpi_runner.RunnerError(
                "provider zero-copy memory type and transport are inconsistent"
            )


def _steady_measurement_start(mp: Any, sim: Any) -> tuple[dict[str, Any], int]:
    """Warm resident execution once, fence all ranks, then snapshot counters."""
    sim.fields.advance(1)
    mp.all_wait()
    return sim.get_execution_runtime_report(), int(sim.fields.t)


def _steady_measurement_end(mp: Any, sim: Any) -> dict[str, Any]:
    """Fence timed work before reading counters; monitor queries follow this."""
    mp.all_wait()
    return sim.get_execution_runtime_report()


def _atomic(
    value: Mapping[str, Any], path: pathlib.Path, schema_path: pathlib.Path = SCHEMA
) -> None:
    schema = bm.load_json_object(schema_path, "transport acceptance schema")
    bm._validate_schema_structure(value, schema, schema, "transport acceptance")
    if schema_path == SCHEMA:
        validate_acceptance_artifact(value)
    elif schema_path == COMPARISON_SCHEMA:
        validate_comparison_artifact(value)
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
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _worker(route: str, output: pathlib.Path) -> int:
    mp = importlib.import_module("meep")
    mp.verbosity(0)
    if route != "cpu":
        mp.begin_global_communications()
        mp.divide_parallel_processes(1)
    backend = "cpu" if route == "cpu" else "nvidia"
    kwargs = {
        "cell_size": mp.Vector3(0, 0, CASE["cell_z"]),
        "dimensions": CASE["dimensions"],
        "resolution": CASE["resolution"],
        "boundary_layers": [mp.PML(CASE["pml"])],
        "geometry": [],
        "sources": [
            mp.Source(
                mp.GaussianSource(CASE["frequency"], fwidth=CASE["fwidth"]),
                component=mp.Ex,
                center=mp.Vector3(z=CASE["source_z"]),
                amplitude=CASE["amplitude"],
            )
        ],
        "backend": backend,
        "accelerator_strict": route != "cpu",
    }
    if route != "cpu":
        kwargs["device_id"] = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    sim = mp.Simulation(**kwargs)
    sim.init_sim()
    # Force resident compilation/allocation before the measurement baseline.
    # The same deterministic step is applied to CPU, staged, and direct runs.
    before, start_step = _steady_measurement_start(mp, sim)
    decay = mp.stop_when_fields_decayed(
        CASE["decay_dt"], mp.Ex, mp.Vector3(z=0.5), CASE["decay_by"]
    )
    sim.run(until_after_sources=decay)
    after = _steady_measurement_end(mp, sim)
    steps = int(sim.fields.t) - start_step
    if steps > CASE["max_steps"]:
        raise mpi_runner.RunnerError("acceptance decay exceeded its declared max_steps")
    reason = "field_energy_decay"
    observables = {
        "samples": [
            [
                float(complex(sim.get_field_point(mp.Ex, mp.Vector3(z=z))).real),
                float(complex(sim.get_field_point(mp.Ex, mp.Vector3(z=z))).imag),
            ]
            for z in CASE["sample_z"]
        ]
    }
    rank = int(after["communicator_rank"])
    cpu_affinity, cpu_numa_nodes = _cpu_placement()
    local = {
        "rank": rank,
        "runtime": after,
        "steps": steps,
        "stop_reason": reason,
        "observables": observables,
        "device": None if route == "cpu" else _device_provenance(after),
        "cpu_affinity": cpu_affinity,
        "cpu_numa_nodes": cpu_numa_nodes,
        "counter_deltas": {
            name: int(after[name]) - int(before[name])
            for name in mpi_runner.COUNTER_AGGREGATIONS
            if name in after
        },
    }
    if route == "cpu":
        gathered = [local]
    else:
        gathered = [
            json.loads(item)
            for item in mp.active_communicator_allgather_json(
                json.dumps(local, sort_keys=True)
            )
        ]
    if rank == 0:
        source = pathlib.Path(__file__).resolve().parents[3]
        result = {
            "schema_version": 2,
            "kind": "meep_builtin_transport_acceptance",
            "case": CASE,
            "case_sha256": canonical_hash(CASE),
            "route": route,
            "build_identity": {
                "source": str(source),
                "commit": mpi_runner.legacy._git(source, "rev-parse", "HEAD"),
                "dirty": bool(
                    mpi_runner.legacy._git(source, "status", "--porcelain=v1")
                ),
                "python": sys.executable,
                "meep_module": str(mp.__file__),
                "meep_extension": str(
                    getattr(getattr(mp, "_meep", None), "__file__", "")
                ),
            },
            "rank_records": gathered,
        }
        _atomic(result, output)
    if route != "cpu":
        mp.end_divide_parallel()
    return 0


def _run(command, env, timeout):
    subprocess.run(command, env=env, timeout=timeout, check=True)


def _routes(value: str) -> list[str]:
    routes = [item.strip() for item in value.split(",") if item.strip()]
    if len(set(routes)) != len(routes) or any(
        route not in {"cpu", "staged", "direct"} for route in routes
    ):
        raise argparse.ArgumentTypeError(
            "routes must be a unique comma list of cpu,staged,direct"
        )
    if "cpu" not in routes or "staged" not in routes:
        raise argparse.ArgumentTypeError("routes must include cpu and staged")
    return routes


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path)
    parser.add_argument(
        "--prefix", type=pathlib.Path, default=mpi_runner.DEFAULT_PREFIX
    )
    parser.add_argument(
        "--python",
        type=pathlib.Path,
        help="branch-matched Python executable (defaults to PREFIX/bin/python)",
    )
    parser.add_argument(
        "--mpiexec",
        type=pathlib.Path,
        help="MPI launcher (defaults to PREFIX/bin/mpirun)",
    )
    parser.add_argument(
        "--routes",
        type=_routes,
        default=_routes("cpu,staged,direct"),
        help="comma list; use cpu,staged for the staged-only AMD gate",
    )
    parser.add_argument(
        "--visible-devices",
        default="0,1",
        help="two comma-separated ROCr/CUDA visibility selectors",
    )
    parser.add_argument(
        "--rocm-smi",
        type=pathlib.Path,
        help="rocm-smi used to authenticate physical GPU/BDF/NUMA identity",
    )
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--pythonpath")
    parser.add_argument("--library-path")
    parser.add_argument(
        "--provider-zero-copy-log",
        type=pathlib.Path,
        help="UCX protocol log from a separate fail-closed device-buffer MPI probe",
    )
    parser.add_argument(
        "--worker", choices=("cpu", "staged", "direct"), help=argparse.SUPPRESS
    )
    parser.add_argument("--output", type=pathlib.Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.worker:
            return _worker(args.worker, args.output)
        if args.output_dir is None:
            raise mpi_runner.RunnerError("--output-dir is required")
        if not args.pythonpath or not args.library_path:
            raise mpi_runner.RunnerError("--pythonpath and --library-path are required")
        if args.provider_zero_copy_log is not None and "direct" not in args.routes:
            raise mpi_runner.RunnerError(
                "--provider-zero-copy-log requires the direct acceptance route"
            )
        if (
            args.rocm_smi is not None
            and "direct" in args.routes
            and args.provider_zero_copy_log is None
        ):
            raise mpi_runner.RunnerError(
                "ROCm direct acceptance requires --provider-zero-copy-log"
            )
        prefix = args.prefix.resolve()
        python = args.python.resolve() if args.python else prefix / "bin/python"
        mpiexec = args.mpiexec.resolve() if args.mpiexec else prefix / "bin/mpirun"
        selectors = [
            item.strip() for item in args.visible_devices.split(",") if item.strip()
        ]
        if len(selectors) != 2 or len(set(selectors)) != 2:
            raise mpi_runner.RunnerError(
                "--visible-devices requires two distinct selectors"
            )
        if (
            "staged" in args.routes
            and args.rocm_smi is None
            and "direct" not in args.routes
        ):
            raise mpi_runner.RunnerError(
                "staged-only AMD acceptance requires --rocm-smi"
            )
        script = pathlib.Path(__file__).resolve()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        base = dict(os.environ)
        base["PYTHONPATH"] = str(pathlib.Path(args.pythonpath).resolve())
        base["LD_LIBRARY_PATH"] = (
            str(pathlib.Path(args.library_path).resolve()) + ":" + str(prefix / "lib")
        )
        outputs = {route: args.output_dir / f"{route}.json" for route in args.routes}
        _run(
            [
                str(python),
                str(script),
                "--worker",
                "cpu",
                "--output",
                str(outputs["cpu"]),
            ],
            base,
            args.timeout,
        )
        for route in args.routes:
            if route == "cpu":
                continue
            env = mpi_runner.launch_environment(base, route, "off", "eager", "native")
            env["MEEP_ACCEPTANCE_VISIBLE_DEVICES"] = ",".join(selectors)
            if args.rocm_smi:
                env["ROCR_VISIBLE_DEVICES"] = ",".join(selectors)
                env["MEEP_ROCM_SMI"] = str(args.rocm_smi.resolve())
                env["MEEP_GPU_AWARE_MPI"] = "no" if route == "staged" else "yes"
                env.pop("CUDA_VISIBLE_DEVICES", None)
                env.pop("HIP_VISIBLE_DEVICES", None)
                env.pop("OMPI_MCA_opal_cuda_support", None)
                if route == "direct":
                    env["OMPI_MCA_pml"] = "ucx"
                    env["UCX_TLS"] = "self,sm,rocm_copy,rocm_ipc"
                else:
                    env.pop("OMPI_MCA_pml", None)
                    env.pop("UCX_TLS", None)
            else:
                env["CUDA_VISIBLE_DEVICES"] = ",".join(selectors)
            _run(
                [
                    str(mpiexec),
                    "-np",
                    "2",
                    "--map-by",
                    "ppr:1:package:PE=2",
                    "--bind-to",
                    "core",
                    str(python),
                    str(script),
                    "--worker",
                    route,
                    "--output",
                    str(outputs[route]),
                ],
                env,
                args.timeout,
            )
        results = {
            route: load_acceptance_artifact(
                outputs[route], route, selectors if route != "cpu" else None
            )
            for route in args.routes
        }
        cpu_result = results["cpu"]
        if any(
            result["build_identity"] != cpu_result["build_identity"]
            for result in results.values()
        ):
            raise mpi_runner.RunnerError(
                "acceptance routes did not use the same source and module build"
            )
        baseline = cpu_result["rank_records"][0]["observables"]
        for route, candidate in results.items():
            if route == "cpu":
                continue
            for record in candidate["rank_records"]:
                for actual, expected in zip(
                    record["observables"]["samples"], baseline["samples"]
                ):
                    if abs(complex(*actual) - complex(*expected)) > 1e-11 + 1e-8 * abs(
                        complex(*expected)
                    ):
                        raise mpi_runner.RunnerError(
                            "GPU field sample failed CPU tolerance"
                        )
        staged_result = results["staged"]
        direct_result = results.get("direct")
        if direct_result is not None and [
            r["observables"] for r in staged_result["rank_records"]
        ] != [r["observables"] for r in direct_result["rank_records"]]:
            raise mpi_runner.RunnerError(
                "staged/direct acceptance observables are not bitwise identical"
            )
        for record in staged_result["rank_records"]:
            runtime, counters = record["runtime"], record["counter_deltas"]
            if (
                runtime["resolved_transport"] != "staged"
                or counters["direct_bytes"] != 0
                or counters["device_to_host_bytes"] != counters["bytes_sent"]
                or counters["host_to_device_bytes"] != counters["bytes_received"]
                or counters["device_to_host_calls"] <= 0
                or counters["host_to_device_calls"] <= 0
                or runtime["transport_pinned_bytes"] <= 0
            ):
                raise mpi_runner.RunnerError(
                    "staged acceptance did not prove host-staged halo traffic on every rank"
                )
        if direct_result is not None:
            for record in direct_result["rank_records"]:
                runtime, counters = record["runtime"], record["counter_deltas"]
                if (
                    not runtime["mpi_query_available"]
                    or not runtime["mpi_cuda_aware"]
                    or runtime["resolved_transport"] != "direct"
                    or counters["direct_bytes"] <= 0
                    or counters["direct_bytes"]
                    != counters["bytes_sent"] + counters["bytes_received"]
                    or any(
                        counters[name]
                        for name in (
                            "device_to_host_calls",
                            "device_to_host_bytes",
                            "host_to_device_calls",
                            "host_to_device_bytes",
                        )
                    )
                    or runtime["transport_pinned_bytes"] != 0
                ):
                    raise mpi_runner.RunnerError(
                        "direct acceptance did not prove GPU-aware MPI traffic on every rank"
                    )
        comparison = {
            "schema_version": 2,
            "kind": "meep_builtin_transport_acceptance_comparison",
            "case_sha256": canonical_hash(CASE),
            "cpu": {
                "path": str(outputs["cpu"].resolve()),
                "sha256": bm.sha256_file(outputs["cpu"]),
            },
            "staged": {
                "path": str(outputs["staged"].resolve()),
                "sha256": bm.sha256_file(outputs["staged"]),
            },
            "cpu_tolerance_passed": True,
        }
        if direct_result is not None:
            comparison.update(
                {
                    "direct": {
                        "path": str(outputs["direct"].resolve()),
                        "sha256": bm.sha256_file(outputs["direct"]),
                    },
                    "staged_direct_bitwise": True,
                    "device_buffer_mpi_positive": True,
                }
            )
        if args.provider_zero_copy_log is not None:
            comparison["provider_zero_copy"] = provider_zero_copy_evidence(
                args.provider_zero_copy_log, "rocm" if args.rocm_smi else "cuda"
            )
        _atomic(comparison, args.output_dir / "comparison.json", COMPARISON_SCHEMA)
        return 0
    except Exception as error:
        print(f"run_transport_acceptance.py: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
