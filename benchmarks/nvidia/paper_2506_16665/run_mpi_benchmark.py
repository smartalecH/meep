#!/usr/bin/env python3
"""Run and reconcile the normative multi-rank NVIDIA benchmark artifact.

The outer process configures Open MPI before MPI_Init and launches this file in
``--worker`` mode.  Workers exchange JSON records through Meep's active
communicator; stdout is deliberately not an evidence channel.
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
import run_benchmark as legacy


HERE = pathlib.Path(__file__).resolve().parent
SCHEMA = HERE / "mpi_benchmark_result.schema.json"
DEFAULT_PREFIX = pathlib.Path("/home/alechammond/meep-env")
MPIEXEC = DEFAULT_PREFIX / "bin/mpirun"
PYTHON = DEFAULT_PREFIX / "bin/python"
UCX_TLS = "self,sm,cuda_copy,cuda_ipc"
COUNTER_AGGREGATIONS = {
    "messages_sent": "sum", "messages_received": "sum",
    "bytes_sent": "sum", "bytes_received": "sum",
    "device_to_host_calls": "sum", "device_to_host_bytes": "sum",
    "host_to_device_calls": "sum", "host_to_device_bytes": "sum",
    "direct_bytes": "sum", "testsome_polls": "sum", "waitall_calls": "sum",
    "slot_reuses": "sum", "overlap_stages": "sum",
    "overlap_interior_launches": "sum", "overlap_boundary_launches": "sum",
    "graph_capture_count": "sum", "graph_launch_count": "sum",
    "graph_boundary_count": "sum", "host_fallback_count": "sum",
    "host_fallback_device_to_host_bytes": "sum",
    "host_fallback_host_to_device_bytes": "sum",
    "material_fallback_warning_count": "sum",
    "gather_launches": "sum", "scatter_launches": "sum",
    "request_completions": "sum", "host_fallback_steady_capacity_growths": "sum",
    "material_recipe_prepare_nanoseconds": "sum",
    "material_initialize_nanoseconds": "sum", "graph_build_nanoseconds": "sum",
    "gather_pack_nanoseconds": "sum", "device_to_host_nanoseconds": "sum",
    "mpi_progress_nanoseconds": "sum", "mpi_wait_nanoseconds": "sum",
    "host_to_device_nanoseconds": "sum", "scatter_unpack_nanoseconds": "sum",
    "steady_allocation_count": "sum", "graph_recapture_count": "sum",
    "full_field_copy_count": "sum",
}
GAUGE_AGGREGATIONS = {
    "transport_device_bytes": "max", "transport_pinned_bytes": "max",
    "process_device_bytes_current": "max", "process_device_bytes_peak": "max",
    "process_pinned_bytes_current": "max", "process_pinned_bytes_peak": "max",
    "executable_build_count": "max",
    "high_water_requests": "max",
}


class RunnerError(RuntimeError):
    pass


def manifest_snapshot(path: pathlib.Path) -> tuple[dict[str, Any], str]:
    payload = path.resolve().read_bytes()
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise RunnerError("manifest must be a JSON object")
    return value, hashlib.sha256(payload).hexdigest()


def configured_build_provenance(build_directory: pathlib.Path,
                                mpiexec: pathlib.Path,
                                python: pathlib.Path) -> dict[str, Any]:
    makefile = build_directory / "Makefile"
    config_status = build_directory / "config.status"
    if not makefile.is_file() or not config_status.is_file():
        raise RunnerError("build directory must contain Makefile and config.status")
    flag_names = ("CXX", "CXXFLAGS", "CPPFLAGS", "NVCCFLAGS", "CUDA_CPPFLAGS",
                  "CUDA_HOST_FLAGS", "CUDA_ARCH_FLAGS")
    flags = {name: "" for name in flag_names}
    for line in makefile.read_text(encoding="utf-8", errors="replace").splitlines():
        name, separator, value = line.partition("=")
        name = name.strip()
        if separator and name in flags:
            flags[name] = value.strip()
    command_environment = dict(os.environ)
    command_environment["PATH"] = str(mpiexec.parent) + os.pathsep + command_environment.get(
        "PATH", ""
    )

    def required_output(command: Sequence[str], label: str) -> str:
        try:
            return subprocess.run(
                list(command), check=True, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, env=command_environment,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise RunnerError(f"cannot record {label}: {error}") from error

    configure_arguments = required_output([str(config_status), "--config"],
                                          "configure arguments")
    compiler = pathlib.Path(flags["CXX"].split()[0]) if flags["CXX"] else pathlib.Path("mpic++")
    if not compiler.is_absolute():
        compiler = mpiexec.parent / compiler
    compiler_version = required_output([str(compiler), "--version"], "compiler version")
    if not configure_arguments or not compiler_version:
        raise RunnerError("configured build provenance commands are unavailable")
    return {
        "configure_arguments": configure_arguments,
        "compiler": compiler_version,
        "compiler_flags": flags,
        "executable_sha256": {
            "mpiexec": bm.sha256_file(mpiexec),
            "python": bm.sha256_file(python),
        },
    }


def observed_cpu_affinity() -> list[int]:
    if not hasattr(os, "sched_getaffinity"):
        raise RunnerError("this platform cannot report observed CPU affinity")
    affinity = sorted(os.sched_getaffinity(0))
    if not affinity:
        raise RunnerError("observed CPU affinity is empty")
    return affinity


def python_purelib(python: pathlib.Path) -> pathlib.Path:
    environment = dict(os.environ)
    environment["PATH"] = str(python.parent) + os.pathsep + environment.get("PATH", "")
    try:
        output = subprocess.run(
            [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
            check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=environment,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RunnerError(f"cannot query the selected Python installation: {error}") from error
    if not output:
        raise RunnerError("selected Python reported an empty purelib path")
    return pathlib.Path(output).resolve()


def normalize_gpu_uuid(value: Any) -> str:
    if not isinstance(value, str):
        raise RunnerError("GPU UUID must be a string")
    normalized = value.strip().lower()
    if normalized.startswith("gpu-"):
        normalized = normalized[4:]
    normalized = normalized.replace("-", "")
    if len(normalized) != 32 or any(character not in "0123456789abcdef"
                                    for character in normalized):
        raise RunnerError(f"GPU UUID is not canonicalizable: {value!r}")
    return normalized


def launch_environment(base: Mapping[str, str], route: str, overlap: str,
                       graph: str, precision: str) -> dict[str, str]:
    if route not in {"staged", "direct", "auto"}:
        raise RunnerError("route must be staged, direct, or auto")
    env = dict(base)
    env.update({
        "OMPI_MCA_opal_cuda_support": "true",
        "OMPI_MCA_pml": "ucx",
        "UCX_TLS": UCX_TLS,
        "MEEP_GPU_AWARE_MPI": {"staged": "no", "direct": "yes", "auto": "auto"}[route],
        "MEEP_NVIDIA_MPI_OVERLAP": overlap,
        "MEEP_NVIDIA_GRAPH_MODE": graph,
        "MEEP_PRECISION": precision,
    })
    env.pop("OMPI_MCA_btl", None)
    return env


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RunnerError(f"{label} must be a finite number")
    return float(value)


def _linear_at(xs: Sequence[float], ys: Sequence[float], target: float) -> float:
    pairs = sorted(zip(map(float, xs), map(float, ys)))
    for x, y in pairs:
        if x == target:
            return y
    for (left_x, left_y), (right_x, right_y) in zip(pairs, pairs[1:]):
        if left_x < target < right_x:
            return left_y + (right_y - left_y) * (target - left_x) / (right_x - left_x)
    raise RunnerError("center wavelength is outside the sampled monitor band")


def derive_observables(manifest: Mapping[str, Any], monitors: Sequence[Mapping[str, Any]],
                       references: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_name = {item["name"]: item for item in monitors}
    wavelengths = manifest["excitation"]["monitor_sampling"]["wavelengths_um"]
    center = float(manifest["excitation"]["center_wavelength_um"])
    definitions = {item["name"]: item for item in manifest["case"]["monitors"]}
    incident = by_name.get("input_incident")
    if incident is None:
        raise RunnerError("input_incident monitor is missing")
    denominator = list(map(float, incident["forward_mode_power"]))
    ratios: dict[str, list[float]] = {}
    for name, definition in definitions.items():
        if definition.get("normalization", {}).get("kind") == "mode_power_ratio":
            numerator = list(map(float, by_name[name]["forward_mode_power"]))
            if len(numerator) != len(wavelengths) or len(denominator) != len(wavelengths):
                raise RunnerError("monitor array length disagrees with wavelength sampling")
            if any(value <= 0 for value in denominator):
                raise RunnerError("incident mode power must be positive")
            ratios[name] = [num / den for num, den in zip(numerator, denominator)]
    ring_values: dict[str, float] = {}
    if any(policy["evaluation"].startswith("not_a_knot") for policy in manifest["validation_policy"]["required_observables"]):
        try:
            import numpy as np
            from scipy.interpolate import CubicSpline
        except ImportError as error:
            raise RunnerError("ring observable evaluation requires NumPy and SciPy") from error
        values = ratios["through_te0"]
        maximum = max(values)
        if maximum <= 0:
            raise RunnerError("ring transmission maximum must be positive")
        normalized = [value / maximum for value in values]
        sample_x = np.linspace(float(wavelengths[0]), float(wavelengths[-1]), 1000)
        sample_y = CubicSpline(wavelengths, normalized, bc_type="not-a-knot", extrapolate=False)(sample_x)
        minimum_index = int(np.argmin(sample_y)); half = 0.5 * (1.0 + float(sample_y[minimum_index]))
        def crossing(indices: Sequence[int]) -> float:
            hits = []
            for index in indices:
                y0, y1 = float(sample_y[index]), float(sample_y[index + 1])
                if (y0 - half) * (y1 - half) <= 0 and y0 != y1:
                    hits.append(float(sample_x[index] + (half - y0) * (sample_x[index + 1] - sample_x[index]) / (y1 - y0)))
            if not hits:
                raise RunnerError("ring half-depth crossing is missing")
            return hits[-1] if indices.start == 0 else hits[0]
        left = crossing(range(0, minimum_index)); right = crossing(range(minimum_index, len(sample_x) - 1))
        width = right - left
        if width <= 0:
            raise RunnerError("ring FWHM is not positive")
        resonance = 0.5 * (left + right)
        ring_values = {"resonance_wavelength_um": resonance, "fwhm_um": width,
                       "quality_factor": resonance / width}
    output = []
    for policy in manifest["validation_policy"]["required_observables"]:
        evaluation = policy["evaluation"]
        if evaluation == "linear_center_wavelength_mode_power_ratio":
            value = _linear_at(wavelengths, ratios[policy["monitor"]["name"]], center)
        elif evaluation == "negative_ten_log10_linear_center_wavelength_mode_power_ratio":
            ratio = _linear_at(wavelengths, ratios[policy["monitor"]["name"]], center)
            if ratio <= 0:
                raise RunnerError("excess-loss ratio must be positive")
            value = -10.0 * math.log10(ratio)
        else:
            value = ring_values[policy["name"]]
        reference = references[policy["name"]]
        allowed = policy["absolute_tolerance"] + policy["relative_tolerance"] * abs(reference["value"])
        output.append({"name": policy["name"], "monitor": policy["monitor"],
                       "unit": policy["unit"], "evaluation": evaluation, "value": value,
                       "reference_value": reference["value"],
                       "absolute_tolerance": policy["absolute_tolerance"],
                       "relative_tolerance": policy["relative_tolerance"],
                       "passed": abs(value - reference["value"]) <= allowed})
    return output


def advance_with_collective_stop(mp: Any, simulation: Any,
                                 stopping: Mapping[str, Any]) -> tuple[int, str]:
    start_step = int(simulation.fields.t)
    if stopping["kind"] == "fixed_steps":
        simulation.fields.advance(int(stopping["steps"]))
        return int(simulation.fields.t) - start_step, "fixed_steps"
    if stopping.get("observable") == "total_field_energy":
        raise RunnerError(
            "paper end-to-end total_field_energy is blocked: the resident multi-rank "
            "host query currently crashes; use the built-in point-field acceptance gate"
        )
    maximum = int(stopping["max_steps"])
    component_name = stopping.get("component")
    point = stopping.get("point")
    if not isinstance(component_name, str) or not isinstance(point, Sequence) or len(point) != 3:
        raise RunnerError("collective point-field decay requires component and three-vector point")
    decay = mp.stop_when_fields_decayed(float(stopping["check_interval_meep_time"]),
                                        getattr(mp, component_name), mp.Vector3(*point),
                                        float(stopping["relative_threshold"]))
    state = {"decayed": False}

    def collective_stop(sim: Any) -> bool:
        if int(sim.fields.t) - start_step >= maximum:
            return True
        # stop_when_energy_decayed calls field_energy_in_box, whose active-
        # communicator reduction gives every rank the same value and decision.
        state["decayed"] = bool(decay(sim))
        return state["decayed"]

    simulation.run(until=collective_stop)
    return int(simulation.fields.t) - start_step, (
        "field_energy_decay" if state["decayed"] else "max_steps"
    )


def timed_advance_with_collective_stop(
    mp: Any, simulation: Any, stopping: Mapping[str, Any]
) -> tuple[int, str, float, Mapping[str, Any], Mapping[str, Any]]:
    mp.all_wait()
    initial_runtime = simulation.get_execution_runtime_report()
    started = time.perf_counter()
    steps, reason = advance_with_collective_stop(mp, simulation, stopping)
    mp.all_wait()
    elapsed = time.perf_counter() - started
    final_runtime = simulation.get_execution_runtime_report()
    return steps, reason, elapsed, initial_runtime, final_runtime


def reconcile_rank_records(records: Sequence[Mapping[str, Any]], manifest_sha256: str,
                           requested: Mapping[str, Any]) -> dict[str, Any]:
    if not records:
        raise RunnerError("no rank records were gathered")
    size = len(records)
    ranks = [record.get("rank") for record in records]
    if ranks != list(range(size)):
        raise RunnerError(f"rank records are not communicator ordered: {ranks}")
    if size != requested["ranks"]:
        raise RunnerError("gathered communicator size does not match the request")
    identities = set()
    hashes = set()
    launch_hashes = set()
    provenance_hashes = set()
    module_hashes = set()
    for record in records:
        hashes.add(record.get("manifest_sha256"))
        launch_hashes.add(record.get("launch_sha256"))
        provenance_hashes.add(record.get("provenance_sha256"))
        module_hashes.add(tuple(sorted(record.get("module_sha256", {}).items())))
        identities.add((record.get("communicator_size"), record.get("communicator_generation")))
        if record.get("communicator_size") != size:
            raise RunnerError("rank communicator size disagrees with gathered records")
        runtime = record.get("runtime")
        if not isinstance(runtime, Mapping):
            raise RunnerError("rank runtime report is missing")
        for key, expected in (("requested_backend", requested["backend"]),
                              ("requested_precision", requested["precision"]),
                              ("requested_transport", requested["route"]),
                              ("requested_overlap", requested["overlap"]),
                              ("requested_graph", requested["graph"])):
            if runtime.get(key) != expected:
                raise RunnerError(f"rank {record['rank']} {key} disagrees with the request")
        if runtime.get("communicator_rank") != record["rank"] or runtime.get("communicator_size") != size:
            raise RunnerError("runtime report communicator identity disagrees with its rank record")
        if runtime.get("counter_scope") != "rank_local_current_epoch" or runtime.get("backend_counter_scope") != "rank_local_backend_lifetime" or runtime.get("memory_gauge_scope") != "rank_local_process_lifetime":
            raise RunnerError("runtime counter scope is not rank-local current-epoch")
        if (runtime.get("setup_counter_scope") != "rank_local_current_backend_state" or
                runtime.get("transport_timing_scope") !=
                "rank_local_current_transport_epoch_host_elapsed" or
                runtime.get("allocation_counter_scope") != "rank_local_process_lifetime"):
            raise RunnerError("runtime timing/allocation counter scope is invalid")
        if not runtime.get("captured_transport_epoch_active") or not runtime.get("captured_transport_epoch_fresh"):
            raise RunnerError("runtime transport epoch is absent or stale")
        if runtime.get("captured_requested_transport") != requested["route"] or runtime.get("captured_overlap_policy") != requested["overlap"]:
            raise RunnerError("captured transport policy disagrees with the request")
        if not runtime.get("device_owner") or record.get("role") != "owner":
            raise RunnerError("the normative np2 runner requires one GPU-owning rank per process")
    if hashes != {manifest_sha256}:
        raise RunnerError("immutable manifest hash disagrees across ranks")
    if len(launch_hashes) != 1 or len(provenance_hashes) != 1:
        raise RunnerError("launch/provenance records disagree across ranks")
    if len(module_hashes) != 1:
        raise RunnerError("loaded Meep module hashes disagree across ranks")
    if len(identities) != 1:
        raise RunnerError("communicator identity disagrees across ranks")

    repetition_count = len(records[0]["repetitions"])
    if repetition_count < 1 or any(len(r["repetitions"]) != repetition_count for r in records):
        raise RunnerError("rank repetition counts disagree")
    critical = []
    per_repetition = []
    for index in range(repetition_count):
        values = [_number(r["repetitions"][index]["steady_seconds"], "steady_seconds") for r in records]
        maximum, median = max(values), statistics.median(values)
        critical.append(maximum)
        per_repetition.append({
            "index": index, "per_rank_seconds": values,
            "global_max_seconds": maximum,
            "rank_median_seconds": median,
            "imbalance_ratio": maximum / median if median else None,
        })
    counters = []
    for name, aggregation in COUNTER_AGGREGATIONS.items():
        values = []
        for record in records:
            deltas = []
            for repetition in record["repetitions"]:
                if set(repetition["counter_start"]) != set(COUNTER_AGGREGATIONS) or set(repetition["counter_end"]) != set(COUNTER_AGGREGATIONS) or set(repetition["counter_deltas"]) != set(COUNTER_AGGREGATIONS):
                    raise RunnerError("counter snapshots do not contain the declared counter set")
                delta = repetition["counter_end"][name] - repetition["counter_start"][name]
                if repetition["counter_deltas"][name] != delta:
                    raise RunnerError(f"counter delta {name} disagrees with its snapshots")
                deltas.append(delta)
            value = sum(deltas)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RunnerError(f"rank {record['rank']} counter {name} is invalid")
            values.append(value)
        counters.append({"name": name, "aggregation": aggregation,
                         "per_rank": values, "value": sum(values)})
    for name, aggregation in GAUGE_AGGREGATIONS.items():
        values = [int(record["runtime"][name]) for record in records]
        if any(value < 0 for value in values):
            raise RunnerError(f"gauge {name} is invalid")
        counters.append({"name": name, "aggregation": aggregation,
                         "per_rank": values, "value": max(values)})
    resolved = records[0]["runtime"]
    for record in records[1:]:
        for key in ("resolved_backend", "resolved_precision", "resolved_transport",
                    "resolved_overlap", "resolved_graph", "mpi_provider",
                    "mpi_query_available", "mpi_cuda_aware"):
            if record["runtime"].get(key) != resolved.get(key):
                raise RunnerError(f"resolved {key} disagrees across ranks")
    if len({record["runtime"]["captured_provider_signature"] for record in records}) != 1:
        raise RunnerError("captured MPI provider signature disagrees across ranks")
    for requested_key, resolved_key in (("precision", "resolved_precision"),
                                        ("route", "resolved_transport")):
        if requested[requested_key] != "auto" and resolved[resolved_key] != requested[requested_key]:
            raise RunnerError(f"requested {requested_key} was downgraded")
    if requested["overlap"] == "required" and resolved["resolved_overlap"] != "overlap":
        raise RunnerError("required overlap was not admitted")
    if requested["graph"] == "required" and resolved["resolved_graph"] != "graph":
        raise RunnerError("required graph execution was not enabled")
    shapes = {tuple(rep["grid_shape"]) for record in records for rep in record["repetitions"]}
    steps = {rep["steps"] for record in records for rep in record["repetitions"]}
    dts = {rep["dt_meep"] for record in records for rep in record["repetitions"]}
    if len(shapes) != 1 or len(steps) != 1 or len(dts) != 1:
        raise RunnerError("simulation shape/step/timestep disagrees across ranks or repetitions")
    for index in range(repetition_count):
        monitor_sets = [record["repetitions"][index]["monitors"] for record in records]
        if any(value != monitor_sets[0] for value in monitor_sets[1:]):
            raise RunnerError("global monitor output disagrees across ranks")
    shape, step_count, dt_meep = next(iter(shapes)), next(iter(steps)), next(iter(dts))
    return {
        "communicator": {"size": size, "generation": records[0]["communicator_generation"]},
        "resolved_execution": {
            "backend": resolved["resolved_backend"], "precision": resolved["resolved_precision"],
            "route": resolved["resolved_transport"], "overlap": resolved["resolved_overlap"],
            "graph": resolved["resolved_graph"],
        },
        "rank_records": list(records),
        "timing": {
            "repetitions": per_repetition, "critical_path_samples_seconds": critical,
            "minimum_seconds": min(critical), "median_seconds": statistics.median(critical),
            "maximum_seconds": max(critical),
        },
        "counter_aggregates": counters,
        "simulation": {"grid_shape": list(shape), "grid_points_exact": math.prod(shape),
                       "steps": step_count, "dt_meep": dt_meep,
                       "physical_time_meep": step_count * dt_meep},
        "performance": {"grid_timesteps_per_second": math.prod(shape) * step_count / statistics.median(critical),
                        "rate_basis": "critical_path_median_seconds"},
    }


def validate_result(result: Mapping[str, Any], *, manifest: Mapping[str, Any] | None = None,
                    references: Mapping[str, Mapping[str, Any]] | None = None) -> None:
    schema = bm.load_json_object(SCHEMA, "MPI benchmark result schema")
    bm._validate_schema_structure(result, schema, schema, "MPI benchmark result")
    launch = result["launch"]
    environment = launch["environment"]
    expected_provider = {"OMPI_MCA_opal_cuda_support": "true", "OMPI_MCA_pml": "ucx",
                         "UCX_TLS": UCX_TLS}
    if any(environment.get(key) != value for key, value in expected_provider.items()):
        raise RunnerError("result does not record the authoritative UCX provider environment")
    if pathlib.Path(launch["argv"][0]) != pathlib.Path(launch["mpiexec"]) or launch["python"] not in launch["argv"]:
        raise RunnerError("result launcher argv disagrees with its recorded prefix")
    requested = result["requested_execution"]
    expected_policy = {"staged": "no", "direct": "yes", "auto": "auto"}[requested["route"]]
    if environment.get("MEEP_GPU_AWARE_MPI") != expected_policy:
        raise RunnerError("recorded route environment contradicts requested execution")
    for key, environment_name in (("overlap", "MEEP_NVIDIA_MPI_OVERLAP"),
                                  ("graph", "MEEP_NVIDIA_GRAPH_MODE"),
                                  ("precision", "MEEP_PRECISION")):
        if environment.get(environment_name) != requested[key]:
            raise RunnerError(f"recorded {environment_name} contradicts requested execution")
    reconciled = reconcile_rank_records(result["rank_records"], result["run_manifest"]["sha256"], requested)
    for key in ("communicator", "resolved_execution", "timing", "counter_aggregates",
                "simulation", "performance"):
        if result[key] != reconciled[key]:
            raise RunnerError(f"published {key} does not match rank records")
    runtime = result["rank_records"][0]["runtime"]
    if requested["route"] == "direct" and not (runtime["mpi_query_available"] and runtime["mpi_cuda_aware"]):
        raise RunnerError("direct result lacks a positive CUDA-aware MPI provider query")
    uuids = [normalize_gpu_uuid(record["device"].get("uuid"))
             for record in result["rank_records"] if record["device"]]
    if len(uuids) != len(set(uuids)):
        raise RunnerError("GPU UUIDs are not unique across owner ranks")
    for record in result["rank_records"]:
        if record["role"] == "owner":
            required_device = {"visible_device", "process_device_id", "physical_selector", "uuid", "name",
                               "memory_bytes", "sm_clock_hz", "memory_clock_hz", "driver_version"}
            if (not isinstance(record["device"], Mapping) or
                    set(record["device"]) != required_device or
                    normalize_gpu_uuid(record["device"].get("uuid")) !=
                    normalize_gpu_uuid(record["runtime"].get("device_uuid"))):
                raise RunnerError(f"rank {record['rank']} GPU identity is inconsistent")
        elif record["device"] is not None:
            raise RunnerError(f"idle rank {record['rank']} unexpectedly records a GPU")
    canonical = lambda value: json.dumps(value, sort_keys=True).encode("utf-8")
    launch_hash = hashlib.sha256(canonical(result["launch"])).hexdigest()
    provenance_hash = hashlib.sha256(canonical(result["provenance"])).hexdigest()
    if any(record["launch_sha256"] != launch_hash or record["provenance_sha256"] != provenance_hash for record in result["rank_records"]):
        raise RunnerError("rank launch/provenance hashes do not bind the published records")
    for record in result["rank_records"]:
        affinity = record["cpu_affinity"]
        if affinity != sorted(set(affinity)):
            raise RunnerError(f"rank {record['rank']} CPU affinity is not canonical")
    if not all(observable["passed"] for observable in result["physics_observables"]):
        raise RunnerError("published result contains a failed physics observable")
    counters = {item["name"]: item["value"] for item in result["counter_aggregates"]}
    if counters["messages_sent"] != counters["messages_received"] or counters["bytes_sent"] != counters["bytes_received"]:
        raise RunnerError("global MPI send/receive accounting is unbalanced")
    route = result["resolved_execution"]["route"]
    for record in result["rank_records"]:
        per_rank = {
            name: sum(repetition["counter_deltas"][name]
                      for repetition in record["repetitions"])
            for name in COUNTER_AGGREGATIONS
        }
        runtime = record["runtime"]
        if route == "direct":
            if not runtime["mpi_query_available"] or not runtime["mpi_cuda_aware"]:
                raise RunnerError(
                    f"rank {record['rank']} direct route lacks a positive CUDA-aware MPI query"
                )
            if (any(per_rank[name] for name in (
                    "device_to_host_calls", "device_to_host_bytes",
                    "host_to_device_calls", "host_to_device_bytes")) or
                    runtime["transport_pinned_bytes"] != 0 or
                    per_rank["direct_bytes"] <= 0 or
                    per_rank["direct_bytes"] !=
                    per_rank["bytes_sent"] + per_rank["bytes_received"]):
                raise RunnerError(
                    f"rank {record['rank']} direct transport accounting is inconsistent"
                )
        elif route == "staged":
            if (per_rank["direct_bytes"] != 0 or
                    per_rank["device_to_host_calls"] <= 0 or
                    per_rank["host_to_device_calls"] <= 0 or
                    per_rank["device_to_host_bytes"] != per_rank["bytes_sent"] or
                    per_rank["host_to_device_bytes"] != per_rank["bytes_received"] or
                    runtime["transport_pinned_bytes"] <= 0):
                raise RunnerError(
                    f"rank {record['rank']} staged transport accounting is inconsistent"
                )
        if result["resolved_execution"]["graph"] == "graph" and (
            not runtime["graph_enabled"] or not runtime["graph_valid"] or
            per_rank["graph_launch_count"] <= 0
        ):
            raise RunnerError(
                f"rank {record['rank']} resolved graph mode did not execute a valid graph"
            )
    if route == "direct":
        if any(counters[name] for name in ("device_to_host_calls", "device_to_host_bytes",
                                           "host_to_device_calls", "host_to_device_bytes",
                                           "transport_pinned_bytes")):
            raise RunnerError("direct transport recorded halo staging")
        if counters["direct_bytes"] != counters["bytes_sent"] + counters["bytes_received"]:
            raise RunnerError("direct byte accounting is inconsistent")
    elif route == "staged":
        if counters["direct_bytes"] != 0 or counters["device_to_host_bytes"] != counters["bytes_sent"] or counters["host_to_device_bytes"] != counters["bytes_received"] or counters["transport_pinned_bytes"] <= 0:
            raise RunnerError("staged byte accounting is inconsistent")
    if result["resolved_execution"]["overlap"] == "off" and any(
        counters[name] for name in ("overlap_stages", "overlap_interior_launches", "overlap_boundary_launches")
    ):
        raise RunnerError("disabled overlap recorded overlap work")
    if result["requested_execution"]["overlap"] == "required" and any(
        counters[name] <= 0 for name in ("overlap_stages", "overlap_interior_launches", "overlap_boundary_launches")
    ):
        raise RunnerError("required overlap did not execute")
    if manifest is not None and references is not None:
        requested_manifest = manifest["execution"]["requested"]
        expected_requested = {"backend": requested_manifest["backend"],
                              "precision": requested_manifest["precision"],
                              "route": requested_manifest["mpi_transport"],
                              "overlap": requested_manifest["overlap"],
                              "graph": requested_manifest["graph"],
                              "ranks": requested_manifest["ranks"]}
        if result["requested_execution"] != expected_requested:
            raise RunnerError("published requested execution does not match the manifest")
        expected_repetitions = manifest["execution"]["measured_repetitions"]
        if any(len(record["repetitions"]) != expected_repetitions for record in result["rank_records"]):
            raise RunnerError("published repetition count does not match the manifest")
        for policy in manifest["validation_policy"]["required_observables"]:
            reference = references[policy["name"]]
            if reference["monitor"] != policy["monitor"]["name"] or reference["unit"] != policy["unit"]:
                raise RunnerError(f"physics reference {policy['name']} monitor/unit is not canonical")
        expected = []
        for repetition in result["rank_records"][0]["repetitions"]:
            expected.append(derive_observables(manifest, repetition["monitors"], references))
        published = result["physics_observables"]
        if [item["name"] for item in published] != [item["name"] for item in expected[0]]:
            raise RunnerError("published physics observable set is not canonical")
        for index, item in enumerate(published):
            values = [group[index]["value"] for group in expected]
            candidate = dict(expected[0][index]); candidate["values_by_repetition"] = values
            candidate["value"] = statistics.median(values)
            candidate["passed"] = all(group[index]["passed"] for group in expected)
            if item != candidate:
                raise RunnerError(f"published physics observable {item['name']} is not derived from raw monitors")


def authenticate_result_files(result: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    manifest, manifest_hash = manifest_snapshot(pathlib.Path(result["run_manifest"]["path"]))
    if manifest_hash != result["run_manifest"]["sha256"]:
        raise RunnerError("published result manifest hash is stale")
    details = bm._validate_run_manifest_for_result(manifest)
    if details["case_id"] != result["run_manifest"]["case_id"]:
        raise RunnerError("published result case does not match its manifest")
    reference_path = pathlib.Path(result["physics_reference"]["path"])
    reference_hash = bm.sha256_file(reference_path)
    if reference_hash != result["physics_reference"]["sha256"]:
        raise RunnerError("published result physics-reference hash is stale")
    references = bm._load_physics_reference(reference_path, reference_hash, details)
    validate_result(result, manifest=manifest, references=references)
    return manifest, references


def atomic_publish(result: Mapping[str, Any], path: pathlib.Path, *,
                   manifest: Mapping[str, Any] | None = None,
                   references: Mapping[str, Mapping[str, Any]] | None = None) -> None:
    validate_result(result, manifest=manifest, references=references)
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temporary = pathlib.Path(stream.name)
            stream.write(rendered); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _local_rank() -> int:
    for name in ("OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID", "SLURM_LOCALID"):
        if name in os.environ:
            return int(os.environ[name])
    return 0


def _worker(args: argparse.Namespace) -> int:
    mp = importlib.import_module("meep")
    mp.verbosity(0)
    if mp.count_processors() > 1:
        mp.begin_global_communications()
        mp.divide_parallel_processes(1)
    gather = getattr(mp, "active_communicator_allgather_json", None)
    if gather is None:
        raise RunnerError("Meep lacks active_communicator_allgather_json(payload: str) -> list[str]")
    try:
        manifest, digest = manifest_snapshot(args.manifest)
        if digest != args.manifest_sha256:
            raise RunnerError("manifest changed between launch and worker startup")
        details = bm._validate_run_manifest_for_result(manifest)
        if manifest["result_schema"]["schema_version"] != 2:
            raise RunnerError("multi-rank execution requires the normative result schema version 2")
        reference, reference_digest = manifest_snapshot(args.physics_reference)
        if reference_digest != args.physics_reference_sha256:
            raise RunnerError("physics reference changed between launch and worker startup")
        references = bm._load_physics_reference(args.physics_reference, reference_digest, details)
        gdstk = importlib.import_module("gdstk")
        manifest_phase = {"ok": True, "error": None, "manifest_sha256": digest,
                          "physics_reference_sha256": reference_digest,
                          "requested": details["requested"]}
    except Exception as error:
        manifest_phase = {"ok": False, "error": f"{type(error).__name__}: {error}",
                          "manifest_sha256": None, "physics_reference_sha256": None,
                          "requested": None}
    manifest_phases = [json.loads(item) for item in gather(json.dumps(manifest_phase, sort_keys=True))]
    phase_errors = [f"rank {index}: {item.get('error')}" for index, item in enumerate(manifest_phases) if not item.get("ok")]
    if phase_errors:
        raise RunnerError("collective manifest phase failed: " + "; ".join(phase_errors))
    if {item["manifest_sha256"] for item in manifest_phases} != {args.manifest_sha256}:
        raise RunnerError("immutable manifest hash disagrees during collective preflight")
    if {item["physics_reference_sha256"] for item in manifest_phases} != {args.physics_reference_sha256}:
        raise RunnerError("physics reference hash disagrees during collective preflight")
    if any(item["requested"] != manifest_phases[0]["requested"] for item in manifest_phases[1:]):
        raise RunnerError("requested execution disagrees during collective preflight")
    try:
        records, translation = legacy._geometry_records(manifest, gdstk)
        planes = legacy.validate_planes(manifest, legacy.transformed_ports(manifest, translation))
        geometry_phase = {"ok": True, "error": None}
    except Exception as error:
        geometry_phase = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    geometry_phases = [json.loads(item) for item in gather(json.dumps(geometry_phase, sort_keys=True))]
    phase_errors = [f"rank {index}: {item.get('error')}" for index, item in enumerate(geometry_phases) if not item.get("ok")]
    if phase_errors:
        raise RunnerError("collective geometry phase failed: " + "; ".join(phase_errors))
    try:
        repetitions = []
        runtime = None
        for repetition_index in range(manifest["execution"]["measured_repetitions"]):
            try:
                started = time.perf_counter()
                simulation, monitors = legacy._build_simulation(mp, manifest, records, planes,
                                                                 device_id=_local_rank())
                initialization = time.perf_counter() - started
                warmup = manifest["execution"]["warmup_steps"]
                warmup_started = time.perf_counter()
                if warmup: simulation.fields.advance(warmup)
                warmup_seconds = time.perf_counter() - warmup_started
                steps, stop_reason, steady, initial_runtime, runtime = \
                    timed_advance_with_collective_stop(
                    mp, simulation, details["stopping"]
                )
                monitor_started = time.perf_counter()
                monitor_values = legacy._monitor_output(mp, simulation, monitors, manifest)
                monitor_seconds = time.perf_counter() - monitor_started
                counter_deltas = {name: int(runtime[name]) - int(initial_runtime[name]) for name in COUNTER_AGGREGATIONS}
                if any(value < 0 for value in counter_deltas.values()):
                    raise RunnerError("a runtime counter decreased within a repetition")
                repetition = {"initialization_seconds": initialization, "warmup_seconds": warmup_seconds,
                              "steady_seconds": steady, "monitor_seconds": monitor_seconds, "steps": steps,
                              "dt_meep": float(simulation.fields.dt), "grid_shape": legacy._grid_shape(simulation),
                              "stop_reason": stop_reason,
                              "counter_start": {name: int(initial_runtime[name]) for name in COUNTER_AGGREGATIONS},
                              "counter_end": {name: int(runtime[name]) for name in COUNTER_AGGREGATIONS},
                              "counter_deltas": counter_deltas, "monitors": monitor_values}
                simulation.reset_meep()
                repetition_phase = {"ok": True, "error": None, "repetition": repetition,
                                    "runtime": runtime}
            except Exception as error:
                repetition_phase = {"ok": False, "error": f"{type(error).__name__}: {error}",
                                    "repetition": None, "runtime": None}
            repetition_phases = [json.loads(item) for item in gather(json.dumps(repetition_phase, sort_keys=True, allow_nan=False))]
            phase_errors = [f"rank {index}: {item.get('error')}" for index, item in enumerate(repetition_phases) if not item.get("ok")]
            if phase_errors:
                raise RunnerError(f"collective repetition {repetition_index} failed: " + "; ".join(phase_errors))
            local_phase = repetition_phases[int(mp.my_rank())]
            repetitions.append(local_phase["repetition"]); runtime = local_phase["runtime"]
        if runtime is None:
            raise RunnerError("no benchmark repetition ran")
        rank = int(runtime["communicator_rank"])
        device = legacy._nvidia_device_provenance(int(runtime["device_id"]))[0] if runtime["device_owner"] else None
        launch_bytes = args.launch_record_file.read_bytes()
        provenance_bytes = args.provenance_record_file.read_bytes()
        meep_module = pathlib.Path(str(mp.__file__)).resolve()
        meep_extension = pathlib.Path(
            str(getattr(getattr(mp, "_meep", None), "__file__", ""))
        ).resolve()
        if not meep_module.is_file() or not meep_extension.is_file():
            raise RunnerError("loaded Meep module or extension is not a regular file")
        local = {
        "rank": rank, "communicator_size": int(runtime["communicator_size"]),
        "communicator_generation": int(runtime["communicator_generation"]),
        "manifest_sha256": digest,
        "launch_sha256": hashlib.sha256(launch_bytes).hexdigest(),
        "provenance_sha256": hashlib.sha256(provenance_bytes).hexdigest(),
        "hostname": socket.gethostname(), "local_rank": _local_rank(),
        "role": "owner" if runtime["device_owner"] else "idle", "device": device,
        "runtime": runtime, "repetitions": repetitions,
        "module_paths": {"meep": str(meep_module), "extension": str(meep_extension)},
        "module_sha256": {"meep": bm.sha256_file(meep_module),
                          "extension": bm.sha256_file(meep_extension)},
        "cpu_affinity": observed_cpu_affinity(),
        }
        envelope = {"ok": True, "record": local, "error": None}
    except Exception as error:
        envelope = {"ok": False, "record": None, "error": f"{type(error).__name__}: {error}"}
    envelopes = [json.loads(item) for item in gather(json.dumps(envelope, sort_keys=True, allow_nan=False))]
    errors = [f"rank {index}: {item.get('error')}" for index, item in enumerate(envelopes) if not item.get("ok")]
    if errors:
        raise RunnerError("collective worker phase failed: " + "; ".join(errors))
    gathered = [item["record"] for item in envelopes]
    rank = int(mp.my_rank())
    requested = {"backend": details["requested"]["backend"], "precision": args.precision,
                 "route": args.route, "overlap": details["requested"]["overlap"],
                 "graph": details["requested"]["graph"],
                 "ranks": details["requested"]["ranks"]}
    reconciled = reconcile_rank_records(gathered, digest, requested)
    per_repetition_observables = [derive_observables(manifest, gathered[0]["repetitions"][index]["monitors"], references)
                                  for index in range(len(gathered[0]["repetitions"]))]
    physics_observables = []
    for index, observable in enumerate(per_repetition_observables[0]):
        values = [items[index]["value"] for items in per_repetition_observables]
        merged = dict(observable)
        merged["values_by_repetition"] = values
        merged["value"] = statistics.median(values)
        reference_value = float(merged["reference_value"])
        allowed = float(merged["absolute_tolerance"]) + float(merged["relative_tolerance"]) * abs(reference_value)
        merged["passed"] = all(abs(value - reference_value) <= allowed for value in values)
        physics_observables.append(merged)
    if rank == 0:
        result = {
            "schema_version": 2, "kind": "paper_2506_16665_multi_rank_benchmark",
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
            "run_manifest": {"path": str(args.manifest.resolve()), "sha256": digest,
                             "case_id": details["case_id"]},
            "physics_reference": {"path": str(args.physics_reference.resolve()),
                                  "sha256": args.physics_reference_sha256},
            "status": {"succeeded": True, "errors": []},
            "launch": json.loads(args.launch_record_file.read_text(encoding="utf-8")),
            "provenance": json.loads(args.provenance_record_file.read_text(encoding="utf-8")),
            "requested_execution": requested,
            "physics_observables": physics_observables,
            **reconciled,
        }
        atomic_publish(result, args.output, manifest=manifest, references=references)
    if mp.count_processors() > 1:
        mp.end_divide_parallel()
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--manifest", required=True, type=pathlib.Path)
    value.add_argument("--output", required=True, type=pathlib.Path)
    value.add_argument("--physics-reference", required=True, type=pathlib.Path)
    value.add_argument("--route", required=True, choices=("staged", "direct", "auto"))
    value.add_argument("--overlap", required=True, choices=("off", "auto", "required"))
    value.add_argument("--graph", required=True, choices=("eager", "auto", "required"))
    value.add_argument("--precision", required=True, choices=("native", "f32", "mixed"))
    value.add_argument("--timeout", type=float, default=1800.0)
    value.add_argument("--prefix", type=pathlib.Path, default=DEFAULT_PREFIX)
    value.add_argument("--build-directory", required=True, type=pathlib.Path)
    value.add_argument("--cuda-visible-devices", default="0,1")
    value.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    value.add_argument("--manifest-sha256", help=argparse.SUPPRESS)
    value.add_argument("--physics-reference-sha256", help=argparse.SUPPRESS)
    value.add_argument("--launch-record-file", type=pathlib.Path, help=argparse.SUPPRESS)
    value.add_argument("--provenance-record-file", type=pathlib.Path, help=argparse.SUPPRESS)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.worker:
            return _worker(args)
        manifest, digest = manifest_snapshot(args.manifest)
        details = bm._validate_run_manifest_for_result(manifest)
        if manifest["result_schema"]["schema_version"] != 2:
            raise RunnerError("multi-rank execution requires the normative result schema version 2")
        _reference, reference_digest = manifest_snapshot(args.physics_reference)
        bm._load_physics_reference(args.physics_reference, reference_digest, details)
        requested = details["requested"]
        if (requested["ranks"] != 2 or requested["mpi_transport"] != args.route or
                requested["precision"] != args.precision or
                requested["overlap"] != args.overlap or
                requested["graph"] != args.graph):
            raise RunnerError("normative runner currently requires an exact two-rank manifest and matching policies")
        mpiexec, python = args.prefix.resolve() / "bin/mpirun", args.prefix.resolve() / "bin/python"
        if not mpiexec.is_file() or not python.is_file():
            raise RunnerError("the selected prefix lacks bin/mpirun or bin/python")
        env = launch_environment(os.environ, args.route, args.overlap, args.graph, args.precision)
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        site_packages = str(python_purelib(python))
        env["PYTHONPATH"] = site_packages + ((":" + env["PYTHONPATH"]) if env.get("PYTHONPATH") else "")
        libdir = str(args.prefix.resolve() / "lib")
        env["LD_LIBRARY_PATH"] = libdir + ((":" + env["LD_LIBRARY_PATH"]) if env.get("LD_LIBRARY_PATH") else "")
        env_record = {name: env[name] for name in ("OMPI_MCA_opal_cuda_support", "OMPI_MCA_pml", "UCX_TLS",
                      "MEEP_GPU_AWARE_MPI", "MEEP_NVIDIA_MPI_OVERLAP", "MEEP_NVIDIA_GRAPH_MODE", "MEEP_PRECISION",
                      "CUDA_VISIBLE_DEVICES", "PYTHONPATH", "LD_LIBRARY_PATH")}
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=args.output.resolve().parent,
                                         prefix=".mpi-launch.", suffix=".json") as launch_file, tempfile.NamedTemporaryFile(
                                             "w", encoding="utf-8", dir=args.output.resolve().parent,
                                             prefix=".mpi-provenance.", suffix=".json") as provenance_file:
            command = [str(mpiexec), "-np", str(requested["ranks"]), "--map-by", "ppr:1:package:PE=8",
                       "--bind-to", "core", str(python), str(pathlib.Path(__file__).resolve()), "--worker",
                       "--manifest", str(args.manifest.resolve()), "--output", str(args.output.resolve()),
                       "--physics-reference", str(args.physics_reference.resolve()),
                       "--route", args.route, "--overlap", args.overlap, "--graph", args.graph,
                       "--precision", args.precision, "--timeout", str(args.timeout), "--manifest-sha256", digest,
                       "--physics-reference-sha256", reference_digest,
                       "--build-directory", str(args.build_directory.resolve()),
                       "--launch-record-file", launch_file.name,
                       "--provenance-record-file", provenance_file.name]
            launch = {"argv": command, "cwd": str(pathlib.Path.cwd()),
                      "environment": env_record, "timeout_seconds": args.timeout,
                      "mpiexec": str(mpiexec), "python": str(python),
                      "mpi_version": legacy._command_output([str(mpiexec), "--version"]),
                      "ucx_version": legacy._command_output([str(mpiexec.parent / "ucx_info"), "-v"])}
            json.dump(launch, launch_file, sort_keys=True); launch_file.flush(); os.fsync(launch_file.fileno())
            worktree = pathlib.Path(__file__).resolve().parents[3]
            source_tree = pathlib.Path(os.environ.get("MEEP_SOURCE_TREE", str(worktree))).resolve()
            build_directory = args.build_directory.resolve()
            provenance = {"meep_source": str(source_tree),
                          "meep_commit": legacy._git(source_tree, "rev-parse", "HEAD"),
                          "meep_dirty": bool(legacy._git(source_tree, "status", "--porcelain=v1")),
                          "runner_commit": legacy._git(worktree, "rev-parse", "HEAD"),
                          "runner_dirty": bool(legacy._git(worktree, "status", "--porcelain=v1")),
                          "build_directory": str(build_directory),
                          "cuda_toolkit": legacy._command_output(["nvcc", "--version"]),
                          **configured_build_provenance(build_directory, mpiexec, python)}
            json.dump(provenance, provenance_file, sort_keys=True); provenance_file.flush(); os.fsync(provenance_file.fileno())
            completed = subprocess.run(command, env=env, timeout=args.timeout)
            return completed.returncode
    except (bm.ValidationError, RunnerError, OSError, ValueError, subprocess.TimeoutExpired) as error:
        print(f"run_mpi_benchmark.py: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
