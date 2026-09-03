#!/usr/bin/env python3
"""Run a small built-in CPU/staged/direct end-to-end acceptance suite."""

from __future__ import annotations

import argparse, hashlib, importlib, json, math, os, pathlib, subprocess, sys, tempfile
from typing import Any, Mapping

import benchmark_manifest as bm
import run_mpi_benchmark as mpi_runner

CASE = {"name": "builtin_d1_transport_acceptance", "cell_z": 4.0, "dimensions": 1,
        "resolution": 10, "pml": 0.5, "medium": "vacuum", "component": "Ex",
        "source_z": 0.0, "frequency": 0.35, "fwidth": 0.14, "amplitude": 0.37,
        "decay_dt": 2.0, "decay_by": 1e-4, "max_steps": 2000,
        "sample_z": [-0.5, -0.25, 0.25, 0.5]}
SCHEMA = pathlib.Path(__file__).with_name("transport_acceptance.schema.json")
COMPARISON_SCHEMA = pathlib.Path(__file__).with_name("transport_acceptance_comparison.schema.json")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_acceptance_artifact(value: Mapping[str, Any], expected_route: str | None = None) -> None:
    schema = bm.load_json_object(SCHEMA, "transport acceptance schema")
    bm._validate_schema_structure(value, schema, schema, "transport acceptance")
    route = value["route"]
    if expected_route is not None and route != expected_route:
        raise mpi_runner.RunnerError(
            f"expected {expected_route} acceptance artifact, got {route}"
        )
    if set(value["case"]) != set(CASE) or value["case"] != CASE:
        raise mpi_runner.RunnerError("acceptance case does not exactly match the built-in case")
    if value["case_sha256"] != canonical_hash(CASE):
        raise mpi_runner.RunnerError("acceptance case hash does not match the built-in case")

    expected_ranks = 1 if route == "cpu" else 2
    records = value["rank_records"]
    if len(records) != expected_ranks or [record["rank"] for record in records] != list(
        range(expected_ranks)
    ):
        raise mpi_runner.RunnerError(
            f"{route} acceptance requires {expected_ranks} ordered rank record(s)"
        )
    required_runtime = {
        "resolved_backend", "requested_transport", "resolved_transport",
        "captured_requested_transport", "mpi_provider", "mpi_query_available",
        "mpi_cuda_aware", "communicator_rank", "communicator_size", "device_owner",
        "captured_transport_epoch_active", "captured_transport_epoch_fresh",
        "transport_pinned_bytes",
    }
    expected_transport = "none" if route == "cpu" else route
    for index, record in enumerate(records):
        runtime = record["runtime"]
        missing_runtime = required_runtime - set(runtime)
        if missing_runtime:
            raise mpi_runner.RunnerError(
                f"{route} rank {index} runtime is missing {sorted(missing_runtime)}"
            )
        if not isinstance(runtime["mpi_provider"], str) or not runtime["mpi_provider"]:
            raise mpi_runner.RunnerError(f"{route} rank {index} MPI provider is invalid")
        for name in (
            "mpi_query_available", "mpi_cuda_aware", "device_owner",
            "captured_transport_epoch_active", "captured_transport_epoch_fresh",
        ):
            if not isinstance(runtime[name], bool):
                raise mpi_runner.RunnerError(f"{route} rank {index} runtime {name} is invalid")
        for name in ("communicator_rank", "communicator_size", "transport_pinned_bytes"):
            if isinstance(runtime[name], bool) or not isinstance(runtime[name], int) or runtime[name] < 0:
                raise mpi_runner.RunnerError(f"{route} rank {index} runtime {name} is invalid")
        if runtime["communicator_rank"] != index or runtime["communicator_size"] != expected_ranks:
            raise mpi_runner.RunnerError(f"{route} runtime communicator identity is inconsistent")
        expected_backend = "cpu" if route == "cpu" else "nvidia"
        if runtime["resolved_backend"] != expected_backend or runtime["resolved_transport"] != expected_transport:
            raise mpi_runner.RunnerError(f"{route} runtime resolved route/backend is inconsistent")
        if route == "cpu":
            if runtime["device_owner"] or runtime["captured_transport_epoch_active"] or runtime["transport_pinned_bytes"] != 0:
                raise mpi_runner.RunnerError("CPU acceptance records unexpected GPU transport state")
        else:
            if (runtime["requested_transport"] != route or
                    runtime["captured_requested_transport"] != route or
                    not runtime["device_owner"] or
                    not runtime["captured_transport_epoch_active"] or
                    not runtime["captured_transport_epoch_fresh"]):
                raise mpi_runner.RunnerError(f"{route} runtime transport state is inconsistent")
            if route == "staged" and runtime["transport_pinned_bytes"] <= 0:
                raise mpi_runner.RunnerError("staged runtime has no pinned transport storage")
            if route == "direct" and (
                not runtime["mpi_query_available"] or not runtime["mpi_cuda_aware"] or
                runtime["transport_pinned_bytes"] != 0
            ):
                raise mpi_runner.RunnerError("direct runtime lacks positive GPU-aware MPI state")

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
            "steady_allocation_count", "graph_recapture_count", "full_field_copy_count"
        ):
            if counters[name] != 0:
                raise mpi_runner.RunnerError(
                    f"{route} rank {index} steady-state counter {name} is nonzero"
                )
        if route == "staged" and (
            counters["messages_sent"] <= 0 or counters["messages_received"] <= 0 or
            counters["bytes_sent"] <= 0 or counters["bytes_received"] <= 0 or
            counters["direct_bytes"] != 0 or
            counters["device_to_host_calls"] <= 0 or
            counters["host_to_device_calls"] <= 0 or
            counters["device_to_host_bytes"] != counters["bytes_sent"] or
            counters["host_to_device_bytes"] != counters["bytes_received"]
        ):
            raise mpi_runner.RunnerError(
                f"staged rank {index} counters do not prove host-staged transport"
            )
        if route == "direct" and (
            counters["messages_sent"] <= 0 or counters["messages_received"] <= 0 or
            counters["bytes_sent"] <= 0 or counters["bytes_received"] <= 0 or
            counters["direct_bytes"] != counters["bytes_sent"] + counters["bytes_received"] or
            counters["direct_bytes"] <= 0 or
            any(counters[name] for name in (
                "device_to_host_calls", "device_to_host_bytes",
                "host_to_device_calls", "host_to_device_bytes",
            ))
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
            if (not isinstance(sample, list) or len(sample) != 2 or
                    any(isinstance(component, bool) or
                        not isinstance(component, (int, float)) or
                        not math.isfinite(component) for component in sample)):
                raise mpi_runner.RunnerError(
                    f"{route} rank {index} contains an invalid observable sample"
                )


def load_acceptance_artifact(path: pathlib.Path, expected_route: str) -> dict[str, Any]:
    value = bm.load_json_object(path, f"{expected_route} acceptance")
    validate_acceptance_artifact(value, expected_route)
    return value


def _steady_measurement_start(mp: Any, sim: Any) -> tuple[dict[str, Any], int]:
    """Warm resident execution once, fence all ranks, then snapshot counters."""
    sim.fields.advance(1)
    mp.all_wait()
    return sim.get_execution_runtime_report(), int(sim.fields.t)


def _steady_measurement_end(mp: Any, sim: Any) -> dict[str, Any]:
    """Fence timed work before reading counters; monitor queries follow this."""
    mp.all_wait()
    return sim.get_execution_runtime_report()


def _atomic(value: Mapping[str, Any], path: pathlib.Path, schema_path: pathlib.Path = SCHEMA) -> None:
    schema = bm.load_json_object(schema_path, "transport acceptance schema")
    bm._validate_schema_structure(value, schema, schema, "transport acceptance")
    if schema_path == SCHEMA:
        validate_acceptance_artifact(value)
    path = path.resolve(); path.parent.mkdir(parents=True, exist_ok=True); temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temporary = pathlib.Path(stream.name); json.dump(value, stream, indent=2, sort_keys=True); stream.write("\n")
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path); fd = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    finally:
        if temporary is not None and temporary.exists(): temporary.unlink()


def _worker(route: str, output: pathlib.Path) -> int:
    mp = importlib.import_module("meep"); mp.verbosity(0)
    if route != "cpu":
        mp.begin_global_communications()
        mp.divide_parallel_processes(1)
    backend = "cpu" if route == "cpu" else "nvidia"
    kwargs = {"cell_size": mp.Vector3(0, 0, CASE["cell_z"]), "dimensions": CASE["dimensions"],
              "resolution": CASE["resolution"],
              "boundary_layers": [mp.PML(CASE["pml"])],
              "geometry": [],
              "sources": [mp.Source(mp.GaussianSource(CASE["frequency"], fwidth=CASE["fwidth"]), component=mp.Ex, center=mp.Vector3(z=CASE["source_z"]), amplitude=CASE["amplitude"])],
              "backend": backend, "accelerator_strict": False}
    if route != "cpu": kwargs["device_id"] = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    sim = mp.Simulation(**kwargs)
    sim.init_sim()
    # Force resident compilation/allocation before the measurement baseline.
    # The same deterministic step is applied to CPU, staged, and direct runs.
    before, start_step = _steady_measurement_start(mp, sim)
    decay = mp.stop_when_fields_decayed(CASE["decay_dt"], mp.Ex, mp.Vector3(z=0.5), CASE["decay_by"])
    sim.run(until_after_sources=decay)
    after = _steady_measurement_end(mp, sim)
    steps = int(sim.fields.t) - start_step
    if steps > CASE["max_steps"]:
        raise mpi_runner.RunnerError("acceptance decay exceeded its declared max_steps")
    reason = "field_energy_decay"
    observables = {"samples": [[float(complex(sim.get_field_point(mp.Ex, mp.Vector3(z=z))).real),
                                float(complex(sim.get_field_point(mp.Ex, mp.Vector3(z=z))).imag)] for z in CASE["sample_z"]]}
    rank = int(after["communicator_rank"])
    local = {"rank": rank, "runtime": after, "steps": steps, "stop_reason": reason,
             "observables": observables,
             "counter_deltas": {name: int(after[name]) - int(before[name])
                                for name in mpi_runner.COUNTER_AGGREGATIONS if name in after}}
    if route == "cpu": gathered = [local]
    else:
        gathered = [json.loads(item) for item in mp.active_communicator_allgather_json(json.dumps(local, sort_keys=True))]
    if rank == 0:
        source = pathlib.Path(__file__).resolve().parents[3]
        result = {"schema_version": 1, "kind": "meep_builtin_transport_acceptance",
                  "case": CASE, "case_sha256": canonical_hash(CASE), "route": route,
                  "build_identity": {"source": str(source), "commit": mpi_runner.legacy._git(source, "rev-parse", "HEAD"),
                                     "dirty": bool(mpi_runner.legacy._git(source, "status", "--porcelain=v1")),
                                     "python": sys.executable, "meep_module": str(mp.__file__),
                                     "meep_extension": str(getattr(getattr(mp, "_meep", None), "__file__", ""))},
                  "rank_records": gathered}
        _atomic(result, output)
    if route != "cpu": mp.end_divide_parallel()
    return 0


def _run(command, env, timeout):
    subprocess.run(command, env=env, timeout=timeout, check=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path)
    parser.add_argument("--prefix", type=pathlib.Path, default=mpi_runner.DEFAULT_PREFIX)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--pythonpath")
    parser.add_argument("--library-path")
    parser.add_argument("--worker", choices=("cpu", "staged", "direct"), help=argparse.SUPPRESS)
    parser.add_argument("--output", type=pathlib.Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.worker: return _worker(args.worker, args.output)
        if args.output_dir is None: raise mpi_runner.RunnerError("--output-dir is required")
        if not args.pythonpath or not args.library_path:
            raise mpi_runner.RunnerError("--pythonpath and --library-path are required")
        prefix = args.prefix.resolve(); python = prefix / "bin/python"; mpirun = prefix / "bin/mpirun"
        script = pathlib.Path(__file__).resolve(); args.output_dir.mkdir(parents=True, exist_ok=True)
        base = dict(os.environ); base["PYTHONPATH"] = str(pathlib.Path(args.pythonpath).resolve())
        base["LD_LIBRARY_PATH"] = str(pathlib.Path(args.library_path).resolve()) + ":" + str(prefix / "lib")
        cpu = args.output_dir / "cpu.json"; staged = args.output_dir / "staged.json"; direct = args.output_dir / "direct.json"
        _run([str(python), str(script), "--worker", "cpu", "--output", str(cpu)], base, args.timeout)
        for route, output in (("staged", staged), ("direct", direct)):
            env = mpi_runner.launch_environment(base, route, "off", "eager", "native"); env["CUDA_VISIBLE_DEVICES"] = "0,1"
            _run([str(mpirun), "-np", "2", "--map-by", "ppr:1:package:PE=2", "--bind-to", "core",
                  str(python), str(script), "--worker", route, "--output", str(output)], env, args.timeout)
        cpu_result = load_acceptance_artifact(cpu, "cpu")
        staged_result = load_acceptance_artifact(staged, "staged")
        direct_result = load_acceptance_artifact(direct, "direct")
        if not (cpu_result["build_identity"] == staged_result["build_identity"] == direct_result["build_identity"]):
            raise mpi_runner.RunnerError("acceptance routes did not use the same source and module build")
        baseline = cpu_result["rank_records"][0]["observables"]
        for candidate in (staged_result, direct_result):
            for record in candidate["rank_records"]:
                for actual, expected in zip(record["observables"]["samples"], baseline["samples"]):
                    if abs(complex(*actual) - complex(*expected)) > 1e-11 + 1e-8 * abs(complex(*expected)):
                        raise mpi_runner.RunnerError("GPU field sample failed CPU tolerance")
        if [r["observables"] for r in staged_result["rank_records"]] != [r["observables"] for r in direct_result["rank_records"]]:
            raise mpi_runner.RunnerError("staged/direct acceptance observables are not bitwise identical")
        for record in staged_result["rank_records"]:
            runtime, counters = record["runtime"], record["counter_deltas"]
            if runtime["resolved_transport"] != "staged" or counters["direct_bytes"] != 0 or counters["device_to_host_bytes"] != counters["bytes_sent"] or counters["host_to_device_bytes"] != counters["bytes_received"] or counters["device_to_host_calls"] <= 0 or counters["host_to_device_calls"] <= 0 or runtime["transport_pinned_bytes"] <= 0:
                raise mpi_runner.RunnerError("staged acceptance did not prove host-staged halo traffic on every rank")
        for record in direct_result["rank_records"]:
            runtime, counters = record["runtime"], record["counter_deltas"]
            if not runtime["mpi_query_available"] or not runtime["mpi_cuda_aware"] or runtime["resolved_transport"] != "direct" or counters["direct_bytes"] <= 0 or counters["direct_bytes"] != counters["bytes_sent"] + counters["bytes_received"] or any(counters[name] for name in ("device_to_host_calls", "device_to_host_bytes", "host_to_device_calls", "host_to_device_bytes")) or runtime["transport_pinned_bytes"] != 0:
                raise mpi_runner.RunnerError("direct acceptance did not prove GPU-aware MPI traffic on every rank")
        _atomic({"schema_version": 1, "kind": "meep_builtin_transport_acceptance_comparison",
                 "case_sha256": canonical_hash(CASE),
                 "cpu": {"path": str(cpu.resolve()), "sha256": bm.sha256_file(cpu)},
                 "staged": {"path": str(staged.resolve()), "sha256": bm.sha256_file(staged)},
                 "direct": {"path": str(direct.resolve()), "sha256": bm.sha256_file(direct)},
                 "cpu_tolerance_passed": True, "staged_direct_bitwise": True,
                 "direct_provider_positive": True}, args.output_dir / "comparison.json",
                COMPARISON_SCHEMA)
        return 0
    except Exception as error:
        print(f"run_transport_acceptance.py: error: {error}", file=sys.stderr); return 2


if __name__ == "__main__": raise SystemExit(main())
