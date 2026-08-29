#!/usr/bin/env python3
"""Execute authenticated single-rank Meep CPU/NVIDIA benchmark manifests.

The runner intentionally treats short smoke runs as construction and backend
diagnostics.  It records raw DFT flux and forward/backward mode power, but does
not turn a ten-step run into a paper-equivalent physics or speedup claim.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import importlib
import json
import math
import os
import pathlib
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import benchmark_manifest as bm


RESULT_SCHEMA_VERSION = 1
PROFILE_ENVIRONMENT = (
    "CUDA_VISIBLE_DEVICES",
    "MEEP_FINITE_CHECK",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MEEP_SOURCE_TREE",
    "MEEP_BUILD_DIR",
)


class RunnerError(RuntimeError):
    """Raised for unsupported or invalid executable benchmark inputs."""


def _finite_number(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise RunnerError(f"{label} is not finite")
    return number


def _atomic_json_write(value: Mapping[str, Any], path: pathlib.Path) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = pathlib.Path(stream.name)
        stream.write(rendered)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _git(root: pathlib.Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.rstrip("\n")


def _command_output(command: Sequence[str]) -> Optional[str]:
    try:
        return subprocess.run(
            list(command),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_runtime_modules() -> Tuple[Any, Any]:
    try:
        mp = importlib.import_module("meep")
        mp.verbosity(0)
        return mp, importlib.import_module("gdstk")
    except ImportError as error:
        raise RunnerError(
            "run_benchmark.py requires the branch-matched Meep Python module and "
            "gdstk; install gdstk in an isolated environment (see README.md)"
        ) from error


def validate_executable_manifest(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    details = bm._validate_run_manifest_for_result(manifest)
    requested = details["requested"]
    support = manifest["case"]["runner_support"]
    if not support["supported"]:
        raise RunnerError(
            f"case {manifest['case']['id']} is unsupported: {support['reason']}"
        )
    if requested["mode"] not in {"smoke", "fixed-step"}:
        raise RunnerError("the PR3 runner supports only smoke and fixed-step manifests")
    if requested["ranks"] != 1 or requested["mpi_transport"] != "none":
        raise RunnerError("PR3 NVIDIA execution is single-rank only")
    if requested["backend"] not in {"cpu", "nvidia"}:
        raise RunnerError(
            "the executable runner requires an explicit cpu or nvidia backend"
        )
    if manifest["materials"]["mode"] != "performance-adaptation":
        raise RunnerError("PR3 cannot execute the paper's dispersive material model")
    return details


def _select_gds_cell(gdstk: Any, path: pathlib.Path, name: str) -> Any:
    library = gdstk.read_gds(str(path))
    matches = [cell for cell in library.cells if cell.name == name]
    if len(matches) != 1:
        available = sorted(cell.name for cell in library.cells)
        raise RunnerError(
            f"GDS cell {name!r} was not uniquely found in {path}; available={available}"
        )
    return matches[0]


def _geometry_records(
    manifest: Mapping[str, Any], gdstk: Any
) -> Tuple[list, Dict[str, float]]:
    case = manifest["case"]
    gds_path = pathlib.Path(manifest["input_checkout"]["gds"]["path"])
    if bm.sha256_file(gds_path) != manifest["input_checkout"]["gds"]["sha256"]:
        raise RunnerError("GDS hash changed after manifest generation")
    cell = _select_gds_cell(gdstk, gds_path, case["gds_cell_name"])
    records = []
    all_points = []
    for layer in case["layers"]:
        polygons = cell.get_polygons(
            apply_repetitions=True,
            include_paths=True,
            depth=None,
            layer=int(layer["gds_layer"]),
            datatype=int(layer["gds_datatype"]),
        )
        if not polygons:
            raise RunnerError(
                f"GDS cell {cell.name!r} has no polygons for "
                f"layer/datatype {layer['gds_layer']}/{layer['gds_datatype']}"
            )
        for polygon in polygons:
            points = [[float(x), float(y)] for x, y in polygon.points]
            all_points.extend(points)
            records.append({"layer": dict(layer), "points_um": points})
    x_values = [point[0] for point in all_points]
    y_values = [point[1] for point in all_points]
    translation = {
        "x_um": -0.5 * (min(x_values) + max(x_values)),
        "y_um": -0.5 * (min(y_values) + max(y_values)),
    }
    for record in records:
        record["points_um"] = [
            [point[0] + translation["x_um"], point[1] + translation["y_um"]]
            for point in record["points_um"]
        ]
    return records, translation


def transformed_ports(
    manifest: Mapping[str, Any], translation: Mapping[str, float]
) -> Dict[str, Any]:
    return {
        name: {
            **port,
            "center_um": [
                float(port["center_um"][0]) + translation["x_um"],
                float(port["center_um"][1]) + translation["y_um"],
            ],
        }
        for name, port in manifest["case"]["ports"].items()
    }


def _axis_and_vectors(
    orientation_deg: float,
) -> Tuple[str, Tuple[float, float], Tuple[float, float]]:
    orientation = int(round(float(orientation_deg))) % 360
    outward = {
        0: (1.0, 0.0),
        90: (0.0, 1.0),
        180: (-1.0, 0.0),
        270: (0.0, -1.0),
    }.get(orientation)
    if outward is None:
        raise RunnerError(f"only orthogonal ports are supported, got {orientation_deg}")
    axis = "x" if outward[0] else "y"
    inward = (-outward[0], -outward[1])
    return axis, outward, inward


def plane_spec(port: Mapping[str, Any], plane: Mapping[str, Any]) -> Dict[str, Any]:
    axis, outward, inward = _axis_and_vectors(port["orientation_deg"])
    offset = float(plane.get("plane_offset_from_port_um", 0.0))
    center = [
        float(port["center_um"][0]) + offset * inward[0],
        float(port["center_um"][1]) + offset * inward[1],
        0.0,
    ]
    transverse = float(port["width_um"]) + float(plane["transverse_padding_um"])
    z_span = float(plane["z_span_um"])
    size = [0.0, transverse, z_span] if axis == "x" else [transverse, 0.0, z_span]
    return {
        "center_um": center,
        "size_um": size,
        "axis": axis,
        "outward": list(outward),
        "inward": list(inward),
    }


def validate_planes(
    manifest: Mapping[str, Any], ports: Mapping[str, Any]
) -> Dict[str, Any]:
    case = manifest["case"]
    pml = float(case["boundaries"]["thickness_um"])
    non_pml_half = [0.5 * float(extent) - pml for extent in case["paper_domain_um"]]
    if any(value <= 0 for value in non_pml_half):
        raise RunnerError("PML consumes the entire cell")
    planes: Dict[str, Any] = {
        "source": plane_spec(ports[case["source"]["port"]], case["source"])
    }
    for monitor in case["monitors"]:
        if monitor["kind"] == "mode":
            planes[monitor["name"]] = plane_spec(ports[monitor["port"]], monitor)
    for name, plane in planes.items():
        for dimension, label in enumerate(("x", "y", "z")):
            lower = plane["center_um"][dimension] - 0.5 * plane["size_um"][dimension]
            upper = plane["center_um"][dimension] + 0.5 * plane["size_um"][dimension]
            if (
                lower < -non_pml_half[dimension] - 1e-9
                or upper > non_pml_half[dimension] + 1e-9
            ):
                raise RunnerError(
                    f"{name} plane leaves the non-PML {label} extent: "
                    f"[{lower}, {upper}] versus +/-{non_pml_half[dimension]}"
                )
    return planes


def _mediums(mp: Any, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    constants = manifest["materials"]["performance_adaptation"]
    return {
        name: mp.Medium(epsilon=float(spec["epsilon"]))
        for name, spec in constants.items()
    }


def _vector(mp: Any, values: Sequence[float]) -> Any:
    return mp.Vector3(*values)


def _build_simulation(
    mp: Any,
    manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    planes: Mapping[str, Any],
    *,
    device_id: int,
) -> Tuple[Any, Dict[str, Any]]:
    case = manifest["case"]
    requested = manifest["execution"]["requested"]
    media = _mediums(mp, manifest)
    geometry = []
    for record in records:
        layer = record["layer"]
        vertices = [
            mp.Vector3(x, y, float(layer["z_min_um"])) for x, y in record["points_um"]
        ]
        geometry.append(
            mp.Prism(
                vertices,
                height=float(layer["z_max_um"]) - float(layer["z_min_um"]),
                material=media[layer["material"]],
            )
        )
    source = case["source"]
    source_plane = planes["source"]
    inward = source_plane["inward"]
    fcen = float(source["spectral_envelope"]["center_frequency_meep"])
    eig_guess = mp.Vector3(inward[0] * fcen, inward[1] * fcen)
    sources = [
        mp.EigenModeSource(
            src=mp.GaussianSource(
                fcen, fwidth=float(source["spectral_envelope"]["fwidth_meep"])
            ),
            center=_vector(mp, source_plane["center_um"]),
            size=_vector(mp, source_plane["size_um"]),
            direction=mp.NO_DIRECTION,
            eig_kpoint=eig_guess,
            eig_band=int(source["mode_order"]) + 1,
            eig_parity=mp.NO_PARITY,
            eig_match_freq=True,
        )
    ]
    kwargs = {
        "cell_size": _vector(mp, case["paper_domain_um"]),
        "resolution": float(manifest["discretization"]["resolution_px_per_um"]),
        "Courant": float(case["time_stepping"]["courant_factor"]),
        "boundary_layers": [mp.PML(float(case["boundaries"]["thickness_um"]))],
        "default_material": media[case["cell"]["background_material"]],
        "geometry": geometry,
        "sources": sources,
        "backend": requested["backend"],
        "precision": requested["precision"],
        "accelerator_strict": True,
    }
    if requested["backend"] == "nvidia":
        kwargs["device_id"] = device_id
    simulation = mp.Simulation(**kwargs)
    simulation.init_sim()
    monitor_objects = {}
    frequencies = list(manifest["excitation"]["monitor_sampling"]["frequencies_meep"])
    decimation = int(
        manifest["excitation"]["monitor_sampling"]["dft_decimation_factor"]
    )
    for monitor in case["monitors"]:
        if monitor["kind"] != "mode":
            continue
        plane = planes[monitor["name"]]
        normal = mp.X if plane["axis"] == "x" else mp.Y
        monitor_objects[monitor["name"]] = simulation.add_mode_monitor(
            frequencies,
            mp.ModeRegion(
                center=_vector(mp, plane["center_um"]),
                size=_vector(mp, plane["size_um"]),
                direction=normal,
            ),
            decimation_factor=decimation,
        )
    # fields.advance() deliberately bypasses Simulation.run().  Materialize
    # every delayed DFT object now so the resident backend rebuild sees the
    # complete monitor storage catalog before the timed batch begins.
    simulation._evaluate_dft_objects()
    return simulation, monitor_objects


def _grid_shape(simulation: Any) -> list:
    gv = simulation.fields.gv
    return [int(gv.nx()), int(gv.ny()), int(gv.nz())]


def _cuda_profiler_window(callback: Any) -> None:
    try:
        cudart = ctypes.CDLL("libcudart.so")
    except OSError as error:
        raise RunnerError(
            f"cannot load CUDA profiler API from libcudart: {error}"
        ) from error
    start = cudart.cudaProfilerStart
    stop = cudart.cudaProfilerStop
    start.restype = stop.restype = ctypes.c_int
    if start() != 0:
        raise RunnerError("cudaProfilerStart failed")
    try:
        callback()
    finally:
        if stop() != 0:
            raise RunnerError("cudaProfilerStop failed")


def _monitor_output(
    mp: Any,
    simulation: Any,
    monitor_objects: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> list:
    output = []
    monitor_defs = {
        monitor["name"]: monitor for monitor in manifest["case"]["monitors"]
    }
    for name, monitor_object in monitor_objects.items():
        definition = monitor_defs[name]
        band = int(definition["mode_order"]) + 1
        raw_flux = [
            _finite_number(value, f"{name} DFT flux")
            for value in mp.get_fluxes(monitor_object)
        ]
        coefficients = simulation.get_eigenmode_coefficients(
            monitor_object, [band], eig_parity=mp.NO_PARITY
        ).alpha
        forward = []
        backward = []
        for index in range(len(raw_flux)):
            forward.append(
                _finite_number(
                    abs(coefficients[0, index, 0]) ** 2, f"{name} forward mode power"
                )
            )
            backward.append(
                _finite_number(
                    abs(coefficients[0, index, 1]) ** 2, f"{name} backward mode power"
                )
            )
        output.append(
            {
                "name": name,
                "port": definition["port"],
                "mode_band": band,
                "raw_dft_flux": raw_flux,
                "forward_mode_power": forward,
                "backward_mode_power": backward,
            }
        )
    return output


def _run_once(
    mp: Any,
    manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    planes: Mapping[str, Any],
    *,
    device_id: int,
    steps: int,
    warmup_steps: int,
    profile: bool,
) -> Dict[str, Any]:
    initialized_at = time.perf_counter()
    simulation, monitor_objects = _build_simulation(
        mp, manifest, records, planes, device_id=device_id
    )
    initialization_seconds = time.perf_counter() - initialized_at
    shape = _grid_shape(simulation)
    if warmup_steps:
        simulation.fields.advance(warmup_steps)
    advance = lambda: simulation.fields.advance(steps)
    started = time.perf_counter()
    if profile:
        _cuda_profiler_window(advance)
    else:
        advance()
    elapsed = time.perf_counter() - started
    monitors = _monitor_output(mp, simulation, monitor_objects, manifest)
    result = {
        "initialization_seconds": initialization_seconds,
        "advance_seconds": elapsed,
        "grid_shape": shape,
        "grid_points_exact": math.prod(shape),
        "dt_meep": _finite_number(simulation.fields.dt, "dt"),
        "steps": steps,
        "warmup_steps": warmup_steps,
        "total_steps": warmup_steps + steps,
        "physical_time_meep": (warmup_steps + steps) * float(simulation.fields.dt),
        "monitors": monitors,
    }
    simulation.reset_meep()
    return result


def _nvidia_device_provenance(device_id: int) -> list:
    visible = [
        item.strip()
        for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if item.strip()
    ]
    selector = (
        visible[device_id] if visible and device_id < len(visible) else str(device_id)
    )
    query = [
        "nvidia-smi",
        f"--id={selector}",
        "--query-gpu=index,uuid,name,memory.total,clocks.sm,clocks.mem,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        values = (
            subprocess.run(query, check=True, text=True, stdout=subprocess.PIPE)
            .stdout.strip()
            .split(", ")
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RunnerError(f"cannot query NVIDIA device provenance: {error}") from error
    if len(values) != 7:
        raise RunnerError(f"unexpected nvidia-smi output: {values}")
    return [
        {
            "visible_device": int(values[0]),
            "process_device_id": device_id,
            "physical_selector": selector,
            "uuid": values[1],
            "name": values[2],
            "memory_bytes": int(float(values[3]) * 1024 * 1024),
            "sm_clock_hz": float(values[4]) * 1e6,
            "memory_clock_hz": float(values[5]) * 1e6,
            "driver_version": values[6],
        }
    ]


def build_result(
    manifest_path: pathlib.Path,
    manifest: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    translation: Mapping[str, float],
    planes: Mapping[str, Any],
    *,
    device_id: int,
    profile: bool,
) -> Dict[str, Any]:
    requested = manifest["execution"]["requested"]
    worktree = pathlib.Path(__file__).resolve().parents[3]
    source_tree = pathlib.Path(
        os.environ.get("MEEP_SOURCE_TREE", str(worktree))
    ).resolve()
    build_directory = os.environ.get("MEEP_BUILD_DIR")
    configure_flags = None
    if build_directory:
        config_status = pathlib.Path(build_directory) / "config.status"
        if config_status.is_file():
            configure_flags = _command_output([str(config_status), "--config"])
    timings = [float(run["advance_seconds"]) for run in runs]
    device_records = (
        _nvidia_device_provenance(device_id) if requested["backend"] == "nvidia" else []
    )
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "kind": "paper_2506_16665_single_rank_diagnostic",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "status": {
            "succeeded": True,
            "diagnostic_only": requested["mode"] == "smoke" or profile,
            "profiled": profile,
        },
        "claim_boundary": {
            "single_rank": True,
            "nondispersive_performance_adaptation": True,
            "paper_equivalent_physics": False,
            "multi_gpu_scaling": False,
            "speedup_claim_permitted": requested["mode"] == "fixed-step"
            and not profile,
        },
        "run_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": bm.sha256_file(manifest_path.resolve()),
            "case_id": manifest["case"]["id"],
        },
        "provenance": {
            "meep_build_source": str(source_tree),
            "meep_build_commit": _git(source_tree, "rev-parse", "HEAD"),
            "meep_build_dirty": bool(_git(source_tree, "status", "--porcelain=v1")),
            "runner_commit": _git(worktree, "rev-parse", "HEAD"),
            "runner_dirty": bool(_git(worktree, "status", "--porcelain=v1")),
            "build_directory": build_directory,
            "configure_flags": configure_flags,
            "python": sys.version,
            "meep_module": str(importlib.import_module("meep").__file__),
            "gdstk_version": importlib.import_module("gdstk").__version__,
            "cuda_toolkit": _command_output(["nvcc", "--version"]),
            "argv": list(sys.argv),
            "cwd": str(pathlib.Path.cwd()),
            "environment": {
                name: os.environ[name]
                for name in PROFILE_ENVIRONMENT
                if name in os.environ
            },
            "requested_execution": dict(requested),
            "device_records": device_records,
        },
        "geometry": {
            "translation_um": dict(translation),
            "planes": dict(planes),
            "material_constants": manifest["materials"]["performance_adaptation"],
        },
        "sampling": dict(manifest["excitation"]["monitor_sampling"]),
        "runs": list(runs),
        "timing_summary": {
            "samples_seconds": timings,
            "minimum_seconds": min(timings),
            "median_seconds": statistics.median(timings),
            "maximum_seconds": max(timings),
        },
        "observable_interpretation": (
            "Raw finite DFT flux and mode powers only. Ten-step smoke values are not "
            "normalized transmission, conversion efficiency, resonance, or loss results."
        ),
    }


def validate_runner_result(
    result: Mapping[str, Any], manifest_path: pathlib.Path
) -> None:
    if result.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise RunnerError("runner result schema_version is invalid")
    if result.get("kind") != "paper_2506_16665_single_rank_diagnostic":
        raise RunnerError("runner result kind is invalid")
    manifest_record = result.get("run_manifest", {})
    if manifest_record.get("sha256") != bm.sha256_file(manifest_path.resolve()):
        raise RunnerError("runner result manifest hash is invalid")
    manifest = bm.load_json_object(manifest_path, "run manifest")
    validate_executable_manifest(manifest)
    if manifest_record.get("case_id") != manifest["case"]["id"]:
        raise RunnerError("runner result case ID is invalid")
    runs = result.get("runs")
    if not isinstance(runs, list) or not runs:
        raise RunnerError("runner result must contain runs")
    expected_monitors = {
        monitor["name"]
        for monitor in manifest["case"]["monitors"]
        if monitor["kind"] == "mode"
    }
    expected_frequencies = len(
        manifest["excitation"]["monitor_sampling"]["frequencies_meep"]
    )
    timings = []
    for run_index, run in enumerate(runs):
        shape = run.get("grid_shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or any(int(value) <= 0 for value in shape)
        ):
            raise RunnerError(f"run {run_index} grid shape is invalid")
        if int(run.get("grid_points_exact", 0)) != math.prod(
            int(value) for value in shape
        ):
            raise RunnerError(f"run {run_index} grid point count is invalid")
        for field in (
            "initialization_seconds",
            "advance_seconds",
            "dt_meep",
            "physical_time_meep",
        ):
            if _finite_number(run.get(field), f"run {run_index} {field}") < 0:
                raise RunnerError(f"run {run_index} {field} must be non-negative")
        steps = int(run.get("steps", -1))
        warmup_steps = int(run.get("warmup_steps", -1))
        if steps <= 0 or warmup_steps < 0:
            raise RunnerError(f"run {run_index} step counts are invalid")
        if int(run.get("total_steps", -1)) != steps + warmup_steps:
            raise RunnerError(f"run {run_index} total_steps is invalid")
        if not math.isclose(
            float(run["physical_time_meep"]),
            (steps + warmup_steps) * float(run["dt_meep"]),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise RunnerError(f"run {run_index} physical time is invalid")
        monitors = run.get("monitors", [])
        if {monitor.get("name") for monitor in monitors} != expected_monitors:
            raise RunnerError(f"run {run_index} monitor set is invalid")
        for monitor in monitors:
            for series_name in (
                "raw_dft_flux",
                "forward_mode_power",
                "backward_mode_power",
            ):
                series = monitor.get(series_name)
                if not isinstance(series, list) or len(series) != expected_frequencies:
                    raise RunnerError(
                        f"run {run_index} monitor {series_name} has the wrong length"
                    )
                for value in series:
                    _finite_number(value, f"run {run_index} monitor {series_name}")
        timings.append(float(run["advance_seconds"]))
    summary = result.get("timing_summary", {})
    if summary.get("samples_seconds") != timings:
        raise RunnerError("timing summary samples do not match runs")
    for key, expected in (
        ("minimum_seconds", min(timings)),
        ("median_seconds", statistics.median(timings)),
        ("maximum_seconds", max(timings)),
    ):
        if not math.isclose(float(summary.get(key)), expected, rel_tol=1e-12):
            raise RunnerError(f"timing summary {key} is invalid")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--profile-steps", type=int)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        manifest = bm.load_json_object(args.manifest, "run manifest")
        validate_executable_manifest(manifest)
        if args.validate_only:
            result = bm.load_json_object(args.output, "runner result")
            validate_runner_result(result, args.manifest)
            print(json.dumps({"valid": True, "result": str(args.output.resolve())}))
            return 0
        if args.profile_steps is not None and args.profile_steps <= 0:
            raise RunnerError("--profile-steps must be positive")
        mp, gdstk = _load_runtime_modules()
        records, translation = _geometry_records(manifest, gdstk)
        ports = transformed_ports(manifest, translation)
        planes = validate_planes(manifest, ports)
        requested = manifest["execution"]["requested"]
        steps = int(args.profile_steps or manifest["stopping"]["steps"])
        repetitions = (
            1
            if args.profile_steps
            else int(manifest["execution"]["measured_repetitions"])
        )
        warmup = 1 if args.profile_steps else int(manifest["execution"]["warmup_steps"])
        runs = [
            _run_once(
                mp,
                manifest,
                records,
                planes,
                device_id=args.device_id,
                steps=steps,
                warmup_steps=warmup,
                profile=args.profile_steps is not None,
            )
            for _ in range(repetitions)
        ]
        result = build_result(
            args.manifest,
            manifest,
            runs,
            translation,
            planes,
            device_id=args.device_id,
            profile=args.profile_steps is not None,
        )
        validate_runner_result(result, args.manifest)
        _atomic_json_write(result, args.output)
        return 0
    except (
        bm.ValidationError,
        RunnerError,
        OSError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"run_benchmark.py: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
