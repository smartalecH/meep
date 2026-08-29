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
import hashlib
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
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise RunnerError(f"{label} is not numeric") from error
    if not math.isfinite(number):
        raise RunnerError(f"{label} is not finite")
    return number


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise RunnerError(f"{label} is not an integer")
    try:
        integer = int(value)
    except (TypeError, ValueError) as error:
        raise RunnerError(f"{label} is not an integer") from error
    if integer != value:
        raise RunnerError(f"{label} is not an integer")
    return integer


def _exact_keys(value: Any, expected: Sequence[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RunnerError(f"{label} must be an object")
    actual = set(value)
    required = set(expected)
    if actual != required:
        missing = sorted(required - actual)
        extra = sorted(actual - required)
        raise RunnerError(f"{label} fields are invalid: missing={missing}, extra={extra}")
    return value


def _manifest_snapshot(path: pathlib.Path) -> Tuple[Dict[str, Any], str]:
    """Read, parse, and hash one immutable byte snapshot of a run manifest."""
    try:
        payload = path.resolve().read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RunnerError(f"cannot load run manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise RunnerError("run manifest must contain a JSON object")
    return value, hashlib.sha256(payload).hexdigest()


def _manifest_geometry_inputs(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    case = manifest["case"]
    return {
        "gds": dict(manifest["input_checkout"]["gds"]),
        "geometry_import": dict(case["geometry_import"]),
        "paper_domain_um": list(case["paper_domain_um"]),
        "layers": list(case["layers"]),
        "cell": dict(case["cell"]),
        "boundaries": dict(case["boundaries"]),
        "ports": dict(case["ports"]),
        "source": dict(case["source"]),
        "monitors": list(case["monitors"]),
        "discretization": dict(manifest["discretization"]),
    }


def _expected_grid_shape(manifest: Mapping[str, Any]) -> list:
    resolution = float(manifest["discretization"]["resolution_px_per_um"])
    # Meep explicitly rounds a non-integral cell extent to the nearest pixel.
    return [
        int(round(float(extent) * resolution))
        for extent in manifest["case"]["paper_domain_um"]
    ]


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


def _load_gdstk_module() -> Any:
    try:
        return importlib.import_module("gdstk")
    except ImportError as error:
        raise RunnerError(
            "result validation requires gdstk to authenticate the recorded geometry"
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


def _gds_snapshot(manifest: Mapping[str, Any]) -> bytes:
    """Read and authenticate one immutable snapshot of the selected GDS."""
    gds = manifest["input_checkout"]["gds"]
    path = pathlib.Path(gds["path"])
    try:
        payload = path.resolve().read_bytes()
    except OSError as error:
        raise RunnerError(f"cannot read GDS input {path}: {error}") from error
    if hashlib.sha256(payload).hexdigest() != gds["sha256"]:
        raise RunnerError("GDS hash changed after manifest generation")
    return payload


def _geometry_records_from_snapshot(
    manifest: Mapping[str, Any], gdstk: Any, payload: bytes
) -> Tuple[list, Dict[str, float]]:
    """Parse only the already-authenticated bytes, never the mutable source path."""
    case = manifest["case"]
    if (
        case["geometry_import"]["translation_rule"]
        != "center_extended_geometry_bounds_at_origin"
    ):
        raise RunnerError("the authenticated geometry centering rule is unsupported")
    try:
        with tempfile.NamedTemporaryFile(suffix=".gds") as snapshot:
            snapshot.write(payload)
            snapshot.flush()
            os.fsync(snapshot.fileno())
            cell = _select_gds_cell(
                gdstk, pathlib.Path(snapshot.name), case["gds_cell_name"]
            )
    except OSError as error:
        raise RunnerError(f"cannot materialize authenticated GDS snapshot: {error}") from error

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


def _geometry_records(
    manifest: Mapping[str, Any], gdstk: Any
) -> Tuple[list, Dict[str, float]]:
    return _geometry_records_from_snapshot(manifest, gdstk, _gds_snapshot(manifest))


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
    manifest_sha256: str,
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
            "diagnostic_only": True,
            "profiled": profile,
        },
        "claim_boundary": {
            "single_rank": True,
            "nondispersive_performance_adaptation": True,
            "paper_equivalent_physics": False,
            "multi_gpu_scaling": False,
            "authenticated_cpu_baseline": False,
            "physics_parity_verified": False,
            "speedup_claim_permitted": False,
            "publication_claim_permitted": False,
        },
        "run_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": manifest_sha256,
            "case_id": manifest["case"]["id"],
            "physics_configuration_sha256": manifest["validation_policy"]["reference"][
                "physics_configuration_sha256"
            ],
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
        "execution": {
            "device_id": device_id,
            "profile_steps": int(runs[0]["steps"]) if profile else None,
            "steps": int(runs[0]["steps"]),
            "warmup_steps": int(runs[0]["warmup_steps"]),
            "measured_repetitions": len(runs),
        },
        "geometry": {
            "translation_um": dict(translation),
            "planes": dict(planes),
            "material_constants": manifest["materials"]["performance_adaptation"],
            "manifest_inputs": _manifest_geometry_inputs(manifest),
        },
        "sampling": dict(manifest["excitation"]["monitor_sampling"]),
        "observable_policy": dict(manifest["validation_policy"]),
        "runs": list(runs),
        "timing_summary": {
            "samples_seconds": timings,
            "minimum_seconds": min(timings),
            "median_seconds": statistics.median(timings),
            "maximum_seconds": max(timings),
        },
        "observable_interpretation": (
            "Raw finite DFT flux and mode powers only. These PR3 diagnostics have no "
            "authenticated CPU baseline or verified physics parity and cannot support "
            "speedup or publication claims. Ten-step smoke values are not normalized "
            "transmission, conversion efficiency, resonance, or loss results."
        ),
    }


def validate_runner_result(
    result: Mapping[str, Any],
    manifest_path: pathlib.Path,
    *,
    manifest: Optional[Mapping[str, Any]] = None,
    manifest_sha256: Optional[str] = None,
    authenticated_translation: Optional[Mapping[str, float]] = None,
) -> None:
    _exact_keys(
        result,
        (
            "schema_version",
            "kind",
            "generated_at_utc",
            "status",
            "claim_boundary",
            "run_manifest",
            "provenance",
            "execution",
            "geometry",
            "sampling",
            "observable_policy",
            "runs",
            "timing_summary",
            "observable_interpretation",
        ),
        "runner result",
    )
    if result["schema_version"] != RESULT_SCHEMA_VERSION:
        raise RunnerError("runner result schema_version is invalid")
    if result["kind"] != "paper_2506_16665_single_rank_diagnostic":
        raise RunnerError("runner result kind is invalid")
    try:
        generated = dt.datetime.fromisoformat(str(result["generated_at_utc"]))
    except ValueError as error:
        raise RunnerError("runner result generation time is invalid") from error
    if generated.tzinfo is None or generated.utcoffset() != dt.timedelta(0):
        raise RunnerError("runner result generation time is not UTC")

    if manifest is None or manifest_sha256 is None:
        manifest, manifest_sha256 = _manifest_snapshot(manifest_path)
    validate_executable_manifest(manifest)
    manifest_record = _exact_keys(
        result["run_manifest"],
        ("path", "sha256", "case_id", "physics_configuration_sha256"),
        "run_manifest",
    )
    if manifest_record["path"] != str(manifest_path.resolve()):
        raise RunnerError("runner result manifest path is invalid")
    if manifest_record["sha256"] != manifest_sha256:
        raise RunnerError("runner result manifest hash is invalid")
    if manifest_record["case_id"] != manifest["case"]["id"]:
        raise RunnerError("runner result case ID is invalid")
    expected_physics_hash = manifest["validation_policy"]["reference"][
        "physics_configuration_sha256"
    ]
    if manifest_record["physics_configuration_sha256"] != expected_physics_hash:
        raise RunnerError("runner result physics configuration is invalid")

    status = _exact_keys(
        result["status"], ("succeeded", "diagnostic_only", "profiled"), "status"
    )
    if status["succeeded"] is not True or status["diagnostic_only"] is not True:
        raise RunnerError("runner result status is not a successful diagnostic")
    if type(status["profiled"]) is not bool:
        raise RunnerError("runner result profiled status is invalid")
    claims = _exact_keys(
        result["claim_boundary"],
        (
            "single_rank",
            "nondispersive_performance_adaptation",
            "paper_equivalent_physics",
            "multi_gpu_scaling",
            "authenticated_cpu_baseline",
            "physics_parity_verified",
            "speedup_claim_permitted",
            "publication_claim_permitted",
        ),
        "claim_boundary",
    )
    expected_claims = {
        "single_rank": True,
        "nondispersive_performance_adaptation": True,
        "paper_equivalent_physics": False,
        "multi_gpu_scaling": False,
        "authenticated_cpu_baseline": False,
        "physics_parity_verified": False,
        "speedup_claim_permitted": False,
        "publication_claim_permitted": False,
    }
    if dict(claims) != expected_claims:
        raise RunnerError(
            "PR3 diagnostics cannot claim speedup or publication without an "
            "authenticated compatible CPU baseline and verified physics parity"
        )

    requested = manifest["execution"]["requested"]
    provenance = _exact_keys(
        result["provenance"],
        (
            "meep_build_source",
            "meep_build_commit",
            "meep_build_dirty",
            "runner_commit",
            "runner_dirty",
            "build_directory",
            "configure_flags",
            "python",
            "meep_module",
            "gdstk_version",
            "cuda_toolkit",
            "argv",
            "cwd",
            "environment",
            "requested_execution",
            "device_records",
        ),
        "provenance",
    )
    if provenance["requested_execution"] != requested:
        raise RunnerError("provenance requested execution does not match the manifest")
    for label in ("meep_build_commit", "runner_commit"):
        value = provenance[label]
        if not isinstance(value, str) or len(value) != 40 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise RunnerError(f"provenance {label} is invalid")
    for label in ("meep_build_dirty", "runner_dirty"):
        if type(provenance[label]) is not bool:
            raise RunnerError(f"provenance {label} is invalid")
    for label in ("meep_build_source", "python", "meep_module", "gdstk_version", "cwd"):
        if not isinstance(provenance[label], str) or not provenance[label]:
            raise RunnerError(f"provenance {label} is invalid")
    for label in ("build_directory", "configure_flags", "cuda_toolkit"):
        if provenance[label] is not None and not isinstance(provenance[label], str):
            raise RunnerError(f"provenance {label} is invalid")
    if not isinstance(provenance["argv"], list) or any(
        not isinstance(value, str) for value in provenance["argv"]
    ):
        raise RunnerError("provenance argv is invalid")
    environment = provenance["environment"]
    if not isinstance(environment, Mapping) or any(
        name not in PROFILE_ENVIRONMENT or not isinstance(value, str)
        for name, value in environment.items()
    ):
        raise RunnerError("provenance environment is invalid")

    execution = _exact_keys(
        result["execution"],
        ("device_id", "profile_steps", "steps", "warmup_steps", "measured_repetitions"),
        "execution",
    )
    device_id = _integer(execution["device_id"], "execution device_id")
    if device_id < 0:
        raise RunnerError("execution device_id is invalid")
    profiled = status["profiled"]
    if profiled:
        if (
            execution["profile_steps"] != execution["steps"]
            or _integer(execution["steps"], "execution steps") <= 0
        ):
            raise RunnerError("profile execution step override is invalid")
        expected_steps = _integer(execution["profile_steps"], "execution profile_steps")
        expected_warmup = 1
        expected_repetitions = 1
    else:
        if execution["profile_steps"] is not None:
            raise RunnerError("ordinary execution records an unauthorized profile override")
        expected_steps = int(manifest["stopping"]["steps"])
        expected_warmup = int(manifest["execution"]["warmup_steps"])
        expected_repetitions = int(manifest["execution"]["measured_repetitions"])
    if (
        _integer(execution["steps"], "execution steps") != expected_steps
        or _integer(execution["warmup_steps"], "execution warmup_steps")
        != expected_warmup
        or _integer(
            execution["measured_repetitions"], "execution measured_repetitions"
        )
        != expected_repetitions
    ):
        raise RunnerError("execution semantics do not match the manifest/profile mode")

    device_records = provenance["device_records"]
    if not isinstance(device_records, list):
        raise RunnerError("provenance device records are invalid")
    if requested["backend"] == "cpu":
        if device_records:
            raise RunnerError("CPU execution cannot record an NVIDIA device")
    else:
        if len(device_records) != 1:
            raise RunnerError("single-rank NVIDIA execution requires exactly one device record")
        device = _exact_keys(
            device_records[0],
            (
                "visible_device",
                "process_device_id",
                "physical_selector",
                "uuid",
                "name",
                "memory_bytes",
                "sm_clock_hz",
                "memory_clock_hz",
                "driver_version",
            ),
            "NVIDIA device record",
        )
        if _integer(device["process_device_id"], "NVIDIA process_device_id") != device_id:
            raise RunnerError("NVIDIA device record does not match execution device_id")
        if (
            _integer(device["visible_device"], "NVIDIA visible_device") < 0
            or _integer(device["memory_bytes"], "NVIDIA memory_bytes") <= 0
        ):
            raise RunnerError("NVIDIA device identity or memory is invalid")
        for label in ("sm_clock_hz", "memory_clock_hz"):
            if _finite_number(device[label], f"NVIDIA {label}") <= 0:
                raise RunnerError(f"NVIDIA {label} must be positive")
        for label in ("physical_selector", "uuid", "name", "driver_version"):
            if not isinstance(device[label], str) or not device[label]:
                raise RunnerError(f"NVIDIA {label} is invalid")

    geometry = _exact_keys(
        result["geometry"],
        ("translation_um", "planes", "material_constants", "manifest_inputs"),
        "geometry",
    )
    translation = _exact_keys(
        geometry["translation_um"], ("x_um", "y_um"), "geometry translation"
    )
    for axis in ("x_um", "y_um"):
        _finite_number(translation[axis], f"geometry translation {axis}")
    if geometry["manifest_inputs"] != _manifest_geometry_inputs(manifest):
        raise RunnerError("recorded geometry inputs do not match the manifest")
    if geometry["material_constants"] != manifest["materials"]["performance_adaptation"]:
        raise RunnerError("recorded material constants do not match the manifest")
    if authenticated_translation is None:
        _, authenticated_translation = _geometry_records(
            manifest, _load_gdstk_module()
        )
    for axis in ("x_um", "y_um"):
        expected = _finite_number(
            authenticated_translation[axis], f"authenticated geometry translation {axis}"
        )
        actual = _finite_number(translation[axis], f"geometry translation {axis}")
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise RunnerError(
                "recorded geometry translation does not match authenticated GDS bounds "
                "and the manifest centering rule"
            )
    expected_planes = validate_planes(manifest, transformed_ports(manifest, translation))
    if geometry["planes"] != expected_planes:
        raise RunnerError("recorded geometry planes do not match the manifest and transform")
    if result["sampling"] != manifest["excitation"]["monitor_sampling"]:
        raise RunnerError("recorded sampling does not match the manifest")
    if result["observable_policy"] != manifest["validation_policy"]:
        raise RunnerError("recorded observable policy does not match the manifest")
    expected_interpretation = (
        "Raw finite DFT flux and mode powers only. These PR3 diagnostics have no "
        "authenticated CPU baseline or verified physics parity and cannot support "
        "speedup or publication claims. Ten-step smoke values are not normalized "
        "transmission, conversion efficiency, resonance, or loss results."
    )
    if result["observable_interpretation"] != expected_interpretation:
        raise RunnerError("observable interpretation is invalid")

    runs = result["runs"]
    if not isinstance(runs, list) or len(runs) != expected_repetitions:
        raise RunnerError("runner result repetition count is invalid")
    expected_monitor_defs = [
        monitor
        for monitor in manifest["case"]["monitors"]
        if monitor["kind"] == "mode"
    ]
    expected_monitor_names = [monitor["name"] for monitor in expected_monitor_defs]
    expected_frequencies = len(
        manifest["excitation"]["monitor_sampling"]["frequencies_meep"]
    )
    expected_shape = _expected_grid_shape(manifest)
    expected_dt = float(manifest["case"]["time_stepping"]["courant_factor"]) / float(
        manifest["discretization"]["resolution_px_per_um"]
    )
    timings = []
    for run_index, run in enumerate(runs):
        _exact_keys(
            run,
            (
                "initialization_seconds",
                "advance_seconds",
                "grid_shape",
                "grid_points_exact",
                "dt_meep",
                "steps",
                "warmup_steps",
                "total_steps",
                "physical_time_meep",
                "monitors",
            ),
            f"run {run_index}",
        )
        shape = run["grid_shape"]
        if shape != expected_shape:
            raise RunnerError(f"run {run_index} grid shape does not match the manifest")
        if _integer(run["grid_points_exact"], f"run {run_index} grid_points_exact") != math.prod(
            expected_shape
        ):
            raise RunnerError(f"run {run_index} grid point count is invalid")
        for field in (
            "initialization_seconds",
            "advance_seconds",
            "dt_meep",
            "physical_time_meep",
        ):
            if _finite_number(run[field], f"run {run_index} {field}") < 0:
                raise RunnerError(f"run {run_index} {field} must be non-negative")
        if not math.isclose(float(run["dt_meep"]), expected_dt, rel_tol=1e-12, abs_tol=1e-15):
            raise RunnerError(f"run {run_index} timestep does not match the manifest")
        steps = _integer(run["steps"], f"run {run_index} steps")
        warmup_steps = _integer(run["warmup_steps"], f"run {run_index} warmup_steps")
        if steps != expected_steps or warmup_steps != expected_warmup:
            raise RunnerError(f"run {run_index} step counts do not match execution semantics")
        if _integer(run["total_steps"], f"run {run_index} total_steps") != steps + warmup_steps:
            raise RunnerError(f"run {run_index} total_steps is invalid")
        if not math.isclose(
            float(run["physical_time_meep"]),
            (steps + warmup_steps) * float(run["dt_meep"]),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise RunnerError(f"run {run_index} physical time is invalid")
        monitors = run["monitors"]
        if not isinstance(monitors, list) or any(
            not isinstance(monitor, Mapping) for monitor in monitors
        ):
            raise RunnerError(f"run {run_index} monitors are invalid")
        if [monitor.get("name") for monitor in monitors] != expected_monitor_names:
            raise RunnerError(f"run {run_index} monitor set is invalid")
        for monitor, definition in zip(monitors, expected_monitor_defs):
            _exact_keys(
                monitor,
                (
                    "name",
                    "port",
                    "mode_band",
                    "raw_dft_flux",
                    "forward_mode_power",
                    "backward_mode_power",
                ),
                f"run {run_index} monitor {definition['name']}",
            )
            if (
                monitor["name"] != definition["name"]
                or monitor["port"] != definition["port"]
                or _integer(
                    monitor["mode_band"], f"run {run_index} monitor mode_band"
                )
                != int(definition["mode_order"]) + 1
            ):
                raise RunnerError(f"run {run_index} monitor identity is invalid")
            for series_name in (
                "raw_dft_flux",
                "forward_mode_power",
                "backward_mode_power",
            ):
                series = monitor[series_name]
                if not isinstance(series, list) or len(series) != expected_frequencies:
                    raise RunnerError(
                        f"run {run_index} monitor {series_name} has the wrong length"
                    )
                for value in series:
                    _finite_number(value, f"run {run_index} monitor {series_name}")
        timings.append(float(run["advance_seconds"]))
    summary = _exact_keys(
        result["timing_summary"],
        ("samples_seconds", "minimum_seconds", "median_seconds", "maximum_seconds"),
        "timing_summary",
    )
    if summary["samples_seconds"] != timings:
        raise RunnerError("timing summary samples do not match runs")
    for key, expected in (
        ("minimum_seconds", min(timings)),
        ("median_seconds", statistics.median(timings)),
        ("maximum_seconds", max(timings)),
    ):
        if not math.isclose(_finite_number(summary[key], f"timing summary {key}"), expected, rel_tol=1e-12):
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
        manifest, manifest_sha256 = _manifest_snapshot(args.manifest)
        validate_executable_manifest(manifest)
        if args.validate_only:
            result = bm.load_json_object(args.output, "runner result")
            validate_runner_result(
                result,
                args.manifest,
                manifest=manifest,
                manifest_sha256=manifest_sha256,
            )
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
            manifest_sha256,
            manifest,
            runs,
            translation,
            planes,
            device_id=args.device_id,
            profile=args.profile_steps is not None,
        )
        validate_runner_result(
            result,
            args.manifest,
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            authenticated_translation=translation,
        )
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
