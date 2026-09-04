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
import re
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import benchmark_manifest as bm


RESULT_SCHEMA_VERSION = 2
PROFILE_ENVIRONMENT = (
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "MEEP_ACCELERATOR_RUNTIME",
    "MEEP_FINITE_CHECK",
    "MEEP_GPU_AWARE_MPI",
    "MEEP_NVIDIA_GRAPH_MODE",
    "MEEP_NVIDIA_MPI_OVERLAP",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MEEP_SOURCE_TREE",
    "MEEP_BUILD_DIR",
)

# These are monotonic integer counters in ``execution_runtime_report``.  Store
# both endpoints for every timed window: a claimed delta is not evidence unless
# it can be derived again from the authenticated snapshots.
RUNTIME_COUNTER_NAMES = (
    "messages_sent",
    "messages_received",
    "bytes_sent",
    "bytes_received",
    "gather_launches",
    "scatter_launches",
    "testsome_polls",
    "waitall_calls",
    "request_completions",
    "slot_reuses",
    "device_to_host_calls",
    "device_to_host_bytes",
    "host_to_device_calls",
    "host_to_device_bytes",
    "direct_bytes",
    "overlap_stages",
    "overlap_interior_launches",
    "overlap_boundary_launches",
    "material_recipe_prepare_nanoseconds",
    "material_initialize_nanoseconds",
    "graph_build_nanoseconds",
    "gather_pack_nanoseconds",
    "device_to_host_nanoseconds",
    "mpi_progress_nanoseconds",
    "mpi_wait_nanoseconds",
    "host_to_device_nanoseconds",
    "scatter_unpack_nanoseconds",
    "steady_allocation_count",
    "graph_recapture_count",
    "full_field_copy_count",
    "graph_capture_count",
    "graph_launch_count",
    "graph_boundary_count",
    "host_fallback_count",
    "host_fallback_device_to_host_bytes",
    "host_fallback_host_to_device_bytes",
    "host_fallback_steady_capacity_growths",
    "material_fallback_warning_count",
)
FORBIDDEN_MEASURED_COUNTERS = (
    "steady_allocation_count",
    "graph_recapture_count",
    "full_field_copy_count",
    "host_fallback_count",
    "host_fallback_device_to_host_bytes",
    "host_fallback_host_to_device_bytes",
    "host_fallback_steady_capacity_growths",
    "material_fallback_warning_count",
)
RUNTIME_MEMORY_NAMES = (
    "process_device_bytes_current",
    "process_device_bytes_peak",
    "process_pinned_bytes_current",
    "process_pinned_bytes_peak",
    "transport_device_bytes",
    "transport_pinned_bytes",
)
RUNTIME_STRING_NAMES = (
    "requested_backend",
    "resolved_backend",
    "requested_precision",
    "resolved_precision",
    "requested_transport",
    "resolved_transport",
    "requested_overlap",
    "resolved_overlap",
    "captured_requested_transport",
    "captured_overlap_policy",
    "requested_graph",
    "resolved_graph",
    "mpi_provider",
    "counter_scope",
    "backend_counter_scope",
    "memory_gauge_scope",
    "setup_counter_scope",
    "transport_timing_scope",
    "allocation_counter_scope",
    "device_uuid",
)
RUNTIME_BOOL_NAMES = (
    "mpi_query_available",
    "mpi_cuda_aware",
    "device_owner",
    "graph_enabled",
    "captured_transport_epoch_active",
    "captured_transport_epoch_fresh",
    "graph_valid",
)
RUNTIME_INTEGER_NAMES = (
    "communicator_rank",
    "communicator_size",
    "device_id",
    "communicator_generation",
    "captured_provider_signature",
    "executable_build_count",
    "high_water_requests",
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
        raise RunnerError(
            f"{label} fields are invalid: missing={missing}, extra={extra}"
        )
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


def _source_tree_state(root: pathlib.Path) -> Dict[str, Any]:
    root = root.resolve()

    def git_bytes(*arguments: str) -> bytes:
        try:
            return subprocess.run(
                ["git", "-C", str(root), *arguments], check=True, capture_output=True
            ).stdout
        except (OSError, subprocess.CalledProcessError) as error:
            raise RunnerError(
                f"cannot authenticate source tree {root}: {error}"
            ) from error

    commit = git_bytes("rev-parse", "HEAD").decode("ascii").strip()
    status = git_bytes("status", "--porcelain=v1", "--untracked-files=all", "-z")
    clean = not status
    diff_sha256 = None
    if not clean:
        digest = hashlib.sha256()
        digest.update(b"tracked-diff\0")
        digest.update(git_bytes("diff", "--binary", "HEAD", "--", "."))
        untracked = git_bytes("ls-files", "--others", "--exclude-standard", "-z")
        for encoded_path in sorted(item for item in untracked.split(b"\0") if item):
            path = root / encoded_path.decode("utf-8", errors="surrogateescape")
            digest.update(b"\0untracked-path\0")
            digest.update(encoded_path)
            digest.update(b"\0untracked-content\0")
            digest.update(
                os.readlink(path).encode("utf-8", errors="surrogateescape")
                if path.is_symlink()
                else path.read_bytes()
            )
        diff_sha256 = digest.hexdigest()
    return {"commit": commit, "dirty": not clean, "diff_sha256": diff_sha256}


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


def _required_command_output(command: Sequence[str], label: str) -> str:
    output = _command_output(command)
    if not output:
        raise RunnerError(f"cannot record {label}: {' '.join(command)}")
    return output


def _file_provenance(path: pathlib.Path) -> Dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise RunnerError(f"provenance file does not exist: {resolved}")
    return {"path": str(resolved), "sha256": bm.sha256_file(resolved)}


def _host_peak_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB while macOS reports bytes.
    return peak if sys.platform == "darwin" else peak * 1024


def _counter_snapshot(runtime: Mapping[str, Any]) -> Dict[str, int]:
    if not isinstance(runtime, Mapping):
        raise RunnerError("execution runtime report is missing")
    missing = sorted(set(RUNTIME_COUNTER_NAMES) - set(runtime))
    if missing:
        raise RunnerError(f"execution runtime report lacks counters: {missing}")
    result = {}
    for name in RUNTIME_COUNTER_NAMES:
        value = runtime[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RunnerError(f"execution runtime counter {name} is invalid")
        result[name] = value
    return result


def _runtime_memory(runtime: Mapping[str, Any]) -> Dict[str, int]:
    result = {}
    for name in RUNTIME_MEMORY_NAMES:
        value = runtime.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RunnerError(f"execution runtime memory gauge {name} is invalid")
        result[name] = value
    return result


def _normalize_gpu_uuid(value: Any) -> str:
    if not isinstance(value, str):
        raise RunnerError("GPU UUID must be a string")
    normalized = value.strip().lower()
    if normalized.startswith("gpu-"):
        normalized = normalized[4:]
    normalized = normalized.replace("-", "")
    if len(normalized) != 32 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise RunnerError(f"GPU UUID is not canonicalizable: {value!r}")
    return normalized


def _normalize_pci_bus_id(value: Any) -> str:
    if not isinstance(value, str):
        raise RunnerError("PCI BDF must be a string")
    normalized = value.strip().lower()
    if re.fullmatch(r"[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", normalized):
        normalized = "0000:" + normalized
    if re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", normalized) is None:
        raise RunnerError(f"invalid PCI BDF {value!r}")
    return normalized


def _hip_pci_bus_id(device_id: int) -> str:
    library = os.environ.get("MEEP_HIP_RUNTIME_LIBRARY", "libamdhip64.so")
    try:
        hip = ctypes.CDLL(library)
    except OSError as error:
        raise RunnerError(f"cannot load HIP runtime {library}: {error}") from error
    get_bus_id = hip.hipDeviceGetPCIBusId
    get_bus_id.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    get_bus_id.restype = ctypes.c_int
    value = ctypes.create_string_buffer(32)
    status = get_bus_id(value, len(value), device_id)
    if status != 0:
        raise RunnerError(
            f"hipDeviceGetPCIBusId({device_id}) failed with status {status}"
        )
    return _normalize_pci_bus_id(value.value.decode("ascii"))


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
    cell = manifest["case"]["cell"]
    if (
        cell.get("epsilon_averaging") is not False
        or cell.get("material_discretization") != "yee_grid_point_staircased"
    ):
        raise RunnerError(
            "the executable runner requires manifest-bound staircased "
            "Yee-grid-point material assignment"
        )
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


def _canonicalize_prism_points(
    points: Sequence[Sequence[float]],
) -> list:
    if len(points) < 3:
        raise RunnerError("a Prism polygon must contain at least three vertices")
    centroid_x = sum(float(point[0]) for point in points) / len(points)
    centroid_y = sum(float(point[1]) for point in points) / len(points)
    return [
        [
            centroid_x + (float(point[0]) - centroid_x),
            centroid_y + (float(point[1]) - centroid_y),
        ]
        for point in points
    ]


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
    if (
        case["geometry_import"].get("prism_vertex_canonicalization")
        != bm.PRISM_VERTEX_CANONICALIZATION
    ):
        raise RunnerError("the authenticated Prism canonicalization is unsupported")
    try:
        with tempfile.NamedTemporaryFile(suffix=".gds") as snapshot:
            snapshot.write(payload)
            snapshot.flush()
            os.fsync(snapshot.fileno())
            cell = _select_gds_cell(
                gdstk, pathlib.Path(snapshot.name), case["gds_cell_name"]
            )
    except OSError as error:
        raise RunnerError(
            f"cannot materialize authenticated GDS snapshot: {error}"
        ) from error

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
        translated = [
            [point[0] + translation["x_um"], point[1] + translation["y_um"]]
            for point in record["points_um"]
        ]
        record["points_um"] = _canonicalize_prism_points(translated)
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
        "eps_averaging": bool(case["cell"]["epsilon_averaging"]),
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


def _accelerator_profiler_window(callback: Any, accelerator: str) -> None:
    library_name, start_name, stop_name = {
        "cuda": ("libcudart.so", "cudaProfilerStart", "cudaProfilerStop"),
        "hip": ("libamdhip64.so", "hipProfilerStart", "hipProfilerStop"),
    }[accelerator]
    try:
        runtime = ctypes.CDLL(library_name)
    except OSError as error:
        raise RunnerError(
            f"cannot load {accelerator.upper()} profiler API from {library_name}: {error}"
        ) from error
    start = getattr(runtime, start_name)
    stop = getattr(runtime, stop_name)
    start.restype = stop.restype = ctypes.c_int
    if start() != 0:
        raise RunnerError(f"{start_name} failed")
    try:
        callback()
    finally:
        if stop() != 0:
            raise RunnerError(f"{stop_name} failed")


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


def _run_session(
    mp: Any,
    manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    planes: Mapping[str, Any],
    *,
    device_id: int,
    steps: int,
    warmup_steps: int,
    repetitions: int,
    profile: bool,
) -> Tuple[list, Mapping[str, Any], Dict[str, int]]:
    initialized_at = time.perf_counter()
    simulation, monitor_objects = _build_simulation(
        mp, manifest, records, planes, device_id=device_id
    )
    initialization_seconds = time.perf_counter() - initialized_at
    shape = _grid_shape(simulation)
    requested = manifest["execution"]["requested"]
    if warmup_steps:
        warmup_started = time.perf_counter()
        simulation.fields.advance(warmup_steps)
        mp.all_wait()
        warmup_seconds = time.perf_counter() - warmup_started
    else:
        warmup_seconds = 0.0

    runs = []
    final_runtime = None
    try:
        for index in range(repetitions):
            start_step = int(simulation.fields.t)
            before = simulation.get_execution_runtime_report()
            host_memory_start = _host_peak_bytes()
            advance = lambda: simulation.fields.advance(steps)
            started = time.perf_counter()
            if profile and requested["backend"] == "nvidia":
                _accelerator_profiler_window(
                    advance, os.environ.get("MEEP_ACCELERATOR_RUNTIME", "cuda")
                )
            else:
                advance()
            mp.all_wait()
            elapsed = time.perf_counter() - started
            after = simulation.get_execution_runtime_report()
            host_memory_end = _host_peak_bytes()
            end_step = int(simulation.fields.t)
            start_counters = _counter_snapshot(before)
            end_counters = _counter_snapshot(after)
            deltas = {
                name: end_counters[name] - start_counters[name]
                for name in RUNTIME_COUNTER_NAMES
            }
            if any(value < 0 for value in deltas.values()):
                raise RunnerError("a runtime counter decreased during measured work")
            runs.append(
                {
                    "initialization_seconds": (
                        initialization_seconds if index == 0 else 0.0
                    ),
                    "warmup_seconds": warmup_seconds if index == 0 else 0.0,
                    "advance_seconds": elapsed,
                    "grid_shape": shape,
                    "grid_points_exact": math.prod(shape),
                    "dt_meep": _finite_number(simulation.fields.dt, "dt"),
                    "steps": steps,
                    "warmup_steps": warmup_steps if index == 0 else 0,
                    "start_step": start_step,
                    "end_step": end_step,
                    "physical_time_meep": end_step * float(simulation.fields.dt),
                    "counter_start": start_counters,
                    "counter_end": end_counters,
                    "counter_deltas": deltas,
                    "memory_start": {
                        "host_peak_bytes": host_memory_start,
                        **_runtime_memory(before),
                    },
                    "memory_end": {
                        "host_peak_bytes": host_memory_end,
                        **_runtime_memory(after),
                    },
                    "monitors": _monitor_output(
                        mp, simulation, monitor_objects, manifest
                    ),
                }
            )
            final_runtime = simulation.get_execution_runtime_report()
        if final_runtime is None:
            raise RunnerError("benchmark session did not execute a timed window")
        memory = {
            "host_peak_bytes": _host_peak_bytes(),
            **_runtime_memory(final_runtime),
        }
        return runs, dict(final_runtime), memory
    finally:
        simulation.reset_meep()


def _visible_selectors(accelerator: str, device_id: int) -> list[str]:
    variable = (
        "ROCR_VISIBLE_DEVICES" if accelerator == "hip" else "CUDA_VISIBLE_DEVICES"
    )
    visible = [
        item.strip() for item in os.environ.get(variable, "").split(",") if item.strip()
    ]
    if accelerator == "hip" and not visible:
        raise RunnerError(
            "HIP execution requires an explicit ROCR_VISIBLE_DEVICES selector"
        )
    if visible and (len(visible) != 1 or device_id != 0):
        raise RunnerError(
            f"single-rank execution requires one {variable} selector and process device id zero"
        )
    return visible or [str(device_id)]


def _nvidia_device_provenance(
    device_id: int, selectors: Optional[Sequence[str]] = None
) -> list:
    visible = (
        list(selectors)
        if selectors is not None
        else _visible_selectors("cuda", device_id)
    )
    selector = (
        visible[device_id] if visible and device_id < len(visible) else str(device_id)
    )
    query = [
        "nvidia-smi",
        f"--id={selector}",
        "--query-gpu=index,uuid,name,pci.bus_id,memory.total,clocks.sm,clocks.mem,driver_version",
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
    if len(values) != 8:
        raise RunnerError(f"unexpected nvidia-smi output: {values}")
    return [
        {
            "accelerator": "cuda",
            "visible_device": device_id,
            "visible_devices": visible,
            "process_device_id": device_id,
            "physical_selector": selector,
            "inventory_index": int(values[0]),
            "uuid": values[1],
            "name": values[2],
            "pci_bus_id": _normalize_pci_bus_id(values[3]),
            "memory_bytes": int(float(values[4]) * 1024 * 1024),
            "core_clock": values[5] + " MHz",
            "memory_clock": values[6] + " MHz",
            "driver_version": values[7],
        }
    ]


def _rocm_device_provenance(
    device_id: int,
    runtime_uuid: str,
    selectors: Sequence[str],
    rocm_smi: pathlib.Path,
) -> Tuple[list, Dict[str, str]]:
    if device_id < 0 or device_id >= len(selectors):
        raise RunnerError("HIP device id is outside the process-visible selector set")
    runtime_bus_id = _hip_pci_bus_id(device_id)
    try:
        raw_inventory = subprocess.run(
            [
                str(rocm_smi),
                "--showuniqueid",
                "--showbus",
                "--showproductname",
                "--showtoponuma",
                "--showmeminfo",
                "vram",
                "--showclocks",
                "--showdriverversion",
                "--json",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
        inventory = json.loads(raw_inventory)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        raise RunnerError(f"cannot query ROCm device provenance: {error}") from error
    matches = [
        (name, entry)
        for name, entry in inventory.items()
        if isinstance(entry, Mapping)
        and entry.get("PCI Bus")
        and _normalize_pci_bus_id(str(entry["PCI Bus"])) == runtime_bus_id
    ]
    if len(matches) != 1:
        raise RunnerError(
            f"HIP PCI BDF {runtime_bus_id} matched {len(matches)} rocm-smi devices"
        )
    inventory_card, entry = matches[0]

    def first_matching(*fragments: str) -> str:
        for key, value in entry.items():
            lowered = str(key).lower()
            if all(fragment in lowered for fragment in fragments):
                return str(value)
        return ""

    memory_match = re.search(r"([0-9]+)", first_matching("vram", "total"))
    record = {
        "accelerator": "hip",
        "visible_device": device_id,
        "visible_devices": list(selectors),
        "process_device_id": device_id,
        "physical_selector": str(selectors[device_id]),
        "inventory_card": inventory_card,
        "uuid": runtime_uuid,
        "physical_unique_id": str(entry.get("Unique ID", "")),
        "pci_bus_id": runtime_bus_id,
        "numa_node": int(entry.get("(Topology) Numa Node", -1)),
        "name": str(entry.get("Card Series", "")),
        "architecture": str(entry.get("GFX Version", "")),
        "memory_bytes": int(memory_match.group(1)) if memory_match else 0,
        "core_clock": first_matching("sclk", "clock"),
        "memory_clock": first_matching("mclk", "clock"),
        "driver_version": str(
            inventory.get("system", {}).get("Driver version", first_matching("driver"))
        ),
    }
    return [record], {
        "sha256": hashlib.sha256(raw_inventory.encode("utf-8")).hexdigest(),
        "output": raw_inventory,
    }


def _toolchain_provenance(
    accelerator: str, compiler: Optional[pathlib.Path]
) -> Dict[str, str]:
    selected = compiler or pathlib.Path("hipcc" if accelerator == "hip" else "nvcc")
    if not selected.is_absolute():
        resolved = shutil.which(str(selected))
        if not resolved:
            raise RunnerError(f"cannot find {accelerator} compiler {selected}")
        selected = pathlib.Path(resolved)
    selected = selected.resolve()
    return {
        "path": str(selected),
        "sha256": bm.sha256_file(selected),
        "version": _required_command_output(
            [str(selected), "--version"], f"{accelerator} toolchain"
        ),
    }


def _accelerator_device_provenance(
    accelerator: str,
    device_id: int,
    runtime_uuid: str,
    selectors: Sequence[str],
    rocm_smi: Optional[pathlib.Path],
) -> Tuple[list, Optional[Dict[str, str]]]:
    if accelerator == "hip":
        if rocm_smi is None:
            raise RunnerError("HIP provenance requires --rocm-smi")
        return _rocm_device_provenance(
            device_id, runtime_uuid, selectors, rocm_smi.resolve()
        )
    return _nvidia_device_provenance(device_id, selectors), None


def _validate_runtime_report(
    runtime: Mapping[str, Any],
    requested: Mapping[str, Any],
    accelerator: Optional[str],
    device_id: int,
) -> None:
    expected_keys = (
        set(RUNTIME_STRING_NAMES)
        | set(RUNTIME_BOOL_NAMES)
        | set(RUNTIME_INTEGER_NAMES)
        | set(RUNTIME_COUNTER_NAMES)
        | set(RUNTIME_MEMORY_NAMES)
    )
    _exact_keys(runtime, sorted(expected_keys), "execution runtime report")
    for name in RUNTIME_STRING_NAMES:
        if not isinstance(runtime[name], str):
            raise RunnerError(f"execution runtime field {name} is invalid")
    for name in RUNTIME_BOOL_NAMES:
        if type(runtime[name]) is not bool:
            raise RunnerError(f"execution runtime field {name} is invalid")
    for name in (
        set(RUNTIME_INTEGER_NAMES)
        | set(RUNTIME_COUNTER_NAMES)
        | set(RUNTIME_MEMORY_NAMES)
    ):
        value = runtime[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise RunnerError(f"execution runtime field {name} is invalid")
        if name != "device_id" and value < 0:
            raise RunnerError(f"execution runtime field {name} is negative")
    if runtime["communicator_rank"] != 0 or runtime["communicator_size"] != 1:
        raise RunnerError("single-rank runtime communicator identity is invalid")
    if (
        runtime["counter_scope"] != "rank_local_current_epoch"
        or runtime["backend_counter_scope"] != "rank_local_backend_lifetime"
        or runtime["memory_gauge_scope"] != "rank_local_process_lifetime"
        or runtime["setup_counter_scope"] != "rank_local_current_backend_state"
        or runtime["transport_timing_scope"]
        != "rank_local_current_transport_epoch_host_elapsed"
        or runtime["allocation_counter_scope"] != "rank_local_process_lifetime"
    ):
        raise RunnerError("runtime counter or memory scope is invalid")
    if (
        runtime["requested_backend"] != requested["backend"]
        or runtime["resolved_backend"] != requested["backend"]
        or runtime["requested_precision"] != requested["precision"]
        or runtime["resolved_precision"] != requested["precision"]
        or runtime["requested_overlap"] != requested["overlap"]
        or runtime["resolved_overlap"] != "off"
        or runtime["requested_graph"] != requested["graph"]
    ):
        raise RunnerError("runtime backend, precision, or graph request was downgraded")
    if requested["graph"] == "eager" and runtime["resolved_graph"] != "eager":
        raise RunnerError("eager execution unexpectedly resolved to a graph")
    if requested["graph"] == "required" and runtime["resolved_graph"] != "graph":
        raise RunnerError("required graph execution was not enabled")
    if runtime["resolved_graph"] == "eager" and runtime["graph_enabled"]:
        raise RunnerError("eager execution reports an enabled graph")
    if runtime["resolved_graph"] == "graph" and not (
        runtime["graph_enabled"] and runtime["graph_valid"]
    ):
        raise RunnerError("graph execution is not enabled and valid")
    expected_requested_transport = "none" if requested["backend"] == "cpu" else "staged"
    if (
        runtime["requested_transport"] != expected_requested_transport
        or runtime["resolved_transport"] != "none"
        or runtime["captured_transport_epoch_active"]
        or runtime["captured_transport_epoch_fresh"]
        or runtime["captured_requested_transport"] != "none"
        or runtime["captured_overlap_policy"] != "off"
    ):
        raise RunnerError("single-rank transport state is not inactive")
    if requested["backend"] == "cpu":
        if accelerator is not None or runtime["device_owner"] or runtime["device_uuid"]:
            raise RunnerError("CPU execution unexpectedly owns an accelerator device")
    else:
        if accelerator not in {"cuda", "hip"}:
            raise RunnerError("accelerator runtime provenance is invalid")
        if (
            not runtime["device_owner"]
            or runtime["device_id"] != device_id
            or not runtime["device_uuid"]
            or runtime["executable_build_count"] <= 0
            or runtime["process_device_bytes_peak"] <= 0
        ):
            raise RunnerError(
                "accelerator ownership or memory provenance is incomplete"
            )
        _normalize_gpu_uuid(runtime["device_uuid"])
        if any(runtime[name] for name in FORBIDDEN_MEASURED_COUNTERS[-5:]):
            raise RunnerError("accelerator execution used host/material fallback")


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
    runtime_report: Mapping[str, Any],
    memory: Mapping[str, int],
    accelerator: str,
    selectors: Sequence[str],
    toolkit_compiler: Optional[pathlib.Path],
    rocm_smi: Optional[pathlib.Path],
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
    meep_source_state = _source_tree_state(source_tree)
    runner_source_state = _source_tree_state(worktree)
    meep_module = importlib.import_module("meep")
    try:
        meep_extension = importlib.import_module("meep._meep")
    except ImportError:
        meep_extension = importlib.import_module("_meep")
    timings = [float(run["advance_seconds"]) for run in runs]
    if requested["backend"] == "nvidia":
        device_records, inventory_snapshot = _accelerator_device_provenance(
            accelerator,
            device_id,
            str(runtime_report.get("device_uuid", "")),
            selectors,
            rocm_smi,
        )
        toolchain = _toolchain_provenance(accelerator, toolkit_compiler)
        inventory_tool = (
            {
                "path": str(rocm_smi.resolve()),
                "sha256": bm.sha256_file(rocm_smi.resolve()),
            }
            if accelerator == "hip" and rocm_smi is not None
            else None
        )
    else:
        device_records = []
        inventory_snapshot = None
        toolchain = None
        inventory_tool = None
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
            "meep_build_commit": meep_source_state["commit"],
            "meep_build_dirty": meep_source_state["dirty"],
            "meep_build_diff_sha256": meep_source_state["diff_sha256"],
            "runner_commit": runner_source_state["commit"],
            "runner_dirty": runner_source_state["dirty"],
            "runner_diff_sha256": runner_source_state["diff_sha256"],
            "runner_source_sha256": bm.sha256_file(pathlib.Path(__file__).resolve()),
            "build_directory": build_directory,
            "configure_flags": configure_flags,
            "python": sys.version,
            "meep_module": _file_provenance(pathlib.Path(meep_module.__file__)),
            "meep_extension": _file_provenance(pathlib.Path(meep_extension.__file__)),
            "gdstk_version": importlib.import_module("gdstk").__version__,
            "accelerator": {
                "runtime": accelerator if requested["backend"] == "nvidia" else None,
                "toolchain": toolchain,
                "inventory_tool": inventory_tool,
                "inventory_snapshot": inventory_snapshot,
            },
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
            "accelerator": accelerator if requested["backend"] == "nvidia" else None,
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
        "runtime": dict(runtime_report),
        "memory": dict(memory),
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
            "runtime",
            "memory",
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
            "meep_build_diff_sha256",
            "runner_commit",
            "runner_dirty",
            "runner_diff_sha256",
            "runner_source_sha256",
            "build_directory",
            "configure_flags",
            "python",
            "meep_module",
            "meep_extension",
            "gdstk_version",
            "accelerator",
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
        if (
            not isinstance(value, str)
            or len(value) != 40
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise RunnerError(f"provenance {label} is invalid")
    for label in ("meep_build_dirty", "runner_dirty"):
        if type(provenance[label]) is not bool:
            raise RunnerError(f"provenance {label} is invalid")
    runner_source = pathlib.Path(__file__).resolve()
    if provenance["runner_source_sha256"] != bm.sha256_file(runner_source):
        raise RunnerError("runner source hash is stale")
    for label, root, prefix in (
        ("Meep", pathlib.Path(provenance["meep_build_source"]), "meep_build"),
        ("runner", runner_source.parents[3], "runner"),
    ):
        state = _source_tree_state(root)
        if (
            provenance[f"{prefix}_commit"] != state["commit"]
            or provenance[f"{prefix}_dirty"] is not state["dirty"]
            or provenance[f"{prefix}_diff_sha256"] != state["diff_sha256"]
            or (state["dirty"] and state["diff_sha256"] in (None, "", "0" * 64))
        ):
            raise RunnerError(f"{label} source-tree provenance is stale")
    for label in ("meep_build_source", "python", "gdstk_version", "cwd"):
        if not isinstance(provenance[label], str) or not provenance[label]:
            raise RunnerError(f"provenance {label} is invalid")
    for label in ("meep_module", "meep_extension"):
        record = _exact_keys(provenance[label], ("path", "sha256"), label)
        try:
            if (
                not pathlib.Path(record["path"]).is_absolute()
                or bm.sha256_file(pathlib.Path(record["path"])) != record["sha256"]
            ):
                raise RunnerError(f"provenance {label} hash is stale")
        except OSError as error:
            raise RunnerError(f"provenance {label} is unavailable") from error
    for label in ("build_directory", "configure_flags"):
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
    expected_finite_check = (
        "off" if status["profiled"] or requested["mode"] == "fixed-step" else "step"
    )
    if environment.get("MEEP_FINITE_CHECK") != expected_finite_check:
        raise RunnerError(
            f"MEEP_FINITE_CHECK must be {expected_finite_check} for this result"
        )

    accelerator_provenance = _exact_keys(
        provenance["accelerator"],
        ("runtime", "toolchain", "inventory_tool", "inventory_snapshot"),
        "accelerator provenance",
    )

    execution = _exact_keys(
        result["execution"],
        (
            "device_id",
            "accelerator",
            "profile_steps",
            "steps",
            "warmup_steps",
            "measured_repetitions",
        ),
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
            raise RunnerError(
                "ordinary execution records an unauthorized profile override"
            )
        expected_steps = int(manifest["stopping"]["steps"])
        expected_warmup = int(manifest["execution"]["warmup_steps"])
        expected_repetitions = int(manifest["execution"]["measured_repetitions"])
    if (
        _integer(execution["steps"], "execution steps") != expected_steps
        or _integer(execution["warmup_steps"], "execution warmup_steps")
        != expected_warmup
        or _integer(execution["measured_repetitions"], "execution measured_repetitions")
        != expected_repetitions
    ):
        raise RunnerError("execution semantics do not match the manifest/profile mode")

    device_records = provenance["device_records"]
    if not isinstance(device_records, list):
        raise RunnerError("provenance device records are invalid")
    if requested["backend"] == "cpu":
        if device_records:
            raise RunnerError("CPU execution cannot record an accelerator device")
        if execution["accelerator"] is not None or dict(accelerator_provenance) != {
            "runtime": None,
            "toolchain": None,
            "inventory_tool": None,
            "inventory_snapshot": None,
        }:
            raise RunnerError("CPU execution has accelerator provenance")
    else:
        accelerator = execution["accelerator"]
        if (
            accelerator not in {"cuda", "hip"}
            or accelerator_provenance["runtime"] != accelerator
        ):
            raise RunnerError("accelerator runtime provenance is invalid")
        if (
            environment.get("MEEP_ACCELERATOR_RUNTIME") != accelerator
            or environment.get("MEEP_GPU_AWARE_MPI") != "no"
            or environment.get("MEEP_NVIDIA_MPI_OVERLAP") != requested["overlap"]
            or environment.get("MEEP_NVIDIA_GRAPH_MODE") != requested["graph"]
        ):
            raise RunnerError("accelerator environment does not match the manifest")
        toolchain = _exact_keys(
            accelerator_provenance["toolchain"],
            ("path", "sha256", "version"),
            "accelerator toolchain provenance",
        )
        if (
            not pathlib.Path(toolchain["path"]).is_absolute()
            or len(str(toolchain["sha256"])) != 64
            or str(toolchain["sha256"]) == "0" * 64
            or not toolchain["version"]
        ):
            raise RunnerError("accelerator toolchain provenance is incomplete")
        try:
            if bm.sha256_file(pathlib.Path(toolchain["path"])) != toolchain["sha256"]:
                raise RunnerError("accelerator toolchain hash is stale")
        except OSError as error:
            raise RunnerError("accelerator toolchain is unavailable") from error
        if len(device_records) != 1:
            raise RunnerError(
                "single-rank accelerator execution requires exactly one device record"
            )
        device = _exact_keys(
            device_records[0],
            (
                (
                    "accelerator",
                    "visible_device",
                    "visible_devices",
                    "process_device_id",
                    "physical_selector",
                    "inventory_index",
                    "uuid",
                    "name",
                    "pci_bus_id",
                    "memory_bytes",
                    "core_clock",
                    "memory_clock",
                    "driver_version",
                )
                if accelerator == "cuda"
                else (
                    "accelerator",
                    "visible_device",
                    "visible_devices",
                    "process_device_id",
                    "physical_selector",
                    "inventory_card",
                    "uuid",
                    "physical_unique_id",
                    "pci_bus_id",
                    "numa_node",
                    "name",
                    "architecture",
                    "memory_bytes",
                    "core_clock",
                    "memory_clock",
                    "driver_version",
                )
            ),
            "accelerator device record",
        )
        if (
            device["accelerator"] != accelerator
            or _integer(device["process_device_id"], "accelerator process_device_id")
            != device_id
            or _integer(device["visible_device"], "accelerator visible_device")
            != device_id
            or not isinstance(device["visible_devices"], list)
        ):
            raise RunnerError(
                "accelerator device record does not match execution ownership"
            )
        if accelerator == "hip" or device_id < len(device["visible_devices"]):
            expected_selector = device["visible_devices"][device_id]
        else:
            expected_selector = str(device_id)
        if str(device["physical_selector"]) != str(expected_selector):
            raise RunnerError(
                "accelerator physical selector does not match its visibility mask"
            )
        if (
            _integer(device["memory_bytes"], "accelerator memory_bytes") <= 0
            or not _normalize_pci_bus_id(device["pci_bus_id"])
            or _normalize_gpu_uuid(device["uuid"])
            != _normalize_gpu_uuid(result["runtime"]["device_uuid"])
        ):
            raise RunnerError("accelerator device identity or memory is invalid")
        for label in (
            "physical_selector",
            "uuid",
            "name",
            "driver_version",
            "core_clock",
            "memory_clock",
        ):
            if not isinstance(device[label], str) or not device[label]:
                raise RunnerError(f"accelerator {label} is invalid")
        if accelerator == "hip":
            inventory = _exact_keys(
                accelerator_provenance["inventory_tool"],
                ("path", "sha256"),
                "ROCm inventory provenance",
            )
            if (
                not pathlib.Path(inventory["path"]).is_absolute()
                or len(str(inventory["sha256"])) != 64
                or str(inventory["sha256"]) == "0" * 64
                or not device["physical_unique_id"]
                or not device["architecture"]
                or _integer(device["numa_node"], "HIP NUMA node") < 0
            ):
                raise RunnerError("HIP device provenance is incomplete")
            visible = [
                item.strip()
                for item in environment.get("ROCR_VISIBLE_DEVICES", "").split(",")
                if item.strip()
            ]
            if (
                visible != device["visible_devices"]
                or "HIP_VISIBLE_DEVICES" in environment
                or "CUDA_VISIBLE_DEVICES" in environment
            ):
                raise RunnerError(
                    "HIP execution did not use its recorded ROCr selector"
                )
            try:
                if (
                    bm.sha256_file(pathlib.Path(inventory["path"]))
                    != inventory["sha256"]
                ):
                    raise RunnerError("ROCm inventory tool hash is stale")
            except OSError as error:
                raise RunnerError("ROCm inventory tool is unavailable") from error
            snapshot = _exact_keys(
                accelerator_provenance["inventory_snapshot"],
                ("sha256", "output"),
                "ROCm inventory snapshot",
            )
            if (
                not isinstance(snapshot["output"], str)
                or hashlib.sha256(snapshot["output"].encode("utf-8")).hexdigest()
                != snapshot["sha256"]
            ):
                raise RunnerError("ROCm inventory snapshot hash is invalid")
            try:
                inventory_json = json.loads(snapshot["output"])
            except json.JSONDecodeError as error:
                raise RunnerError("ROCm inventory snapshot is invalid JSON") from error
            matches = [
                (name, entry)
                for name, entry in inventory_json.items()
                if isinstance(entry, Mapping)
                and entry.get("PCI Bus")
                and _normalize_pci_bus_id(str(entry["PCI Bus"]))
                == _normalize_pci_bus_id(device["pci_bus_id"])
            ]
            if len(matches) != 1:
                raise RunnerError(
                    "HIP BDF is not uniquely present in the ROCm inventory"
                )
            inventory_card, inventory_entry = matches[0]

            def inventory_value(*fragments: str) -> str:
                for key, value in inventory_entry.items():
                    lowered = str(key).lower()
                    if all(fragment in lowered for fragment in fragments):
                        return str(value)
                return ""

            memory_match = re.search(r"([0-9]+)", inventory_value("vram", "total"))
            expected_inventory = {
                "inventory_card": inventory_card,
                "physical_unique_id": str(inventory_entry.get("Unique ID", "")),
                "numa_node": int(inventory_entry.get("(Topology) Numa Node", -1)),
                "name": str(inventory_entry.get("Card Series", "")),
                "architecture": str(inventory_entry.get("GFX Version", "")),
                "memory_bytes": int(memory_match.group(1)) if memory_match else 0,
                "core_clock": inventory_value("sclk", "clock"),
                "memory_clock": inventory_value("mclk", "clock"),
                "driver_version": str(
                    inventory_json.get("system", {}).get(
                        "Driver version", inventory_value("driver")
                    )
                ),
            }
            if any(
                device[name] != expected
                for name, expected in expected_inventory.items()
            ):
                raise RunnerError(
                    "HIP device record disagrees with the ROCm inventory snapshot"
                )
        elif accelerator_provenance["inventory_tool"] is not None:
            raise RunnerError(
                "CUDA execution unexpectedly records a ROCm inventory tool"
            )
        elif accelerator_provenance["inventory_snapshot"] is not None:
            raise RunnerError(
                "CUDA execution unexpectedly records a ROCm inventory snapshot"
            )

    runtime_accelerator = execution["accelerator"]
    runtime = _exact_keys(
        result["runtime"],
        sorted(
            set(RUNTIME_STRING_NAMES)
            | set(RUNTIME_BOOL_NAMES)
            | set(RUNTIME_INTEGER_NAMES)
            | set(RUNTIME_COUNTER_NAMES)
            | set(RUNTIME_MEMORY_NAMES)
        ),
        "execution runtime report",
    )
    _validate_runtime_report(runtime, requested, runtime_accelerator, device_id)
    memory = _exact_keys(
        result["memory"],
        ("host_peak_bytes", *RUNTIME_MEMORY_NAMES),
        "memory",
    )
    if _integer(memory["host_peak_bytes"], "host peak bytes") <= 0:
        raise RunnerError("host peak memory is invalid")
    for name in RUNTIME_MEMORY_NAMES:
        if memory[name] != runtime[name]:
            raise RunnerError(f"memory gauge {name} disagrees with the runtime report")

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
    if (
        geometry["material_constants"]
        != manifest["materials"]["performance_adaptation"]
    ):
        raise RunnerError("recorded material constants do not match the manifest")
    if authenticated_translation is None:
        _, authenticated_translation = _geometry_records(manifest, _load_gdstk_module())
    for axis in ("x_um", "y_um"):
        expected = _finite_number(
            authenticated_translation[axis],
            f"authenticated geometry translation {axis}",
        )
        actual = _finite_number(translation[axis], f"geometry translation {axis}")
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise RunnerError(
                "recorded geometry translation does not match authenticated GDS bounds "
                "and the manifest centering rule"
            )
    expected_planes = validate_planes(
        manifest, transformed_ports(manifest, translation)
    )
    if geometry["planes"] != expected_planes:
        raise RunnerError(
            "recorded geometry planes do not match the manifest and transform"
        )
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
        monitor for monitor in manifest["case"]["monitors"] if monitor["kind"] == "mode"
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
                "warmup_seconds",
                "advance_seconds",
                "grid_shape",
                "grid_points_exact",
                "dt_meep",
                "steps",
                "warmup_steps",
                "start_step",
                "end_step",
                "physical_time_meep",
                "counter_start",
                "counter_end",
                "counter_deltas",
                "memory_start",
                "memory_end",
                "monitors",
            ),
            f"run {run_index}",
        )
        shape = run["grid_shape"]
        if shape != expected_shape:
            raise RunnerError(f"run {run_index} grid shape does not match the manifest")
        if _integer(
            run["grid_points_exact"], f"run {run_index} grid_points_exact"
        ) != math.prod(expected_shape):
            raise RunnerError(f"run {run_index} grid point count is invalid")
        for field in (
            "initialization_seconds",
            "warmup_seconds",
            "advance_seconds",
            "dt_meep",
            "physical_time_meep",
        ):
            if _finite_number(run[field], f"run {run_index} {field}") < 0:
                raise RunnerError(f"run {run_index} {field} must be non-negative")
        if (
            _finite_number(run["advance_seconds"], f"run {run_index} advance_seconds")
            <= 0
        ):
            raise RunnerError(f"run {run_index} advance_seconds must be positive")
        if not math.isclose(
            float(run["dt_meep"]), expected_dt, rel_tol=1e-12, abs_tol=1e-15
        ):
            raise RunnerError(f"run {run_index} timestep does not match the manifest")
        steps = _integer(run["steps"], f"run {run_index} steps")
        warmup_steps = _integer(run["warmup_steps"], f"run {run_index} warmup_steps")
        expected_window_warmup = expected_warmup if run_index == 0 else 0
        if steps != expected_steps or warmup_steps != expected_window_warmup:
            raise RunnerError(
                f"run {run_index} step counts do not match execution semantics"
            )
        initialization_seconds = _finite_number(
            run["initialization_seconds"], f"run {run_index} initialization_seconds"
        )
        warmup_seconds = _finite_number(
            run["warmup_seconds"], f"run {run_index} warmup_seconds"
        )
        if run_index == 0:
            if initialization_seconds <= 0:
                raise RunnerError("the first window does not record initialization")
            if (expected_warmup > 0) != (warmup_seconds > 0):
                raise RunnerError("the first window warmup timing is invalid")
        elif initialization_seconds != 0 or warmup_seconds != 0:
            raise RunnerError(
                "fixed-step repetitions rebuilt or rewarmed the simulation"
            )
        start_step = _integer(run["start_step"], f"run {run_index} start_step")
        end_step = _integer(run["end_step"], f"run {run_index} end_step")
        expected_start = (
            expected_warmup
            if run_index == 0
            else _integer(
                runs[run_index - 1]["end_step"], f"run {run_index - 1} end_step"
            )
        )
        if start_step != expected_start or end_step - start_step != steps:
            raise RunnerError("timed windows are not sequential fixed-step intervals")
        if not math.isclose(
            float(run["physical_time_meep"]),
            end_step * float(run["dt_meep"]),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise RunnerError(f"run {run_index} physical time is invalid")
        snapshots = {}
        for field in ("counter_start", "counter_end", "counter_deltas"):
            snapshot = _exact_keys(
                run[field], RUNTIME_COUNTER_NAMES, f"run {run_index} {field}"
            )
            snapshots[field] = {}
            for name in RUNTIME_COUNTER_NAMES:
                snapshots[field][name] = _integer(
                    snapshot[name], f"run {run_index} {field}.{name}"
                )
                if snapshots[field][name] < 0:
                    raise RunnerError(f"run {run_index} {field}.{name} is negative")
        for name in RUNTIME_COUNTER_NAMES:
            if (
                snapshots["counter_end"][name] - snapshots["counter_start"][name]
                != snapshots["counter_deltas"][name]
            ):
                raise RunnerError(f"run {run_index} counter delta {name} is invalid")
        if any(
            snapshots["counter_deltas"][name] for name in FORBIDDEN_MEASURED_COUNTERS
        ):
            raise RunnerError(
                f"run {run_index} allocated, recopied, recaptured, or fell back during measured work"
            )
        memory_snapshots = {}
        for field in ("memory_start", "memory_end"):
            snapshot = _exact_keys(
                run[field],
                ("host_peak_bytes", *RUNTIME_MEMORY_NAMES),
                f"run {run_index} {field}",
            )
            memory_snapshots[field] = {
                name: _integer(snapshot[name], f"run {run_index} {field}.{name}")
                for name in ("host_peak_bytes", *RUNTIME_MEMORY_NAMES)
            }
            if any(value < 0 for value in memory_snapshots[field].values()):
                raise RunnerError(f"run {run_index} {field} contains a negative gauge")
        for peak_name in (
            "host_peak_bytes",
            "process_device_bytes_peak",
            "process_pinned_bytes_peak",
        ):
            if (
                memory_snapshots["memory_end"][peak_name]
                < memory_snapshots["memory_start"][peak_name]
            ):
                raise RunnerError(f"run {run_index} memory peak {peak_name} decreased")
            if (
                run_index
                and memory_snapshots["memory_start"][peak_name]
                < runs[run_index - 1]["memory_end"][peak_name]
            ):
                raise RunnerError(
                    f"run {run_index} process-lifetime memory peak {peak_name} "
                    "precedes the previous window"
                )
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
                or _integer(monitor["mode_band"], f"run {run_index} monitor mode_band")
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
    if runs:
        final_counters = _counter_snapshot(runtime)
        if any(
            final_counters[name] < runs[-1]["counter_end"][name]
            for name in RUNTIME_COUNTER_NAMES
        ):
            raise RunnerError(
                "final runtime counters precede the final measured snapshot"
            )
        for peak_name in (
            "host_peak_bytes",
            "process_device_bytes_peak",
            "process_pinned_bytes_peak",
        ):
            if memory[peak_name] < max(run["memory_end"][peak_name] for run in runs):
                raise RunnerError(
                    f"published memory peak {peak_name} precedes a measured snapshot"
                )
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
        if not math.isclose(
            _finite_number(summary[key], f"timing summary {key}"),
            expected,
            rel_tol=1e-12,
        ):
            raise RunnerError(f"timing summary {key} is invalid")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--accelerator",
        choices=("cuda", "hip"),
        default="cuda",
        help="private runtime family for the public nvidia backend selector",
    )
    parser.add_argument(
        "--visible-device",
        help="physical CUDA or ROCr selector to expose as process-local device zero",
    )
    parser.add_argument("--toolkit-compiler", type=pathlib.Path)
    parser.add_argument("--rocm-smi", type=pathlib.Path)
    parser.add_argument("--profile-steps", type=int)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def _begin_singleton_accelerator_context(mp: Any, backend: str) -> bool:
    if backend != "nvidia":
        return False
    begun = False
    try:
        mp.begin_global_communications()
        begun = True
        mp.divide_parallel_processes(1)
    except Exception as error:
        if begun:
            try:
                mp.end_divide_parallel()
            except Exception:
                pass
        raise RunnerError(
            f"cannot initialize singleton accelerator communicator: {error}"
        ) from error
    return True


def _end_singleton_accelerator_context(mp: Any, active: bool) -> None:
    if not active:
        return
    try:
        mp.end_divide_parallel()
    except Exception as error:
        raise RunnerError(
            f"cannot finalize singleton accelerator communicator: {error}"
        ) from error


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = make_parser().parse_args(argv)
    mp = None
    accelerator_context_active = False
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
        requested = manifest["execution"]["requested"]
        expected_finite_check = (
            "off"
            if args.profile_steps is not None or requested["mode"] == "fixed-step"
            else "step"
        )
        configured_finite_check = os.environ.get("MEEP_FINITE_CHECK")
        if configured_finite_check not in (None, expected_finite_check):
            raise RunnerError(
                f"MEEP_FINITE_CHECK must be {expected_finite_check} for this run"
            )
        os.environ["MEEP_FINITE_CHECK"] = expected_finite_check
        logical_device_id = args.device_id
        selectors: list[str] = []
        if requested["backend"] == "nvidia":
            visibility_name = (
                "ROCR_VISIBLE_DEVICES"
                if args.accelerator == "hip"
                else "CUDA_VISIBLE_DEVICES"
            )
            if args.accelerator == "hip":
                os.environ.pop("HIP_VISIBLE_DEVICES", None)
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            if args.visible_device is not None:
                if not args.visible_device.strip() or "," in args.visible_device:
                    raise RunnerError(
                        "--visible-device must name exactly one physical selector"
                    )
                os.environ[visibility_name] = args.visible_device.strip()
                logical_device_id = 0
            selectors = _visible_selectors(args.accelerator, logical_device_id)
            if args.accelerator == "hip" and (
                args.rocm_smi is None or not args.rocm_smi.resolve().is_file()
            ):
                raise RunnerError("HIP execution requires an existing --rocm-smi tool")
            os.environ["MEEP_ACCELERATOR_RUNTIME"] = args.accelerator
            os.environ["MEEP_GPU_AWARE_MPI"] = "no"
            os.environ["MEEP_NVIDIA_MPI_OVERLAP"] = requested["overlap"]
            os.environ["MEEP_NVIDIA_GRAPH_MODE"] = requested["graph"]
        mp, gdstk = _load_runtime_modules()
        accelerator_context_active = _begin_singleton_accelerator_context(
            mp, requested["backend"]
        )
        records, translation = _geometry_records(manifest, gdstk)
        ports = transformed_ports(manifest, translation)
        planes = validate_planes(manifest, ports)
        steps = int(args.profile_steps or manifest["stopping"]["steps"])
        repetitions = (
            1
            if args.profile_steps
            else int(manifest["execution"]["measured_repetitions"])
        )
        warmup = 1 if args.profile_steps else int(manifest["execution"]["warmup_steps"])
        runs, runtime_report, memory = _run_session(
            mp,
            manifest,
            records,
            planes,
            device_id=logical_device_id,
            steps=steps,
            warmup_steps=warmup,
            repetitions=repetitions,
            profile=args.profile_steps is not None,
        )
        result = build_result(
            args.manifest,
            manifest_sha256,
            manifest,
            runs,
            translation,
            planes,
            device_id=logical_device_id,
            profile=args.profile_steps is not None,
            runtime_report=runtime_report,
            memory=memory,
            accelerator=args.accelerator,
            selectors=selectors,
            toolkit_compiler=args.toolkit_compiler,
            rocm_smi=args.rocm_smi,
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
    finally:
        if mp is not None:
            _end_singleton_accelerator_context(mp, accelerator_context_active)


if __name__ == "__main__":
    raise SystemExit(main())
