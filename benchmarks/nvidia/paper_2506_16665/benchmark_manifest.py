#!/usr/bin/env python3
"""Validate paper inputs and emit typed benchmark manifests/results."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import pathlib
import re
import subprocess
import sys
from typing import Any, Dict, Mapping, Optional, Sequence


HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_REFERENCE = HERE / "paper_reference.json"
DEFAULT_CASES = HERE / "runner_cases.json"
DEFAULT_MANIFEST_SCHEMA = HERE / "benchmark_manifest.schema.json"
DEFAULT_RESULT_SCHEMA = HERE / "benchmark_result.schema.json"
DEFAULT_MPI_RESULT_SCHEMA = HERE / "mpi_benchmark_result.schema.json"
GENERATOR_VERSION = 6
MANIFEST_SCHEMA_VERSION = 6
RESULT_SCHEMA_VERSION = 1
BACKENDS = {"auto", "cpu", "nvidia"}
PRECISIONS = {"native", "f32", "mixed"}
TRANSPORTS = {"none", "host", "staged", "direct", "auto"}
OVERLAPS = {"off", "auto", "required"}
GRAPHS = {"eager", "auto", "required"}
MODES = {"smoke", "fixed-step", "end-to-end"}
POWER_OBSERVABLES = {"transmission", "conversion_efficiency", "transmission_spectrum"}
SPECTRAL_ENVELOPE_POLICY = {
    "kind": "gaussian_frequency",
    "bandwidth_definition": "full_centered_wavelength_interval",
    "frequency_conversion": "reciprocal_endpoint_span",
}
PADDING_INTERPRETATION = "total_added_span_split_equally_between_both_sides"
PERFORMANCE_ADAPTATION_MATERIALS = {
    "Si": {"model": "nondispersive", "refractive_index": 3.48, "epsilon": 12.1104},
    "SiO2": {"model": "nondispersive", "epsilon": 2.09},
    "Si3N4": {"model": "nondispersive", "refractive_index": 2.0, "epsilon": 4.0},
}
MATERIAL_DISCRETIZATION_ADAPTATION = (
    "Meep uses per-component scalar-permittivity sampling at staggered Yee-grid "
    "locations (epsilon_averaging=false), producing a staircased material "
    "assignment that remains backend-independent and does not require host "
    "material fallback."
)
PRISM_VERTEX_CANONICALIZATION = "per_polygon_centroid_binary64_round_trip_once"
PRISM_VERTEX_CANONICALIZATION_ADAPTATION = (
    "After the authenticated case-wide translation, each imported GDS polygon "
    "undergoes one binary64 subtraction/addition round trip about its arithmetic "
    "vertex centroid before Prism construction. This selects libctl-consistent "
    "floating representatives without decimal snapping or topology changes."
)


class ValidationError(ValueError):
    """Raised when benchmark data is incomplete or inconsistent."""


def _reject_json_constant(value: str) -> None:
    raise ValidationError(f"JSON contains non-finite number {value}")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValidationError(f"{label} must be an array")
    return value


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(f"{label} must be finite")
    if positive and number <= 0:
        raise ValidationError(f"{label} must be positive")
    return number


def _positive_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{label} must be an integer")
    if value < 0 if allow_zero else value <= 0:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValidationError(f"{label} must be {qualifier}")
    return value


def load_json_object(path: pathlib.Path, label: str) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream, parse_constant=_reject_json_constant)
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot read {label} {path}: {error}") from error
    return dict(_mapping(value, label))


def _resolve_schema_reference(
    schema_root: Mapping[str, Any], reference: str
) -> Mapping[str, Any]:
    if not reference.startswith("#/"):
        raise ValidationError(f"unsupported JSON Schema reference {reference!r}")
    node: Any = schema_root
    for part in reference[2:].split("/"):
        key = part.replace("~1", "/").replace("~0", "~")
        node = _mapping(node, f"schema reference {reference}").get(key)
    return _mapping(node, f"schema reference {reference}")


def _json_equal(left: Any, right: Any) -> bool:
    """Compare JSON values without Python's bool/int equality aliasing."""
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if isinstance(left, str) or isinstance(right, str):
        return isinstance(left, str) and isinstance(right, str) and left == right
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        return set(left) == set(right) and all(
            _json_equal(left[key], right[key]) for key in left
        )
    left_is_sequence = isinstance(left, Sequence) and not isinstance(left, (str, bytes))
    right_is_sequence = isinstance(right, Sequence) and not isinstance(
        right, (str, bytes)
    )
    if left_is_sequence or right_is_sequence:
        if not left_is_sequence or not right_is_sequence or len(left) != len(right):
            return False
        return all(_json_equal(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def _require_schema_version(value: Any, expected: int, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value != expected:
        raise ValidationError(f"{label} must be {expected}")


def _validate_schema_structure(
    value: Any,
    schema: Mapping[str, Any],
    schema_root: Mapping[str, Any],
    label: str,
) -> None:
    if "$ref" in schema:
        _validate_schema_structure(
            value,
            _resolve_schema_reference(schema_root, schema["$ref"]),
            schema_root,
            label,
        )
        return
    if "const" in schema and not _json_equal(value, schema["const"]):
        raise ValidationError(f"{label} must equal {schema['const']!r}")
    if "enum" in schema:
        choices = _sequence(schema["enum"], f"{label} schema enum")
        if not any(_json_equal(value, choice) for choice in choices):
            raise ValidationError(f"{label} must be one of {schema['enum']!r}")
    schema_type = schema.get("type")
    if schema_type is not None:
        allowed_types = (
            list(schema_type) if isinstance(schema_type, list) else [schema_type]
        )

        def matches(candidate: str) -> bool:
            if candidate == "null":
                return value is None
            if candidate == "boolean":
                return isinstance(value, bool)
            if candidate == "integer":
                return isinstance(value, int) and not isinstance(value, bool)
            if candidate == "number":
                return isinstance(value, (int, float)) and not isinstance(value, bool)
            if candidate == "string":
                return isinstance(value, str)
            if candidate == "array":
                return isinstance(value, Sequence) and not isinstance(
                    value, (str, bytes)
                )
            if candidate == "object":
                return isinstance(value, Mapping)
            raise ValidationError(f"unsupported JSON Schema type {candidate!r}")

        if not any(matches(candidate) for candidate in allowed_types):
            raise ValidationError(f"{label} must have type {schema_type!r}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = _finite(value, label)
        if "minimum" in schema and number < float(schema["minimum"]):
            raise ValidationError(f"{label} is below its minimum")
        if "exclusiveMinimum" in schema and number <= float(schema["exclusiveMinimum"]):
            raise ValidationError(f"{label} is not above its exclusive minimum")
        if "maximum" in schema and number > float(schema["maximum"]):
            raise ValidationError(f"{label} exceeds its maximum")
        if "exclusiveMaximum" in schema and number >= float(schema["exclusiveMaximum"]):
            raise ValidationError(f"{label} is not below its exclusive maximum")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < int(schema["minLength"]):
            raise ValidationError(f"{label} is shorter than minLength")
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            raise ValidationError(f"{label} is longer than maxLength")
        if "pattern" in schema and re.search(str(schema["pattern"]), value) is None:
            raise ValidationError(f"{label} does not match its pattern")
    if schema_type == "object" or "properties" in schema:
        mapping = _mapping(value, label)
        properties = _mapping(
            schema.get("properties", {}), f"{label} schema properties"
        )
        required = _sequence(schema.get("required", []), f"{label} schema required")
        missing = set(required).difference(mapping)
        if missing:
            raise ValidationError(f"{label} is missing {sorted(missing)}")
        additional = schema.get("additionalProperties", True)
        for key, child in mapping.items():
            if key in properties:
                _validate_schema_structure(
                    child,
                    _mapping(properties[key], f"{label}.{key} schema"),
                    schema_root,
                    f"{label}.{key}",
                )
            elif additional is False:
                raise ValidationError(f"{label} contains unexpected property {key!r}")
            elif isinstance(additional, Mapping):
                _validate_schema_structure(
                    child, additional, schema_root, f"{label}.{key}"
                )
    elif schema_type == "array":
        values = _sequence(value, label)
        if "minItems" in schema and len(values) < int(schema["minItems"]):
            raise ValidationError(f"{label} has fewer than minItems")
        if "maxItems" in schema and len(values) > int(schema["maxItems"]):
            raise ValidationError(f"{label} has more than maxItems")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, child in enumerate(values):
                _validate_schema_structure(
                    child, item_schema, schema_root, f"{label}[{index}]"
                )


def load_reference(path: pathlib.Path = DEFAULT_REFERENCE) -> Dict[str, Any]:
    reference = load_json_object(path, "paper reference")
    validate_reference(reference)
    return reference


def validate_reference(reference: Mapping[str, Any]) -> None:
    _require_schema_version(
        reference.get("schema_version"), 1, "paper reference schema_version"
    )
    paper = _mapping(reference.get("paper"), "paper")
    common = _mapping(reference.get("common"), "common")
    devices = _mapping(reference.get("devices"), "devices")
    if not paper.get("code_commit"):
        raise ValidationError("paper.code_commit is required")
    if len(devices) != 6:
        raise ValidationError("exactly six normative devices are required")
    required_device_keys = {
        "display_name",
        "gds_relative_path",
        "gds_sha256",
        "domain_um",
        "geometry",
        "input",
        "target",
        "cladding",
        "boundary_exception",
        "timings",
        "workload_25",
        "paper_results",
    }
    if common.get("cells_per_material_wavelength") != [6, 10, 15, 20, 25]:
        raise ValidationError("paper resolution rows must be 6, 10, 15, 20, and 25")
    convention = _mapping(
        reference.get("material_pole_convention"), "material_pole_convention"
    )
    for key in (
        "equation",
        "fourier_time_convention",
        "frequency_unit",
        "residue_unit",
    ):
        if not isinstance(convention.get(key), str) or not convention[key].strip():
            raise ValidationError(f"material_pole_convention.{key} is required")
    for name, raw_device in devices.items():
        device = _mapping(raw_device, f"device {name}")
        missing = required_device_keys.difference(device)
        if missing:
            raise ValidationError(f"device {name!r} is missing {sorted(missing)}")
        if set(_mapping(device["timings"], f"{name}.timings")) != {
            "6",
            "10",
            "15",
            "20",
            "25",
        }:
            raise ValidationError(f"device {name!r} must contain all five timing rows")
        _validate_digest(device["gds_sha256"], f"device {name!r} GDS")


def load_case_definitions(path: pathlib.Path = DEFAULT_CASES) -> Dict[str, Any]:
    definitions = load_json_object(path, "runner case definitions")
    validate_case_definitions(definitions)
    return definitions


def _validate_digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValidationError(f"{label} has an invalid SHA-256")
    return value


def validate_case_definitions(definitions: Mapping[str, Any]) -> None:
    _require_schema_version(
        definitions.get("schema_version"), 1, "runner case schema_version"
    )
    assets = _mapping(definitions.get("assets"), "runner assets")
    layer_stack = _mapping(assets.get("layer_stack"), "layer stack asset")
    if not layer_stack.get("relative_path"):
        raise ValidationError("layer stack relative_path is required")
    _validate_digest(layer_stack.get("sha256"), "layer stack")
    geometry_import = _mapping(definitions.get("geometry_import"), "geometry_import")
    for key in ("gds_scale", "rotation_deg", "port_extension_um"):
        _finite(geometry_import.get(key), f"geometry_import.{key}")
    if (
        geometry_import.get("prism_vertex_canonicalization")
        != PRISM_VERTEX_CANONICALIZATION
    ):
        raise ValidationError(
            "geometry_import.prism_vertex_canonicalization is invalid"
        )
    layers = _sequence(definitions.get("layers"), "layers")
    layer_names = set()
    for index, raw_layer in enumerate(layers):
        layer = _mapping(raw_layer, f"layers[{index}]")
        for key in (
            "name",
            "gds_layer",
            "gds_datatype",
            "z_min_um",
            "z_max_um",
            "material",
        ):
            if key not in layer:
                raise ValidationError(f"layers[{index}].{key} is required")
        if layer["name"] in layer_names:
            raise ValidationError(f"duplicate layer name {layer['name']!r}")
        layer_names.add(layer["name"])
        if _finite(layer["z_max_um"], f"layers[{index}].z_max_um") <= _finite(
            layer["z_min_um"], f"layers[{index}].z_min_um"
        ):
            raise ValidationError(f"layers[{index}] must have positive thickness")
    cell = _mapping(definitions.get("cell"), "cell")
    if not isinstance(cell.get("x_y_extent_rule"), str):
        raise ValidationError("cell.x_y_extent_rule is required")
    if _finite(cell.get("z_max_um"), "cell.z_max_um") <= _finite(
        cell.get("z_min_um"), "cell.z_min_um"
    ):
        raise ValidationError("cell z extent must be positive")
    _finite(cell.get("in_plane_padding_um"), "cell.in_plane_padding_um", positive=True)
    if cell.get("epsilon_averaging") is not False:
        raise ValidationError("cell.epsilon_averaging must be false")
    if cell.get("material_discretization") != "yee_grid_point_staircased":
        raise ValidationError(
            "cell.material_discretization must be yee_grid_point_staircased"
        )
    adaptation_materials = _mapping(
        definitions.get("performance_adaptation_materials"),
        "performance_adaptation_materials",
    )
    if not _json_equal(adaptation_materials, PERFORMANCE_ADAPTATION_MATERIALS):
        raise ValidationError("performance adaptation material constants are invalid")
    time_stepping = _mapping(definitions.get("time_stepping"), "time_stepping")
    courant = _finite(
        time_stepping.get("courant_factor"),
        "time_stepping.courant_factor",
        positive=True,
    )
    if courant > 1:
        raise ValidationError("time_stepping.courant_factor must not exceed one")
    boundaries = _mapping(definitions.get("boundaries"), "boundaries")
    _finite(boundaries.get("thickness_um"), "boundaries.thickness_um", positive=True)
    if set(_sequence(boundaries.get("sides"), "boundaries.sides")) != {
        "x_low",
        "x_high",
        "y_low",
        "y_high",
        "z_low",
        "z_high",
    }:
        raise ValidationError("PML must enumerate all six boundary sides")
    source_defaults = _mapping(definitions.get("source_defaults"), "source_defaults")
    _positive_int(source_defaults.get("mode_count"), "source_defaults.mode_count")
    spectral_envelope = _mapping(
        source_defaults.get("spectral_envelope"),
        "source_defaults.spectral_envelope",
    )
    if dict(spectral_envelope) != SPECTRAL_ENVELOPE_POLICY:
        raise ValidationError("source_defaults.spectral_envelope policy is invalid")
    if (
        source_defaults.get("transverse_padding_interpretation")
        != PADDING_INTERPRETATION
    ):
        raise ValidationError("source transverse padding interpretation is invalid")
    monitor_defaults = _mapping(definitions.get("monitor_defaults"), "monitor_defaults")
    _positive_int(monitor_defaults.get("mode_count"), "monitor_defaults.mode_count")
    if (
        monitor_defaults.get("transverse_padding_interpretation")
        != PADDING_INTERPRETATION
    ):
        raise ValidationError("monitor transverse padding interpretation is invalid")
    _positive_int(
        monitor_defaults.get("dft_decimation_factor"),
        "monitor_defaults.dft_decimation_factor",
    )
    decay = _mapping(definitions.get("decay_stop"), "decay_stop")
    _finite(
        decay.get("relative_threshold"), "decay_stop.relative_threshold", positive=True
    )
    _finite(
        decay.get("sampling_interval_meep_time"),
        "decay_stop.sampling_interval_meep_time",
        positive=True,
    )
    observable_policy = _mapping(
        definitions.get("observable_policy"), "observable_policy"
    )
    _require_schema_version(
        observable_policy.get("schema_version"), 1, "observable_policy.schema_version"
    )
    if observable_policy.get("reference_kind") != "cpu_native_baseline":
        raise ValidationError("observable_policy.reference_kind is invalid")
    _require_schema_version(
        observable_policy.get("reference_artifact_schema_version"),
        1,
        "observable policy reference schema_version",
    )
    tolerances = _mapping(
        observable_policy.get("tolerances_by_precision"),
        "observable_policy.tolerances_by_precision",
    )
    if set(tolerances) != PRECISIONS:
        raise ValidationError("observable policy must define every precision")
    for precision, raw_tolerance in tolerances.items():
        tolerance = _mapping(raw_tolerance, f"observable tolerance {precision}")
        for key in ("absolute", "relative"):
            if _finite(tolerance.get(key), f"{precision} {key} tolerance") < 0:
                raise ValidationError("observable tolerances must be non-negative")
    cases = _mapping(definitions.get("cases"), "cases")
    if len(cases) != 6:
        raise ValidationError("runner cases must contain exactly six cases")
    for name, raw_case in cases.items():
        case = _mapping(raw_case, f"case {name}")
        for key in (
            "gds_cell_name",
            "yaml_relative_path",
            "yaml_sha256",
            "active_layers",
            "ports",
            "source",
            "monitors",
            "required_observables",
        ):
            if key not in case:
                raise ValidationError(f"case {name}.{key} is required")
        _validate_digest(case["yaml_sha256"], f"case {name} YAML")
        active_layers = set(
            _sequence(case["active_layers"], f"case {name}.active_layers")
        )
        if not active_layers or not active_layers.issubset(layer_names):
            raise ValidationError(f"case {name} has invalid active_layers")
        ports = _mapping(case["ports"], f"case {name}.ports")
        if not ports:
            raise ValidationError(f"case {name} must define ports")
        for port_name, raw_port in ports.items():
            port = _mapping(raw_port, f"case {name}.ports.{port_name}")
            center = _sequence(port.get("center_um"), f"port {port_name}.center_um")
            if len(center) != 2:
                raise ValidationError(
                    f"port {port_name}.center_um must have two values"
                )
            for coordinate in center:
                _finite(coordinate, f"port {port_name} center coordinate")
            orientation = _finite(
                port.get("orientation_deg"), f"port {port_name}.orientation_deg"
            )
            if orientation < 0 or orientation >= 360:
                raise ValidationError(
                    f"port {port_name} orientation must be in [0, 360)"
                )
            _finite(port.get("width_um"), f"port {port_name}.width_um", positive=True)
            gds_layer = _sequence(port.get("gds_layer"), f"port {port_name}.gds_layer")
            if len(gds_layer) != 2 or not all(
                isinstance(item, int) for item in gds_layer
            ):
                raise ValidationError(
                    f"port {port_name}.gds_layer must be two integers"
                )
        source = _mapping(case["source"], f"case {name}.source")
        if source.get("port") not in ports:
            raise ValidationError(f"case {name} source references an unknown port")
        _positive_int(
            source.get("mode_order"), f"case {name}.source.mode_order", allow_zero=True
        )
        monitors = _sequence(case["monitors"], f"case {name}.monitors")
        if not monitors:
            raise ValidationError(f"case {name} must define at least one monitor")
        monitor_names = [
            _mapping(monitor, f"case {name} monitor").get("name")
            for monitor in monitors
        ]
        if any(not monitor_name for monitor_name in monitor_names):
            raise ValidationError(f"case {name} monitor names are required")
        if len(set(monitor_names)) != len(monitor_names):
            raise ValidationError(f"case {name} monitor names must be unique")
        monitor_by_name = {}
        for index, raw_monitor in enumerate(monitors):
            monitor = _mapping(raw_monitor, f"case {name}.monitors[{index}]")
            if not monitor.get("name") or not monitor.get("kind"):
                raise ValidationError(f"case {name} monitor name and kind are required")
            monitor_by_name[monitor["name"]] = monitor
            if "port" in monitor and monitor["port"] not in ports:
                raise ValidationError(f"case {name} monitor references an unknown port")
            if (
                "input_monitor" in monitor
                and monitor["input_monitor"] not in monitor_by_name
                and monitor["input_monitor"] not in monitor_names
            ):
                raise ValidationError(
                    f"case {name} derived monitor references an unknown monitor"
                )
            if monitor.get("observable") in POWER_OBSERVABLES:
                normalization = _mapping(
                    monitor.get("normalization"),
                    f"case {name} monitor {monitor['name']} normalization",
                )
                if normalization.get("kind") != "mode_power_ratio":
                    raise ValidationError(
                        f"case {name} monitor {monitor['name']} normalization kind is invalid"
                    )
                for key in ("numerator_quantity", "denominator_quantity"):
                    if normalization.get(key) != "forward_mode_power":
                        raise ValidationError(
                            f"case {name} monitor {monitor['name']} {key} is invalid"
                        )
                denominator = normalization.get("denominator_monitor")
                if denominator not in monitor_names:
                    raise ValidationError(
                        f"case {name} monitor {monitor['name']} normalization references an unknown monitor"
                    )
        incident_monitors = [
            monitor
            for monitor in monitor_by_name.values()
            if monitor.get("observable") == "incident_power"
        ]
        if len(incident_monitors) != 1:
            raise ValidationError(
                f"case {name} must define exactly one incident-power monitor"
            )
        incident = incident_monitors[0]
        for key in ("port", "polarization", "mode_order"):
            if incident.get(key) != source.get(key):
                raise ValidationError(
                    f"case {name} incident-power monitor must match source {key}"
                )
        for monitor in monitor_by_name.values():
            if monitor.get("observable") in POWER_OBSERVABLES:
                denominator = monitor["normalization"]["denominator_monitor"]
                if denominator != incident["name"]:
                    raise ValidationError(
                        f"case {name} power monitor must normalize by incident power"
                    )
        required_observables = _sequence(
            case["required_observables"], f"case {name}.required_observables"
        )
        observable_names = set()
        for index, raw_observable in enumerate(required_observables):
            observable = _mapping(
                raw_observable, f"case {name}.required_observables[{index}]"
            )
            for key in ("name", "monitor", "unit", "evaluation"):
                if not isinstance(observable.get(key), str) or not observable[key]:
                    raise ValidationError(
                        f"case {name}.required_observables[{index}].{key} is required"
                    )
            if observable["name"] in observable_names:
                raise ValidationError(
                    f"case {name} required observable names must be unique"
                )
            if observable["monitor"] not in monitor_names:
                raise ValidationError(
                    f"case {name} required observable references an unknown monitor"
                )
            observable_names.add(observable["name"])
        if name == "ring" and observable_names != {
            "resonance_wavelength_um",
            "fwhm_um",
            "quality_factor",
        }:
            raise ValidationError("ring must require resonance wavelength, FWHM, and Q")
        support = case.get("runner_support", {"supported": True, "reason": None})
        support = _mapping(support, f"case {name}.runner_support")
        if not isinstance(support.get("supported"), bool):
            raise ValidationError(
                f"case {name}.runner_support.supported must be boolean"
            )
        reason = support.get("reason")
        if support["supported"] and reason is not None:
            raise ValidationError(
                f"case {name} supported runner must not have a reason"
            )
        if not support["supported"] and (not isinstance(reason, str) or not reason):
            raise ValidationError(f"case {name} unsupported runner requires a reason")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def physics_configuration_sha256(manifest: Mapping[str, Any]) -> str:
    """Identify all backend-independent inputs that can change physics output."""
    return sha256_json(
        {
            "case": manifest.get("case"),
            "excitation": manifest.get("excitation"),
            "discretization": manifest.get("discretization"),
            "materials": manifest.get("materials"),
            "stopping": manifest.get("stopping"),
        }
    )


def git_command(checkout: pathlib.Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(checkout), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValidationError(
            f"cannot inspect git checkout {checkout}: {error}"
        ) from error
    return result.stdout.rstrip("\n")


def _validate_asset(
    checkout: pathlib.Path,
    relative_path: str,
    expected_hash: str,
    label: str,
) -> Dict[str, Any]:
    path = checkout / relative_path
    if not path.is_file():
        raise ValidationError(f"missing {label}: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ValidationError(
            f"{label} hash mismatch: got {actual_hash}, expected {expected_hash}"
        )
    return {
        "relative_path": relative_path,
        "path": str(path),
        "sha256": actual_hash,
        "bytes": path.stat().st_size,
    }


def validate_checkout(
    checkout: pathlib.Path,
    reference: Mapping[str, Any],
    case_definitions: Mapping[str, Any],
    selected_cases: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    checkout = checkout.resolve()
    if not checkout.is_dir():
        raise ValidationError(f"fdtd-pipeline checkout is not a directory: {checkout}")
    expected_commit = str(reference["paper"]["code_commit"])
    actual_commit = git_command(checkout, "rev-parse", "HEAD").strip()
    if actual_commit != expected_commit:
        raise ValidationError(
            f"checkout HEAD {actual_commit!r} does not match pinned commit {expected_commit!r}"
        )
    status_entries = [
        line
        for line in git_command(checkout, "status", "--porcelain=v1").splitlines()
        if line
    ]
    names = (
        list(reference["devices"]) if selected_cases is None else list(selected_cases)
    )
    if not names:
        raise ValidationError("at least one case must be selected")
    unknown = set(names).difference(reference["devices"])
    if unknown:
        raise ValidationError(f"unknown selected cases: {sorted(unknown)}")
    if set(reference["devices"]) != set(case_definitions["cases"]):
        raise ValidationError("paper reference and runner case names differ")

    assets: Dict[str, Any] = {"gds": {}, "yaml": {}}
    errors = []
    for name in names:
        device = reference["devices"][name]
        case = case_definitions["cases"][name]
        for kind, relative_path, expected_hash in (
            ("gds", device["gds_relative_path"], device["gds_sha256"]),
            ("yaml", case["yaml_relative_path"], case["yaml_sha256"]),
        ):
            try:
                assets[kind][name] = _validate_asset(
                    checkout, relative_path, expected_hash, f"{kind.upper()} for {name}"
                )
            except ValidationError as error:
                errors.append(str(error))
    layer_asset = case_definitions["assets"]["layer_stack"]
    try:
        assets["layer_stack"] = _validate_asset(
            checkout,
            layer_asset["relative_path"],
            layer_asset["sha256"],
            "layer stack",
        )
    except ValidationError as error:
        errors.append(str(error))
    if errors:
        raise ValidationError("\n".join(errors))
    return {
        "checkout": str(checkout),
        "expected_commit": expected_commit,
        "actual_commit": actual_commit,
        "dirty": bool(status_entries),
        "status_porcelain": status_entries,
        "validated_cases": names,
        "assets": assets,
    }


def convert_resolution(
    cells_per_material_wavelength: float,
    n_max: float,
    lambda_min_um: float,
    rounding: str = "ceil",
) -> Dict[str, Any]:
    cells = _finite(
        cells_per_material_wavelength,
        "cells_per_material_wavelength",
        positive=True,
    )
    index = _finite(n_max, "n_max", positive=True)
    wavelength = _finite(lambda_min_um, "lambda_min_um", positive=True)
    exact = cells * index / wavelength
    if rounding == "ceil":
        selected = float(math.ceil(exact))
    elif rounding == "nearest":
        selected = float(math.floor(exact + 0.5))
    elif rounding == "floor":
        selected = float(math.floor(exact))
    elif rounding == "none":
        selected = exact
    else:
        raise ValidationError(f"unknown resolution rounding policy: {rounding}")
    return {
        "cells_per_material_wavelength": cells,
        "n_max": index,
        "lambda_min_um": wavelength,
        "exact_resolution_px_per_um": exact,
        "rounding": rounding,
        "resolution_px_per_um": selected,
        "dx_um": 1.0 / selected,
    }


def validate_execution(
    backend: str, precision: str, ranks: int, transport: str
) -> None:
    if backend not in BACKENDS:
        raise ValidationError(f"backend must be one of {sorted(BACKENDS)}")
    if precision not in PRECISIONS:
        raise ValidationError(f"precision must be one of {sorted(PRECISIONS)}")
    _positive_int(ranks, "ranks")
    if transport not in TRANSPORTS:
        raise ValidationError(f"MPI transport must be one of {sorted(TRANSPORTS)}")
    if backend == "cpu" and precision != "native":
        raise ValidationError("CPU backend supports only native precision")
    if ranks == 1 and transport != "none":
        raise ValidationError("single-rank execution requires MPI transport none")
    if ranks > 1 and backend == "cpu" and transport != "host":
        raise ValidationError("multi-rank CPU execution requires host MPI transport")
    if (
        ranks > 1
        and backend == "nvidia"
        and transport
        not in {
            "staged",
            "direct",
            "auto",
        }
    ):
        raise ValidationError(
            "multi-rank NVIDIA execution requires staged, direct, or auto MPI transport"
        )
    if ranks > 1 and backend == "auto" and transport != "auto":
        raise ValidationError("multi-rank auto backend requires auto MPI transport")


def build_stopping_spec(
    mode: str,
    steps: Optional[int],
    max_steps: Optional[int],
    auto_shutoff_threshold: float,
    decay_check_interval_meep: float,
) -> Dict[str, Any]:
    if mode not in MODES:
        raise ValidationError(f"mode must be one of {sorted(MODES)}")
    threshold = _finite(auto_shutoff_threshold, "auto-shutoff threshold", positive=True)
    interval = _finite(decay_check_interval_meep, "decay check interval", positive=True)
    if threshold >= 1:
        raise ValidationError("auto-shutoff threshold must be less than one")
    if mode != "end-to-end" and (threshold != 1e-5 or interval != 50.0):
        raise ValidationError(
            "decay threshold and interval are only configurable in end-to-end mode"
        )
    if mode == "smoke":
        if max_steps is not None:
            raise ValidationError("--max-steps is only valid in end-to-end mode")
        count = 10 if steps is None else _positive_int(steps, "smoke steps")
        return {"kind": "fixed_steps", "steps": count, "purpose": "smoke"}
    if mode == "fixed-step":
        if max_steps is not None:
            raise ValidationError("--max-steps is only valid in end-to-end mode")
        if steps is None:
            raise ValidationError("--steps is required in fixed-step mode")
        return {
            "kind": "fixed_steps",
            "steps": _positive_int(steps, "fixed-step steps"),
            "purpose": "steady_state_throughput",
        }
    if steps is not None:
        raise ValidationError("--steps is not valid in end-to-end mode")
    if max_steps is None:
        raise ValidationError("--max-steps is required in end-to-end mode")
    return {
        "kind": "field_energy_decay",
        "relative_threshold": threshold,
        "check_interval_meep_time": interval,
        "max_steps": _positive_int(max_steps, "end-to-end max_steps"),
        "paper_equivalence": "Meep adaptation; commercial auto-shutoff is unspecified",
    }


def load_material_validation(
    path: pathlib.Path,
    reference_path: pathlib.Path,
    reference: Mapping[str, Any],
    required_wavelength_range_um: Sequence[float],
) -> Dict[str, Any]:
    proof = load_json_object(path, "material validation artifact")
    _require_schema_version(
        proof.get("schema_version"), 1, "material validation schema_version"
    )
    if proof.get("paper_reference_sha256") != sha256_file(reference_path.resolve()):
        raise ValidationError("material validation paper reference hash does not match")
    convention = _mapping(proof.get("pole_convention"), "pole_convention")
    for key in (
        "equation",
        "fourier_time_convention",
        "frequency_unit",
        "residue_unit",
    ):
        if not isinstance(convention.get(key), str) or not convention[key].strip():
            raise ValidationError(f"pole_convention.{key} is required")
    if dict(convention) != dict(reference["material_pole_convention"]):
        raise ValidationError(
            "material validation pole convention does not match the paper reference"
        )
    wavelength_range = _sequence(
        proof.get("wavelength_range_um"), "wavelength_range_um"
    )
    if len(wavelength_range) != 2:
        raise ValidationError("wavelength_range_um must have two values")
    low = _finite(wavelength_range[0], "wavelength_range_um[0]", positive=True)
    high = _finite(wavelength_range[1], "wavelength_range_um[1]", positive=True)
    if high <= low:
        raise ValidationError("material validation wavelength range is empty")
    required_low = _finite(
        required_wavelength_range_um[0], "required wavelength minimum", positive=True
    )
    required_high = _finite(
        required_wavelength_range_um[1], "required wavelength maximum", positive=True
    )
    if low > required_low or high < required_high:
        raise ValidationError(
            "material validation wavelength range does not cover the requested source band"
        )
    _positive_int(proof.get("sample_count"), "material validation sample_count")
    error = _mapping(proof.get("error"), "material validation error")
    if not isinstance(error.get("norm"), str) or not error["norm"]:
        raise ValidationError("material validation error.norm is required")
    tolerance = _finite(
        error.get("tolerance"), "material validation tolerance", positive=True
    )
    measured = _finite(error.get("measured"), "material validation measured")
    if measured < 0 or measured > tolerance or error.get("passed") is not True:
        raise ValidationError("material validation artifact does not prove tolerance")
    materials = _mapping(proof.get("materials"), "material validation materials")
    if not {"Si", "SiO2"}.issubset(materials):
        raise ValidationError("material validation must cover Si and SiO2")
    for name in ("Si", "SiO2"):
        record = _mapping(materials[name], f"material validation {name}")
        if _finite(record.get("max_absolute_error"), f"{name}.max_absolute_error") < 0:
            raise ValidationError(f"{name}.max_absolute_error must be non-negative")
        relative_error = _finite(
            record.get("max_relative_error"), f"{name}.max_relative_error"
        )
        if relative_error < 0:
            raise ValidationError(f"{name}.max_relative_error must be non-negative")
        if relative_error > tolerance:
            raise ValidationError(
                f"material validation {name} exceeds the declared tolerance"
            )
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path.resolve()),
        "required_wavelength_range_um": [required_low, required_high],
        "artifact": proof,
    }


def _paper_workload(raw: Mapping[str, Any]) -> Dict[str, Any]:
    result = json.loads(json.dumps(raw))
    for solver in ("tidy3d", "lumerical"):
        workload = result[solver]
        rounded = workload.pop("grid_points")
        shape = workload["grid_shape"]
        workload["reported_grid_points_rounded"] = rounded
        workload["grid_points_from_shape"] = math.prod(shape)
        workload["reported_grid_points_denominator"] = 1_000_000
    return result


def _expand_monitors(
    case_definitions: Mapping[str, Any], runner_case: Mapping[str, Any]
) -> Sequence[Dict[str, Any]]:
    """Apply sampling defaults only to monitors that directly sample fields."""
    defaults = _mapping(case_definitions["monitor_defaults"], "monitor_defaults")
    return [
        {**defaults, **monitor} if monitor.get("kind") == "mode" else dict(monitor)
        for monitor in _sequence(runner_case["monitors"], "case monitors")
    ]


def _resolved_monitor_sampling(
    monitors: Sequence[Mapping[str, Any]],
    wavelength_min_um: float,
    wavelength_max_um: float,
) -> Dict[str, Any]:
    """Resolve wavelength-domain monitor sampling to an explicit shared grid."""
    sampled = [monitor for monitor in monitors if monitor.get("kind") == "mode"]
    if not sampled:
        raise ValidationError("at least one mode monitor is required")
    steps_nm = {
        _finite(
            _mapping(
                monitor.get("frequency_sampling"), "monitor frequency_sampling"
            ).get("step_nm"),
            "monitor frequency step_nm",
            positive=True,
        )
        for monitor in sampled
    }
    step_nm = min(steps_nm)
    span_nm = (wavelength_max_um - wavelength_min_um) * 1000.0
    intervals = int(round(span_nm / step_nm))
    if not math.isclose(intervals * step_nm, span_nm, rel_tol=0.0, abs_tol=1e-9):
        raise ValidationError(
            "monitor wavelength step does not exactly divide the band"
        )
    wavelengths = [
        wavelength_min_um + index * step_nm / 1000.0 for index in range(intervals + 1)
    ]
    wavelengths[-1] = wavelength_max_um
    frequencies = [1.0 / wavelength for wavelength in wavelengths]
    return {
        "domain": "wavelength",
        "step_nm": step_nm,
        "inclusive_endpoints": True,
        "wavelengths_um": wavelengths,
        "frequencies_meep": frequencies,
        "frequency_order": "decreasing_with_increasing_wavelength",
        "dft_decimation_factor": int(sampled[0]["dft_decimation_factor"]),
    }


def build_manifest(
    *,
    reference: Mapping[str, Any],
    reference_path: pathlib.Path,
    case_definitions: Mapping[str, Any],
    cases_path: pathlib.Path,
    result_schema_path: pathlib.Path,
    checkout_validation: Mapping[str, Any],
    device_name: str,
    mode: str,
    cells_per_material_wavelength: float,
    n_max: float,
    lambda_min_um: float,
    rounding: str,
    bandwidth_nm: float,
    material_mode: str,
    material_validation: Optional[pathlib.Path],
    precision: str,
    backend: str,
    ranks: int,
    mpi_transport: str,
    steps: Optional[int],
    max_steps: Optional[int],
    auto_shutoff_threshold: float,
    decay_check_interval_meep: float,
    overlap: str = "off",
    graph: str = "eager",
) -> Dict[str, Any]:
    if ranks > 1 and pathlib.Path(result_schema_path) == DEFAULT_RESULT_SCHEMA:
        result_schema_path = DEFAULT_MPI_RESULT_SCHEMA
    validate_reference(reference)
    validate_case_definitions(case_definitions)
    if device_name not in reference["devices"]:
        raise ValidationError(f"unknown device {device_name!r}")
    bandwidth = _finite(bandwidth_nm, "bandwidth_nm", positive=True)
    if bandwidth not in reference["common"]["bandwidth_sweep_nm"]:
        raise ValidationError(
            "bandwidth must be one of the paper's 20 nm or 50 nm settings"
        )
    cells = _finite(
        cells_per_material_wavelength,
        "cells_per_material_wavelength",
        positive=True,
    )
    if cells not in reference["common"]["cells_per_material_wavelength"]:
        raise ValidationError(
            "cells per material wavelength must be one of the paper's 6, 10, 15, 20, or 25 settings"
        )
    validate_execution(backend, precision, ranks, mpi_transport)
    if overlap not in OVERLAPS:
        raise ValidationError(f"overlap must be one of {sorted(OVERLAPS)}")
    if graph not in GRAPHS:
        raise ValidationError(f"graph must be one of {sorted(GRAPHS)}")
    if material_mode not in {"paper", "performance-adaptation"}:
        raise ValidationError("unknown material mode")
    if device_name not in checkout_validation.get("validated_cases", []):
        raise ValidationError(
            f"checkout did not validate selected case {device_name!r}"
        )

    device = reference["devices"][device_name]
    runner_case = case_definitions["cases"][device_name]
    resolution = convert_resolution(cells, n_max, lambda_min_um, rounding)
    stopping = build_stopping_spec(
        mode, steps, max_steps, auto_shutoff_threshold, decay_check_interval_meep
    )
    center_wavelength_um = _finite(
        reference["common"]["center_wavelength_um"],
        "center_wavelength_um",
        positive=True,
    )
    source_half_width_um = 0.5 * bandwidth / 1000.0
    source_wavelength_min_um = center_wavelength_um - source_half_width_um
    source_wavelength_max_um = center_wavelength_um + source_half_width_um
    if source_wavelength_min_um <= 0:
        raise ValidationError("source wavelength band must be positive")
    required_material_min_um = min(
        resolution["lambda_min_um"], source_wavelength_min_um
    )
    required_material_range_um = [
        required_material_min_um,
        source_wavelength_max_um,
    ]
    proof = None
    if material_mode == "paper":
        if material_validation is None:
            raise ValidationError(
                "paper material mode requires --material-validation with an equivalence proof"
            )
        proof = load_material_validation(
            material_validation,
            reference_path,
            reference,
            required_material_range_um,
        )
    elif material_validation is not None:
        raise ValidationError(
            "--material-validation is only valid with --material-mode paper"
        )
    source = {**case_definitions["source_defaults"], **runner_case["source"]}
    source["spectral_envelope"] = {
        **case_definitions["source_defaults"]["spectral_envelope"],
        "wavelength_min_um": source_wavelength_min_um,
        "wavelength_max_um": source_wavelength_max_um,
        "center_frequency_meep": 1.0 / center_wavelength_um,
        "fwidth_meep": 1.0 / source_wavelength_min_um - 1.0 / source_wavelength_max_um,
    }
    generated = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    case = {
        "id": device_name,
        "display_name": device["display_name"],
        "dimensions": reference["common"]["dimensions"],
        "paper_domain_um": list(device["domain_um"]),
        "geometry_summary": device["geometry"],
        "geometry_import": dict(case_definitions["geometry_import"]),
        "layers": [
            dict(layer)
            for layer in case_definitions["layers"]
            if layer["name"] in runner_case["active_layers"]
        ],
        "cell": dict(case_definitions["cell"]),
        "time_stepping": dict(case_definitions["time_stepping"]),
        "boundaries": dict(case_definitions["boundaries"]),
        "ports": dict(runner_case["ports"]),
        "source": source,
        "monitors": _expand_monitors(case_definitions, runner_case),
        "decay_stop": dict(case_definitions["decay_stop"]),
        "gds_cell_name": runner_case["gds_cell_name"],
        "cladding": device["cladding"],
        "top_cladding": runner_case.get("top_cladding"),
        "runner_support": dict(
            runner_case.get("runner_support", {"supported": True, "reason": None})
        ),
        "paper_input": device["input"],
        "paper_target": device["target"],
        "symmetry": reference["common"]["symmetry"],
    }
    monitor_by_name = {monitor["name"]: monitor for monitor in case["monitors"]}
    monitor_sampling = _resolved_monitor_sampling(
        case["monitors"], source_wavelength_min_um, source_wavelength_max_um
    )
    tolerance = case_definitions["observable_policy"]["tolerances_by_precision"][
        precision
    ]
    required_observables = [
        {
            "name": observable["name"],
            "monitor": dict(monitor_by_name[observable["monitor"]]),
            "unit": observable["unit"],
            "evaluation": observable["evaluation"],
            "absolute_tolerance": tolerance["absolute"],
            "relative_tolerance": tolerance["relative"],
        }
        for observable in runner_case["required_observables"]
    ]
    assets = checkout_validation["assets"]
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_schema": {
            "path": str(DEFAULT_MANIFEST_SCHEMA.resolve()),
            "sha256": sha256_file(DEFAULT_MANIFEST_SCHEMA.resolve()),
            "schema_version": MANIFEST_SCHEMA_VERSION,
        },
        "generator": {
            "name": "paper_2506_16665.benchmark_manifest",
            "version": GENERATOR_VERSION,
            "generated_at_utc": generated,
        },
        "reference": {
            "path": str(reference_path.resolve()),
            "sha256": sha256_file(reference_path.resolve()),
            "paper": dict(reference["paper"]),
        },
        "case_definitions": {
            "path": str(cases_path.resolve()),
            "sha256": sha256_file(cases_path.resolve()),
        },
        "result_schema": {
            "path": str(result_schema_path.resolve()),
            "sha256": sha256_file(result_schema_path.resolve()),
            "schema_version": load_json_object(result_schema_path, "result schema")[
                "properties"
            ]["schema_version"]["const"],
        },
        "input_checkout": {
            "path": checkout_validation["checkout"],
            "commit": checkout_validation["actual_commit"],
            "dirty": checkout_validation["dirty"],
            "status_porcelain": list(checkout_validation["status_porcelain"]),
            "gds": dict(assets["gds"][device_name]),
            "yaml": dict(assets["yaml"][device_name]),
            "layer_stack": dict(assets["layer_stack"]),
        },
        "case": case,
        "excitation": {
            "center_wavelength_um": center_wavelength_um,
            "bandwidth_nm": bandwidth,
            "source_wavelength_min_um": source_wavelength_min_um,
            "source_wavelength_max_um": source_wavelength_max_um,
            "lambda_min_um": resolution["lambda_min_um"],
            "resolution_n_max_wavelength_um": resolution["lambda_min_um"],
            "monitor_sampling": monitor_sampling,
        },
        "discretization": resolution,
        "materials": {
            "mode": material_mode,
            "paper_parameters": (
                dict(reference["materials"]) if material_mode == "paper" else None
            ),
            "validation": proof,
            "performance_adaptation": (
                dict(case_definitions["performance_adaptation_materials"])
                if material_mode == "performance-adaptation"
                else None
            ),
            "caveat": reference["common"]["material_mode_caveat"],
        },
        "execution": {
            "requested": {
                "mode": mode,
                "backend": backend,
                "precision": precision,
                "ranks": ranks,
                "mpi_transport": mpi_transport,
                "overlap": overlap,
                "graph": graph,
            },
            "resolution_required_at_runtime": backend == "auto"
            or (ranks > 1 and mpi_transport == "auto"),
            "requested_gpus": ranks if backend == "nvidia" else 0,
            "warmup_steps": 1 if mode == "smoke" else 100,
            "measured_repetitions": 1 if mode == "smoke" else 5,
        },
        "stopping": stopping,
        "validation_policy": {
            "schema_version": case_definitions["observable_policy"]["schema_version"],
            "reference": {
                "kind": case_definitions["observable_policy"]["reference_kind"],
                "artifact_schema_version": case_definitions["observable_policy"][
                    "reference_artifact_schema_version"
                ],
                "physics_configuration_sha256": "",
                "required_bindings": [
                    "paper_reference_sha256",
                    "case_definitions_sha256",
                    "case_id",
                    "physics_configuration_sha256",
                ],
            },
            "required_observables": required_observables,
        },
        "paper_comparison": {
            "timing_row": dict(device["timings"][str(int(cells))]),
            "workload_25": _paper_workload(device["workload_25"]),
            "paper_results": dict(device["paper_results"]),
            "wall_time_is_stretch_goal": True,
        },
        "adaptations": [
            reference["paper"]["code_snapshot_caveat"],
            reference["common"]["mesh_caveat"],
            reference["common"]["material_mode_caveat"],
            reference["common"]["boundary_caveat"],
            MATERIAL_DISCRETIZATION_ADAPTATION,
            PRISM_VERTEX_CANONICALIZATION_ADAPTATION,
        ],
    }
    manifest["validation_policy"]["reference"]["physics_configuration_sha256"] = (
        physics_configuration_sha256(manifest)
    )
    manifest_schema = load_json_object(DEFAULT_MANIFEST_SCHEMA, "run manifest schema")
    _validate_schema_structure(
        manifest, manifest_schema, manifest_schema, "generated run manifest"
    )
    return manifest


def _validate_run_manifest_for_result(
    manifest: Mapping[str, Any], label: str = "run manifest"
) -> Dict[str, Any]:
    manifest_schema = load_json_object(DEFAULT_MANIFEST_SCHEMA, "run manifest schema")
    _validate_schema_structure(manifest, manifest_schema, manifest_schema, label)
    bundled_reference = load_reference(DEFAULT_REFERENCE)
    bundled_cases = load_case_definitions(DEFAULT_CASES)

    schema_record = _mapping(
        manifest.get("manifest_schema"), f"{label}.manifest_schema"
    )
    expected_manifest_schema_hash = sha256_file(DEFAULT_MANIFEST_SCHEMA)
    if schema_record.get("sha256") != expected_manifest_schema_hash:
        raise ValidationError(
            f"{label}.manifest_schema hash does not match the validator"
        )
    recorded_schema_path = pathlib.Path(str(schema_record.get("path"))).resolve()
    if not recorded_schema_path.is_file() or sha256_file(
        recorded_schema_path
    ) != schema_record.get("sha256"):
        raise ValidationError(f"{label}.manifest_schema path/hash is not authentic")
    reference_record = _mapping(manifest.get("reference"), f"{label}.reference")
    expected_reference_hash = sha256_file(DEFAULT_REFERENCE)
    if reference_record.get("sha256") != expected_reference_hash:
        raise ValidationError(
            f"{label}.reference hash does not match the bundled corpus"
        )
    recorded_reference_path = pathlib.Path(str(reference_record.get("path"))).resolve()
    if not recorded_reference_path.is_file() or sha256_file(
        recorded_reference_path
    ) != reference_record.get("sha256"):
        raise ValidationError(f"{label}.reference path/hash is not authentic")
    if not _json_equal(reference_record.get("paper"), bundled_reference["paper"]):
        raise ValidationError(
            f"{label}.reference.paper does not match the bundled corpus"
        )
    cases_record = _mapping(
        manifest.get("case_definitions"), f"{label}.case_definitions"
    )
    expected_cases_hash = sha256_file(DEFAULT_CASES)
    if cases_record.get("sha256") != expected_cases_hash:
        raise ValidationError(
            f"{label}.case_definitions hash does not match the bundled corpus"
        )
    recorded_cases_path = pathlib.Path(str(cases_record.get("path"))).resolve()
    if not recorded_cases_path.is_file() or sha256_file(
        recorded_cases_path
    ) != cases_record.get("sha256"):
        raise ValidationError(f"{label}.case_definitions path/hash is not authentic")

    case = _mapping(manifest.get("case"), f"{label}.case")
    case_id = case.get("id")
    if case_id not in bundled_reference["devices"]:
        raise ValidationError(f"{label}.case.id is not in the bundled corpus")
    reference_device = bundled_reference["devices"][case_id]
    runner_case = bundled_cases["cases"][case_id]
    monitors = _sequence(case.get("monitors"), f"{label}.case.monitors")
    result_schema = _mapping(manifest.get("result_schema"), f"{label}.result_schema")
    recorded_result_schema_version = result_schema.get("schema_version")
    expected_result_schema_path = (
        DEFAULT_MPI_RESULT_SCHEMA
        if recorded_result_schema_version == 2
        else DEFAULT_RESULT_SCHEMA
    )
    _require_schema_version(
        recorded_result_schema_version,
        (
            2
            if expected_result_schema_path == DEFAULT_MPI_RESULT_SCHEMA
            else RESULT_SCHEMA_VERSION
        ),
        f"{label}.result_schema schema_version",
    )
    expected_result_schema_hash = sha256_file(expected_result_schema_path)
    if result_schema.get("sha256") != expected_result_schema_hash:
        raise ValidationError(
            f"{label}.result_schema hash does not match the validator"
        )
    recorded_result_schema_path = pathlib.Path(str(result_schema.get("path"))).resolve()
    if not recorded_result_schema_path.is_file() or sha256_file(
        recorded_result_schema_path
    ) != result_schema.get("sha256"):
        raise ValidationError(f"{label}.result_schema path/hash is not authentic")
    input_checkout = _mapping(manifest.get("input_checkout"), f"{label}.input_checkout")
    if input_checkout.get("commit") != bundled_reference["paper"]["code_commit"]:
        raise ValidationError(f"{label}.input_checkout commit is not the pinned corpus")
    if input_checkout.get("dirty") != bool(input_checkout.get("status_porcelain")):
        raise ValidationError(f"{label}.input_checkout dirty status is inconsistent")
    checkout_path = pathlib.Path(str(input_checkout.get("path"))).resolve()
    for asset_name, expected in (
        (
            "gds",
            {
                "relative_path": reference_device["gds_relative_path"],
                "sha256": reference_device["gds_sha256"],
            },
        ),
        (
            "yaml",
            {
                "relative_path": runner_case["yaml_relative_path"],
                "sha256": runner_case["yaml_sha256"],
            },
        ),
        ("layer_stack", bundled_cases["assets"]["layer_stack"]),
    ):
        asset = _mapping(
            input_checkout.get(asset_name), f"{label}.input_checkout.{asset_name}"
        )
        for key in ("relative_path", "sha256"):
            if asset.get(key) != expected[key]:
                raise ValidationError(
                    f"{label}.input_checkout.{asset_name}.{key} does not match the bundled corpus"
                )
        if (
            pathlib.Path(str(asset.get("path"))).resolve()
            != (checkout_path / str(asset["relative_path"])).resolve()
        ):
            raise ValidationError(
                f"{label}.input_checkout.{asset_name}.path is inconsistent"
            )

    execution = _mapping(manifest.get("execution"), f"{label}.execution")
    requested = _mapping(execution.get("requested"), f"{label}.execution.requested")
    if requested.get("mode") not in MODES:
        raise ValidationError(f"{label} execution mode is invalid")
    validate_execution(
        requested.get("backend"),
        requested.get("precision"),
        requested.get("ranks"),
        requested.get("mpi_transport"),
    )
    if requested.get("overlap") not in OVERLAPS:
        raise ValidationError(f"{label} execution overlap policy is invalid")
    if requested.get("graph") not in GRAPHS:
        raise ValidationError(f"{label} execution graph policy is invalid")
    _positive_int(
        execution.get("measured_repetitions"),
        f"{label}.execution.measured_repetitions",
    )
    _positive_int(
        execution.get("warmup_steps"),
        f"{label}.execution.warmup_steps",
        allow_zero=True,
    )
    expected_warmup = 1 if requested["mode"] == "smoke" else 100
    expected_repetitions = 1 if requested["mode"] == "smoke" else 5
    if execution["warmup_steps"] != expected_warmup:
        raise ValidationError(f"{label}.execution.warmup_steps is not canonical")
    if execution["measured_repetitions"] != expected_repetitions:
        raise ValidationError(
            f"{label}.execution.measured_repetitions is not canonical"
        )
    expected_gpus = requested["ranks"] if requested["backend"] == "nvidia" else 0
    if execution.get("requested_gpus") != expected_gpus:
        raise ValidationError(f"{label}.execution.requested_gpus is inconsistent")
    expected_resolution_required = requested["backend"] == "auto" or (
        requested["ranks"] > 1 and requested["mpi_transport"] == "auto"
    )
    if execution.get("resolution_required_at_runtime") != expected_resolution_required:
        raise ValidationError(f"{label}.execution resolution flag is inconsistent")

    stopping = _mapping(manifest.get("stopping"), f"{label}.stopping")
    if stopping.get("kind") == "fixed_steps":
        _positive_int(stopping.get("steps"), f"{label}.stopping.steps")
    elif stopping.get("kind") == "field_energy_decay":
        _positive_int(stopping.get("max_steps"), f"{label}.stopping.max_steps")
    else:
        raise ValidationError(f"{label}.stopping.kind is invalid")

    excitation = _mapping(manifest.get("excitation"), f"{label}.excitation")
    center = _finite(
        excitation.get("center_wavelength_um"),
        f"{label}.excitation.center_wavelength_um",
        positive=True,
    )
    bandwidth = _finite(
        excitation.get("bandwidth_nm"),
        f"{label}.excitation.bandwidth_nm",
        positive=True,
    )
    expected_low = center - 0.5 * bandwidth / 1000.0
    expected_high = center + 0.5 * bandwidth / 1000.0
    if center != bundled_reference["common"]["center_wavelength_um"]:
        raise ValidationError(f"{label}.excitation center does not match the corpus")
    if (
        excitation.get("source_wavelength_min_um") != expected_low
        or excitation.get("source_wavelength_max_um") != expected_high
    ):
        raise ValidationError(f"{label}.excitation wavelength band is inconsistent")
    expected_sampling = _resolved_monitor_sampling(
        _expand_monitors(bundled_cases, runner_case), expected_low, expected_high
    )
    if not _json_equal(excitation.get("monitor_sampling"), expected_sampling):
        raise ValidationError(f"{label}.excitation monitor sampling is inconsistent")

    discretization = _mapping(manifest.get("discretization"), f"{label}.discretization")
    expected_discretization = convert_resolution(
        discretization.get("cells_per_material_wavelength"),
        discretization.get("n_max"),
        discretization.get("lambda_min_um"),
        discretization.get("rounding"),
    )
    if not _json_equal(discretization, expected_discretization):
        raise ValidationError(f"{label}.discretization is internally inconsistent")
    if (
        excitation.get("lambda_min_um") != discretization["lambda_min_um"]
        or excitation.get("resolution_n_max_wavelength_um")
        != discretization["lambda_min_um"]
    ):
        raise ValidationError(f"{label}.excitation does not match discretization")

    expected_monitors = _expand_monitors(bundled_cases, runner_case)
    expected_source = {**bundled_cases["source_defaults"], **runner_case["source"]}
    expected_source["spectral_envelope"] = {
        **bundled_cases["source_defaults"]["spectral_envelope"],
        "wavelength_min_um": expected_low,
        "wavelength_max_um": expected_high,
        "center_frequency_meep": 1.0 / center,
        "fwidth_meep": 1.0 / expected_low - 1.0 / expected_high,
    }
    expected_case_fields = {
        "display_name": reference_device["display_name"],
        "dimensions": bundled_reference["common"]["dimensions"],
        "paper_domain_um": reference_device["domain_um"],
        "geometry_summary": reference_device["geometry"],
        "geometry_import": bundled_cases["geometry_import"],
        "layers": [
            layer
            for layer in bundled_cases["layers"]
            if layer["name"] in runner_case["active_layers"]
        ],
        "cell": bundled_cases["cell"],
        "time_stepping": bundled_cases["time_stepping"],
        "boundaries": bundled_cases["boundaries"],
        "ports": runner_case["ports"],
        "source": expected_source,
        "monitors": expected_monitors,
        "decay_stop": bundled_cases["decay_stop"],
        "gds_cell_name": runner_case["gds_cell_name"],
        "cladding": reference_device["cladding"],
        "top_cladding": runner_case.get("top_cladding"),
        "runner_support": runner_case.get(
            "runner_support", {"supported": True, "reason": None}
        ),
        "paper_input": reference_device["input"],
        "paper_target": reference_device["target"],
        "symmetry": bundled_reference["common"]["symmetry"],
    }
    for key, expected in expected_case_fields.items():
        if not _json_equal(case.get(key), expected):
            raise ValidationError(
                f"{label}.case.{key} does not match the bundled corpus"
            )

    materials = _mapping(manifest.get("materials"), f"{label}.materials")
    if materials.get("caveat") != bundled_reference["common"]["material_mode_caveat"]:
        raise ValidationError(f"{label}.materials caveat is not canonical")
    if materials.get("mode") == "performance-adaptation":
        if (
            materials.get("paper_parameters") is not None
            or materials.get("validation") is not None
        ):
            raise ValidationError(
                f"{label}.materials adaptation contains paper-only data"
            )
        if not _json_equal(
            materials.get("performance_adaptation"),
            bundled_cases["performance_adaptation_materials"],
        ):
            raise ValidationError(
                f"{label}.materials performance adaptation is not canonical"
            )
    elif materials.get("mode") == "paper":
        if not _json_equal(
            materials.get("paper_parameters"), bundled_reference["materials"]
        ):
            raise ValidationError(
                f"{label}.materials parameters do not match the paper"
            )
        validation = _mapping(
            materials.get("validation"), f"{label}.materials.validation"
        )
        proof_path = pathlib.Path(str(validation.get("path")))
        if sha256_file(proof_path) != validation.get("sha256"):
            raise ValidationError(f"{label}.materials validation hash mismatch")
        if materials.get("performance_adaptation") is not None:
            raise ValidationError(
                f"{label}.materials paper mode contains adaptation constants"
            )
    else:
        raise ValidationError(f"{label}.materials mode is invalid")

    tolerance = bundled_cases["observable_policy"]["tolerances_by_precision"][
        requested["precision"]
    ]
    monitor_by_name = {monitor["name"]: monitor for monitor in expected_monitors}
    expected_observables = [
        {
            "name": observable["name"],
            "monitor": monitor_by_name[observable["monitor"]],
            "unit": observable["unit"],
            "evaluation": observable["evaluation"],
            "absolute_tolerance": tolerance["absolute"],
            "relative_tolerance": tolerance["relative"],
        }
        for observable in runner_case["required_observables"]
    ]
    validation_policy = _mapping(
        manifest.get("validation_policy"), f"{label}.validation_policy"
    )
    expected_policy = {
        "schema_version": 1,
        "reference": {
            "kind": "cpu_native_baseline",
            "artifact_schema_version": 1,
            "physics_configuration_sha256": physics_configuration_sha256(manifest),
            "required_bindings": [
                "paper_reference_sha256",
                "case_definitions_sha256",
                "case_id",
                "physics_configuration_sha256",
            ],
        },
        "required_observables": expected_observables,
    }
    if not _json_equal(validation_policy, expected_policy):
        raise ValidationError(f"{label}.validation_policy is not canonical")

    expected_paper_comparison = {
        "timing_row": dict(
            reference_device["timings"][
                str(int(discretization["cells_per_material_wavelength"]))
            ]
        ),
        "workload_25": _paper_workload(reference_device["workload_25"]),
        "paper_results": dict(reference_device["paper_results"]),
        "wall_time_is_stretch_goal": True,
    }
    if not _json_equal(manifest.get("paper_comparison"), expected_paper_comparison):
        raise ValidationError(f"{label}.paper_comparison is not canonical")
    expected_adaptations = [
        bundled_reference["paper"]["code_snapshot_caveat"],
        bundled_reference["common"]["mesh_caveat"],
        bundled_reference["common"]["material_mode_caveat"],
        bundled_reference["common"]["boundary_caveat"],
        MATERIAL_DISCRETIZATION_ADAPTATION,
        PRISM_VERTEX_CANONICALIZATION_ADAPTATION,
    ]
    if not _json_equal(manifest.get("adaptations"), expected_adaptations):
        raise ValidationError(f"{label}.adaptations are not canonical")
    return {
        "case_id": case_id,
        "monitors": monitors,
        "execution": execution,
        "requested": requested,
        "stopping": stopping,
        "validation_policy": validation_policy,
        "paper_reference_sha256": expected_reference_hash,
        "case_definitions_sha256": expected_cases_hash,
        "physics_configuration_sha256": expected_policy["reference"][
            "physics_configuration_sha256"
        ],
    }


def _load_physics_reference(
    path: pathlib.Path,
    expected_hash: str,
    manifest_details: Mapping[str, Any],
) -> Dict[str, Mapping[str, Any]]:
    _validate_digest(expected_hash, "physics reference")
    if expected_hash == "0" * 64:
        raise ValidationError(
            "successful result requires a real physics reference hash"
        )
    artifact = load_json_object(path, "physics reference artifact")
    if sha256_file(path) != expected_hash:
        raise ValidationError("physics reference artifact hash mismatch")
    _require_schema_version(
        artifact.get("schema_version"), 1, "physics reference schema_version"
    )
    if artifact.get("kind") != "cpu_native_baseline":
        raise ValidationError("physics reference kind is invalid")
    expected_bindings = {
        "paper_reference_sha256": manifest_details["paper_reference_sha256"],
        "case_definitions_sha256": manifest_details["case_definitions_sha256"],
        "case_id": manifest_details["case_id"],
        "physics_configuration_sha256": manifest_details[
            "physics_configuration_sha256"
        ],
    }
    for key, expected in expected_bindings.items():
        if artifact.get(key) != expected:
            raise ValidationError(
                f"physics reference {key} does not match run manifest"
            )
    observables = _sequence(
        artifact.get("observables"), "physics reference observables"
    )
    by_name: Dict[str, Mapping[str, Any]] = {}
    for index, raw_observable in enumerate(observables):
        observable = _mapping(raw_observable, f"physics reference observables[{index}]")
        name = observable.get("name")
        monitor = observable.get("monitor")
        unit = observable.get("unit")
        if not isinstance(name, str) or not name:
            raise ValidationError("physics reference observable name is required")
        if name in by_name:
            raise ValidationError("physics reference observable names must be unique")
        if not isinstance(monitor, str) or not monitor:
            raise ValidationError(
                f"physics reference observable {name} monitor is required"
            )
        if not isinstance(unit, str) or not unit:
            raise ValidationError(
                f"physics reference observable {name} unit is required"
            )
        _finite(observable.get("value"), f"physics reference observable {name} value")
        by_name[name] = observable
    expected_names = {
        policy["name"]
        for policy in manifest_details["validation_policy"]["required_observables"]
    }
    if set(by_name) != expected_names:
        raise ValidationError(
            "physics reference observable set does not match the run manifest"
        )
    return by_name


def build_result_template(manifest_path: pathlib.Path) -> Dict[str, Any]:
    manifest = load_json_object(manifest_path, "run manifest")
    manifest_details = _validate_run_manifest_for_result(manifest)
    requested = manifest_details["requested"]
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path.resolve()),
            "case_id": manifest_details["case_id"],
        },
        "physics_reference": {"path": "", "sha256": "0" * 64},
        "status": {"exit_code": -1, "succeeded": False, "errors": ["not run"]},
        "provenance": {
            "meep": {
                "commit": "0" * 40,
                "dirty": False,
                "configure_flags": [],
                "build_flags": [],
            },
            "invocation": {"argv": [], "cwd": "", "environment": {}},
            "requested_execution": dict(requested),
            "resolved_execution": dict(requested),
            "cuda": {
                "driver_version": None,
                "runtime_version": None,
                "toolkit_version": None,
            },
            "mpi": {
                "provider": None,
                "version": None,
                "cuda_aware_query": None,
                "mca": {},
            },
            "rank_devices": [],
        },
        "simulation": {
            "grid_shape": [1, 1, 1],
            "grid_points_exact": 1,
            "courant_factor": 0.5,
            "dt_meep": 1.0,
            "steps": 0,
            "physical_time_meep": 0.0,
        },
        "timing": {
            "initialization_seconds": 0.0,
            "graph_build_seconds": 0.0,
            "warmup_steps": 0,
            "steady_state_repetitions_seconds": [],
            "steady_state_min_seconds": 0.0,
            "steady_state_median_seconds": 0.0,
            "steady_state_max_seconds": 0.0,
            "kernel_seconds": 0.0,
            "pack_seconds": 0.0,
            "unpack_seconds": 0.0,
            "staging_d2h_seconds": 0.0,
            "staging_h2d_seconds": 0.0,
            "mpi_progress_seconds": 0.0,
            "mpi_wait_seconds": 0.0,
        },
        "performance": {
            "grid_timesteps_per_second": 0.0,
            "component_updates_per_second": 0.0,
            "component_updates_per_grid_timestep": 0.0,
            "rate_basis": "steady_state_median_seconds",
        },
        "memory": {
            "peak_host_bytes": 0,
            "peak_device_bytes": 0,
            "device_allocation_count": 0,
        },
        "counters": {
            "h2d_bytes": 0,
            "d2h_bytes": 0,
            "halo_bytes": 0,
            "full_field_copy_count": 0,
            "fallback_count": 0,
            "graph_launches": 0,
            "graph_launches_per_step": 0.0,
        },
        "physics_observables": [],
    }


def validate_result_document(
    result: Mapping[str, Any], result_path: Optional[pathlib.Path] = None
) -> None:
    result_schema = load_json_object(DEFAULT_RESULT_SCHEMA, "benchmark result schema")
    _validate_schema_structure(result, result_schema, result_schema, "result")
    _require_schema_version(
        result.get("schema_version"), RESULT_SCHEMA_VERSION, "result schema_version"
    )
    run_manifest = _mapping(result.get("run_manifest"), "run_manifest")
    manifest_path_value = run_manifest.get("path")
    if not isinstance(manifest_path_value, str) or not manifest_path_value:
        raise ValidationError("run_manifest.path must be a nonempty string")
    expected_manifest_hash = _validate_digest(
        run_manifest.get("sha256"), "run manifest"
    )
    manifest_path = pathlib.Path(manifest_path_value)
    if not manifest_path.is_absolute() and result_path is not None:
        manifest_path = result_path.resolve().parent / manifest_path
    manifest_path = manifest_path.resolve()
    manifest = load_json_object(manifest_path, "run manifest")
    actual_manifest_hash = sha256_file(manifest_path)
    if actual_manifest_hash != expected_manifest_hash:
        raise ValidationError(
            "run manifest hash mismatch: "
            f"got {actual_manifest_hash}, expected {expected_manifest_hash}"
        )
    manifest_details = _validate_run_manifest_for_result(manifest)
    if run_manifest.get("case_id") != manifest_details["case_id"]:
        raise ValidationError("result case_id does not match the run manifest")
    physics_reference_record = _mapping(
        result.get("physics_reference"), "physics_reference"
    )
    status = _mapping(result.get("status"), "status")
    if not isinstance(status.get("exit_code"), int) or isinstance(
        status.get("exit_code"), bool
    ):
        raise ValidationError("status.exit_code must be an integer")
    if not isinstance(status.get("succeeded"), bool):
        raise ValidationError("status.succeeded must be boolean")
    errors = _sequence(status.get("errors"), "status.errors")
    if not all(isinstance(error, str) for error in errors):
        raise ValidationError("status.errors must contain strings")
    if status["succeeded"] != (status["exit_code"] == 0 and not errors):
        raise ValidationError("status fields are inconsistent")

    provenance = _mapping(result.get("provenance"), "provenance")
    meep = _mapping(provenance.get("meep"), "provenance.meep")
    commit = meep.get("commit")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise ValidationError("provenance.meep.commit must be a 40-character SHA")
    if not isinstance(meep.get("dirty"), bool):
        raise ValidationError("provenance.meep.dirty must be boolean")
    for key in ("configure_flags", "build_flags"):
        values = _sequence(meep.get(key), f"provenance.meep.{key}")
        if not all(isinstance(value, str) for value in values):
            raise ValidationError(f"provenance.meep.{key} must contain strings")
    invocation = _mapping(provenance.get("invocation"), "provenance.invocation")
    argv = _sequence(invocation.get("argv"), "provenance.invocation.argv")
    if not all(isinstance(value, str) for value in argv):
        raise ValidationError("provenance.invocation.argv must contain strings")
    if not isinstance(invocation.get("cwd"), str):
        raise ValidationError("provenance.invocation.cwd must be a string")
    environment = _mapping(
        invocation.get("environment"), "provenance.invocation.environment"
    )
    if not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in environment.items()
    ):
        raise ValidationError(
            "provenance.invocation.environment must map strings to strings"
        )
    requested = _mapping(
        provenance.get("requested_execution"), "provenance.requested_execution"
    )
    resolved = _mapping(
        provenance.get("resolved_execution"), "provenance.resolved_execution"
    )
    if requested.get("mode") not in MODES or resolved.get("mode") not in MODES:
        raise ValidationError("requested and resolved execution modes must be valid")
    validate_execution(
        requested.get("backend"),
        requested.get("precision"),
        requested.get("ranks"),
        requested.get("mpi_transport"),
    )
    validate_execution(
        resolved.get("backend"),
        resolved.get("precision"),
        resolved.get("ranks"),
        resolved.get("mpi_transport"),
    )
    if not _json_equal(requested, manifest_details["requested"]):
        raise ValidationError(
            "provenance.requested_execution does not match the run manifest"
        )
    for key in ("mode", "precision", "ranks"):
        if resolved.get(key) != requested.get(key):
            raise ValidationError(f"resolved execution changed requested {key}")
    if requested["backend"] != "auto" and resolved["backend"] != requested["backend"]:
        raise ValidationError("resolved backend does not match the requested backend")
    if (
        requested["mpi_transport"] != "auto"
        and resolved["mpi_transport"] != requested["mpi_transport"]
    ):
        raise ValidationError(
            "resolved MPI transport does not match the requested transport"
        )
    cuda = _mapping(provenance.get("cuda"), "provenance.cuda")
    for key in ("driver_version", "runtime_version", "toolkit_version"):
        if key not in cuda or cuda[key] is not None and not isinstance(cuda[key], str):
            raise ValidationError(f"provenance.cuda.{key} must be a string or null")
    mpi = _mapping(provenance.get("mpi"), "provenance.mpi")
    for key in ("provider", "version", "cuda_aware_query", "mca"):
        if key not in mpi:
            raise ValidationError(f"provenance.mpi.{key} is required")
    if mpi["provider"] is not None and not isinstance(mpi["provider"], str):
        raise ValidationError("provenance.mpi.provider must be a string or null")
    if mpi["version"] is not None and not isinstance(mpi["version"], str):
        raise ValidationError("provenance.mpi.version must be a string or null")
    if mpi["cuda_aware_query"] is not None and not isinstance(
        mpi["cuda_aware_query"], bool
    ):
        raise ValidationError("provenance.mpi.cuda_aware_query must be boolean or null")
    _mapping(mpi["mca"], "provenance.mpi.mca")
    rank_devices = _sequence(provenance.get("rank_devices"), "provenance.rank_devices")
    observed_ranks = set()
    observed_uuids = set()
    for index, raw_device in enumerate(rank_devices):
        device = _mapping(raw_device, f"rank_devices[{index}]")
        rank = _positive_int(
            device.get("rank"), f"rank_devices[{index}].rank", allow_zero=True
        )
        _positive_int(
            device.get("visible_device"),
            f"rank_devices[{index}].visible_device",
            allow_zero=True,
        )
        _positive_int(device.get("memory_bytes"), f"rank_devices[{index}].memory_bytes")
        _finite(
            device.get("sm_clock_hz"),
            f"rank_devices[{index}].sm_clock_hz",
            positive=True,
        )
        _finite(
            device.get("memory_clock_hz"),
            f"rank_devices[{index}].memory_clock_hz",
            positive=True,
        )
        for key in ("uuid", "name"):
            if not isinstance(device.get(key), str) or not device[key]:
                raise ValidationError(f"rank_devices[{index}].{key} is required")
        if rank in observed_ranks:
            raise ValidationError("rank_devices contains duplicate ranks")
        if device["uuid"] in observed_uuids:
            raise ValidationError("rank_devices contains duplicate GPU UUIDs")
        observed_ranks.add(rank)
        observed_uuids.add(device["uuid"])
    if status["succeeded"]:
        if commit == "0" * 40:
            raise ValidationError("successful result requires a real Meep commit")
        if not argv or any(not value for value in argv):
            raise ValidationError(
                "successful result requires a nonempty invocation argv"
            )
        if not invocation["cwd"].strip():
            raise ValidationError(
                "successful result requires a nonempty invocation cwd"
            )
        if resolved["backend"] == "auto" or resolved["mpi_transport"] == "auto":
            raise ValidationError(
                "successful results must record concrete resolved execution"
            )
        expected_devices = resolved["ranks"] if resolved["backend"] == "nvidia" else 0
        if len(rank_devices) != expected_devices:
            raise ValidationError("rank_devices must match resolved NVIDIA rank count")
        if resolved["backend"] == "nvidia":
            if observed_ranks != set(range(resolved["ranks"])):
                raise ValidationError(
                    "rank_devices must cover every resolved rank exactly once"
                )
            if any(
                not isinstance(cuda[key], str) or not cuda[key].strip()
                for key in ("driver_version", "runtime_version", "toolkit_version")
            ):
                raise ValidationError(
                    "successful NVIDIA results require CUDA version provenance"
                )
        if resolved["ranks"] > 1:
            if not mpi["provider"] or not mpi["version"]:
                raise ValidationError(
                    "successful multi-rank results require MPI provenance"
                )
        if (
            resolved["mpi_transport"] == "direct"
            and mpi["cuda_aware_query"] is not True
        ):
            raise ValidationError(
                "successful direct MPI requires a positive CUDA-aware query"
            )

    simulation = _mapping(result.get("simulation"), "simulation")
    shape = _sequence(simulation.get("grid_shape"), "simulation.grid_shape")
    if len(shape) != 3:
        raise ValidationError("simulation.grid_shape must have three dimensions")
    for index, extent in enumerate(shape):
        _positive_int(extent, f"simulation.grid_shape[{index}]")
    points = _positive_int(simulation.get("grid_points_exact"), "grid_points_exact")
    if points != math.prod(shape):
        raise ValidationError("grid_points_exact must equal the product of grid_shape")
    _finite(simulation.get("courant_factor"), "courant_factor", positive=True)
    dt_meep = _finite(simulation.get("dt_meep"), "dt_meep", positive=True)
    simulation_steps = _positive_int(simulation.get("steps"), "steps", allow_zero=True)
    physical_time_meep = _finite(
        simulation.get("physical_time_meep"), "physical_time_meep"
    )
    if physical_time_meep < 0:
        raise ValidationError("physical_time_meep must be non-negative")

    timing = _mapping(result.get("timing"), "timing")
    timing_fields = (
        "initialization_seconds",
        "graph_build_seconds",
        "steady_state_min_seconds",
        "steady_state_median_seconds",
        "steady_state_max_seconds",
        "kernel_seconds",
        "pack_seconds",
        "unpack_seconds",
        "staging_d2h_seconds",
        "staging_h2d_seconds",
        "mpi_progress_seconds",
        "mpi_wait_seconds",
    )
    required_timing = set(timing_fields) | {
        "warmup_steps",
        "steady_state_repetitions_seconds",
    }
    missing_timing = required_timing.difference(timing)
    if missing_timing:
        raise ValidationError(f"timing is missing {sorted(missing_timing)}")
    for field in timing_fields:
        value = _finite(timing.get(field), f"timing.{field}")
        if value < 0:
            raise ValidationError(f"timing.{field} must be non-negative")
    repetitions = _sequence(
        timing.get("steady_state_repetitions_seconds"),
        "timing.steady_state_repetitions_seconds",
    )
    for index, value in enumerate(repetitions):
        if _finite(value, f"steady_state_repetitions_seconds[{index}]") < 0:
            raise ValidationError("steady-state repetitions must be non-negative")
    median_seconds = None
    if repetitions:
        ordered = sorted(float(value) for value in repetitions)
        middle = len(ordered) // 2
        median_seconds = (
            ordered[middle]
            if len(ordered) % 2
            else 0.5 * (ordered[middle - 1] + ordered[middle])
        )
        expected_stats = {
            "steady_state_min_seconds": min(ordered),
            "steady_state_median_seconds": median_seconds,
            "steady_state_max_seconds": max(ordered),
        }
        for key, expected in expected_stats.items():
            if not math.isclose(
                float(timing[key]), expected, rel_tol=1e-12, abs_tol=0.0
            ):
                raise ValidationError(f"timing.{key} does not match raw repetitions")
    warmup_steps = _positive_int(
        timing.get("warmup_steps"), "timing.warmup_steps", allow_zero=True
    )

    performance = _mapping(result.get("performance"), "performance")
    if performance.get("rate_basis") != "steady_state_median_seconds":
        raise ValidationError("performance.rate_basis is invalid")
    grid_timesteps_per_second = _finite(
        performance.get("grid_timesteps_per_second"),
        "performance.grid_timesteps_per_second",
    )
    component_updates_per_second = _finite(
        performance.get("component_updates_per_second"),
        "performance.component_updates_per_second",
    )
    component_updates_per_grid_timestep = _finite(
        performance.get("component_updates_per_grid_timestep"),
        "performance.component_updates_per_grid_timestep",
    )
    for label, value in (
        ("grid_timesteps_per_second", grid_timesteps_per_second),
        ("component_updates_per_second", component_updates_per_second),
        (
            "component_updates_per_grid_timestep",
            component_updates_per_grid_timestep,
        ),
    ):
        if value < 0:
            raise ValidationError(f"performance.{label} must be non-negative")
    required_sections = {
        "memory": {"peak_host_bytes", "peak_device_bytes", "device_allocation_count"},
        "counters": {
            "h2d_bytes",
            "d2h_bytes",
            "halo_bytes",
            "full_field_copy_count",
            "fallback_count",
            "graph_launches",
            "graph_launches_per_step",
        },
    }
    for section_name, required_fields in required_sections.items():
        section = _mapping(result.get(section_name), section_name)
        missing = required_fields.difference(section)
        if missing:
            raise ValidationError(f"{section_name} is missing {sorted(missing)}")
        for field, value in section.items():
            number = _finite(value, f"{section_name}.{field}")
            if number < 0:
                raise ValidationError(f"{section_name}.{field} must be non-negative")
    observables = _sequence(result.get("physics_observables"), "physics_observables")
    manifest_monitor_names = {
        _mapping(monitor, "run manifest monitor").get("name")
        for monitor in manifest_details["monitors"]
    }
    observables_by_name: Dict[str, Mapping[str, Any]] = {}
    for index, raw_observable in enumerate(observables):
        observable = _mapping(raw_observable, f"physics_observables[{index}]")
        for key in ("name", "unit"):
            if not isinstance(observable.get(key), str) or not observable[key]:
                raise ValidationError(f"physics_observables[{index}].{key} is required")
        for key in (
            "value",
            "reference_value",
            "absolute_tolerance",
            "relative_tolerance",
        ):
            number = _finite(observable.get(key), f"physics_observables[{index}].{key}")
            if key.endswith("tolerance") and number < 0:
                raise ValidationError(
                    f"physics_observables[{index}].{key} must be non-negative"
                )
        if not isinstance(observable.get("passed"), bool):
            raise ValidationError(
                f"physics_observables[{index}].passed must be boolean"
            )
        monitor = _mapping(
            observable.get("monitor"), f"physics_observables[{index}].monitor"
        )
        if monitor.get("name") not in manifest_monitor_names:
            raise ValidationError(
                f"physics_observables[{index}] references a monitor outside the run manifest"
            )
        if observable["name"] in observables_by_name:
            raise ValidationError("physics observable names must be unique")
        observables_by_name[observable["name"]] = observable
        error = abs(float(observable["value"]) - float(observable["reference_value"]))
        allowed = float(observable["absolute_tolerance"]) + float(
            observable["relative_tolerance"]
        ) * abs(float(observable["reference_value"]))
        if observable["passed"] != (error <= allowed):
            raise ValidationError(
                f"physics_observables[{index}].passed is inconsistent with its tolerances"
            )

    if status["succeeded"]:
        reference_path_value = physics_reference_record.get("path")
        if not isinstance(reference_path_value, str) or not reference_path_value:
            raise ValidationError("successful result requires a physics reference path")
        reference_path = pathlib.Path(reference_path_value)
        if not reference_path.is_absolute() and result_path is not None:
            reference_path = result_path.resolve().parent / reference_path
        reference_path = reference_path.resolve()
        reference_observables = _load_physics_reference(
            reference_path,
            physics_reference_record.get("sha256"),
            manifest_details,
        )
        policies = {
            policy["name"]: policy
            for policy in manifest_details["validation_policy"]["required_observables"]
        }
        if set(observables_by_name) != set(policies):
            raise ValidationError(
                "physics observable set does not match the run manifest policy"
            )
        for name, policy in policies.items():
            observable = observables_by_name[name]
            reference_observable = reference_observables[name]
            if observable["monitor"] != policy["monitor"]:
                raise ValidationError(
                    f"physics observable {name} monitor is not canonical"
                )
            if observable["unit"] != policy["unit"]:
                raise ValidationError(
                    f"physics observable {name} unit is not canonical"
                )
            for result_key, policy_key in (
                ("absolute_tolerance", "absolute_tolerance"),
                ("relative_tolerance", "relative_tolerance"),
            ):
                if observable[result_key] != policy[policy_key]:
                    raise ValidationError(
                        f"physics observable {name} tolerance is not canonical"
                    )
            if reference_observable["monitor"] != policy["monitor"]["name"]:
                raise ValidationError(
                    f"physics reference observable {name} monitor is not canonical"
                )
            if reference_observable["unit"] != policy["unit"]:
                raise ValidationError(
                    f"physics reference observable {name} unit is not canonical"
                )
            if observable["reference_value"] != reference_observable["value"]:
                raise ValidationError(
                    f"physics observable {name} reference value is not authenticated"
                )
        expected_repetitions = manifest_details["execution"]["measured_repetitions"]
        if len(repetitions) != expected_repetitions:
            raise ValidationError(
                "successful result repetition count does not match the run manifest"
            )
        if any(float(value) <= 0 for value in repetitions):
            raise ValidationError("successful result repetitions must be positive")
        if median_seconds is None or median_seconds <= 0:
            raise ValidationError("successful result requires positive timing samples")
        if warmup_steps != manifest_details["execution"]["warmup_steps"]:
            raise ValidationError("result warmup_steps does not match the run manifest")
        if simulation_steps <= 0:
            raise ValidationError(
                "successful result must execute at least one timestep"
            )
        if not math.isclose(
            physical_time_meep,
            simulation_steps * dt_meep,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValidationError("physical_time_meep must equal steps * dt_meep")
        stopping = manifest_details["stopping"]
        if stopping["kind"] == "fixed_steps" and simulation_steps != stopping["steps"]:
            raise ValidationError("result steps do not match fixed-step run manifest")
        if (
            stopping["kind"] == "field_energy_decay"
            and simulation_steps > stopping["max_steps"]
        ):
            raise ValidationError("result steps exceed end-to-end safety cap")
        if not observables:
            raise ValidationError("successful result requires physics observables")
        if not all(observable.get("passed") is True for observable in observables):
            raise ValidationError(
                "successful result contains a failed physics observable"
            )
        if component_updates_per_grid_timestep <= 0:
            raise ValidationError(
                "successful result requires component updates per grid timestep"
            )
        expected_grid_rate = points * simulation_steps / median_seconds
        expected_component_rate = (
            expected_grid_rate * component_updates_per_grid_timestep
        )
        for label, actual, expected in (
            (
                "grid_timesteps_per_second",
                grid_timesteps_per_second,
                expected_grid_rate,
            ),
            (
                "component_updates_per_second",
                component_updates_per_second,
                expected_component_rate,
            ),
        ):
            if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=0.0):
                raise ValidationError(
                    f"performance.{label} does not match grid, steps, and median time"
                )


def default_lambda_min(center_um: float, bandwidth_nm: float) -> float:
    return (
        _finite(center_um, "center_um", positive=True)
        - 0.5 * _finite(bandwidth_nm, "bandwidth_nm", positive=True) / 1000.0
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=pathlib.Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--cases", type=pathlib.Path, default=DEFAULT_CASES)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate all external inputs")
    validate.add_argument("--fdtd-pipeline", required=True, type=pathlib.Path)

    manifest = subparsers.add_parser("manifest", help="emit a run manifest")
    manifest.add_argument("--fdtd-pipeline", required=True, type=pathlib.Path)
    manifest.add_argument("--device", required=True)
    manifest.add_argument("--mode", choices=sorted(MODES), required=True)
    manifest.add_argument(
        "--cells-per-material-wavelength",
        type=float,
        choices=(6, 10, 15, 20, 25),
        default=15.0,
    )
    manifest.add_argument("--n-max", type=float, required=True)
    manifest.add_argument("--lambda-min-um", type=float)
    manifest.add_argument(
        "--resolution-rounding",
        choices=("ceil", "nearest", "floor", "none"),
        default="ceil",
    )
    manifest.add_argument(
        "--bandwidth-nm", type=float, choices=(20.0, 50.0), default=20.0
    )
    manifest.add_argument(
        "--material-mode",
        choices=("paper", "performance-adaptation"),
        default="performance-adaptation",
    )
    manifest.add_argument("--material-validation", type=pathlib.Path)
    manifest.add_argument("--precision", choices=sorted(PRECISIONS), default="native")
    manifest.add_argument("--backend", choices=sorted(BACKENDS), default="nvidia")
    manifest.add_argument("--ranks", type=int, default=1)
    manifest.add_argument("--mpi-transport", choices=sorted(TRANSPORTS), default="none")
    manifest.add_argument("--overlap", choices=sorted(OVERLAPS), default="off")
    manifest.add_argument("--graph", choices=sorted(GRAPHS), default="eager")
    manifest.add_argument("--steps", type=int)
    manifest.add_argument("--max-steps", type=int)
    manifest.add_argument("--auto-shutoff-threshold", type=float, default=1e-5)
    manifest.add_argument("--decay-check-interval-meep", type=float, default=50.0)
    manifest.add_argument("--output", type=pathlib.Path, default=pathlib.Path("-"))

    template = subparsers.add_parser(
        "result-template", help="create a typed result/provenance template"
    )
    template.add_argument("--manifest", required=True, type=pathlib.Path)
    template.add_argument("--output", type=pathlib.Path, default=pathlib.Path("-"))

    validate_result = subparsers.add_parser(
        "validate-result", help="validate a populated result document"
    )
    validate_result.add_argument("--result", required=True, type=pathlib.Path)
    return parser


def _write_json(value: Mapping[str, Any], output: pathlib.Path) -> None:
    try:
        rendered = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except ValueError as error:
        raise ValidationError(f"document contains NaN or Infinity: {error}") from error
    if str(output) == "-":
        sys.stdout.write(rendered)
    else:
        output.write_text(rendered, encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "result-template":
            result = build_result_template(args.manifest)
            validate_result_document(result)
            _write_json(result, args.output)
            return 0
        if args.command == "validate-result":
            result = load_json_object(args.result, "benchmark result")
            validate_result_document(result, args.result)
            print(json.dumps({"valid": True, "result": str(args.result.resolve())}))
            return 0

        reference = load_reference(args.reference)
        case_definitions = load_case_definitions(args.cases)
        selected = None if args.command == "validate" else [args.device]
        checkout = validate_checkout(
            args.fdtd_pipeline, reference, case_definitions, selected
        )
        if args.command == "validate":
            _write_json(checkout, pathlib.Path("-"))
            return 0
        lambda_min_um = args.lambda_min_um
        if lambda_min_um is None:
            lambda_min_um = default_lambda_min(
                reference["common"]["center_wavelength_um"], args.bandwidth_nm
            )
        manifest = build_manifest(
            reference=reference,
            reference_path=args.reference,
            case_definitions=case_definitions,
            cases_path=args.cases,
            result_schema_path=(
                DEFAULT_MPI_RESULT_SCHEMA if args.ranks > 1 else DEFAULT_RESULT_SCHEMA
            ),
            checkout_validation=checkout,
            device_name=args.device,
            mode=args.mode,
            cells_per_material_wavelength=args.cells_per_material_wavelength,
            n_max=args.n_max,
            lambda_min_um=lambda_min_um,
            rounding=args.resolution_rounding,
            bandwidth_nm=args.bandwidth_nm,
            material_mode=args.material_mode,
            material_validation=args.material_validation,
            precision=args.precision,
            backend=args.backend,
            ranks=args.ranks,
            mpi_transport=args.mpi_transport,
            overlap=args.overlap,
            graph=args.graph,
            steps=args.steps,
            max_steps=args.max_steps,
            auto_shutoff_threshold=args.auto_shutoff_threshold,
            decay_check_interval_meep=args.decay_check_interval_meep,
        )
        _write_json(manifest, args.output)
        return 0
    except ValidationError as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
