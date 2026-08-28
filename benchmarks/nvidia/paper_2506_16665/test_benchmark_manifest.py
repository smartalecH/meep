import json
import math
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import benchmark_manifest as bm


def synthetic_checkout_validation(reference, cases):
    return {
        "checkout": "/external/fdtd-pipeline",
        "expected_commit": reference["paper"]["code_commit"],
        "actual_commit": reference["paper"]["code_commit"],
        "dirty": False,
        "status_porcelain": [],
        "validated_cases": list(reference["devices"]),
        "assets": {
            "gds": {
                name: {
                    "relative_path": device["gds_relative_path"],
                    "path": str(
                        pathlib.Path("/external/fdtd-pipeline")
                        / device["gds_relative_path"]
                    ),
                    "sha256": device["gds_sha256"],
                    "bytes": 1,
                }
                for name, device in reference["devices"].items()
            },
            "yaml": {
                name: {
                    "relative_path": case["yaml_relative_path"],
                    "path": str(
                        pathlib.Path("/external/fdtd-pipeline")
                        / case["yaml_relative_path"]
                    ),
                    "sha256": case["yaml_sha256"],
                    "bytes": 1,
                }
                for name, case in cases["cases"].items()
            },
            "layer_stack": {
                "relative_path": "stack_universal.json",
                "path": "/external/fdtd-pipeline/stack_universal.json",
                "sha256": cases["assets"]["layer_stack"]["sha256"],
                "bytes": 1,
            },
        },
    }


def build_test_manifest(reference, cases, **updates):
    args = dict(
        reference=reference,
        reference_path=bm.DEFAULT_REFERENCE,
        case_definitions=cases,
        cases_path=bm.DEFAULT_CASES,
        result_schema_path=bm.DEFAULT_RESULT_SCHEMA,
        checkout_validation=synthetic_checkout_validation(reference, cases),
        device_name="coupler",
        mode="smoke",
        cells_per_material_wavelength=15,
        n_max=3.48,
        lambda_min_um=1.54,
        rounding="ceil",
        bandwidth_nm=20.0,
        material_mode="performance-adaptation",
        material_validation=None,
        precision="native",
        backend="cpu",
        ranks=1,
        mpi_transport="none",
        steps=None,
        max_steps=None,
        auto_shutoff_threshold=1e-5,
        decay_check_interval_meep=50.0,
    )
    args.update(updates)
    return bm.build_manifest(**args)


class ReferenceAndCaseTests(unittest.TestCase):
    def test_bundled_inputs_cover_same_six_cases(self):
        reference = bm.load_reference()
        cases = bm.load_case_definitions()
        expected = {
            "coupler",
            "crossing",
            "mmi2x2",
            "mode_converter",
            "psr",
            "ring",
        }
        self.assertEqual(set(reference["devices"]), expected)
        self.assertEqual(set(cases["cases"]), expected)
        self.assertEqual(cases["cases"]["crossing"]["active_layers"], ["Si", "SLAB"])
        self.assertEqual(cases["cases"]["psr"]["source"]["polarization"], "TM")
        self.assertEqual(len(cases["cases"]["ring"]["monitors"]), 3)
        self.assertEqual(
            {
                observable["name"]
                for observable in cases["cases"]["ring"]["required_observables"]
            },
            {"resonance_wavelength_um", "fwhm_um", "quality_factor"},
        )

    def test_structured_cases_have_resolvable_ports(self):
        cases = bm.load_case_definitions()
        for case in cases["cases"].values():
            self.assertIn(case["source"]["port"], case["ports"])
            incident = [
                monitor
                for monitor in case["monitors"]
                if monitor.get("observable") == "incident_power"
            ]
            self.assertEqual(len(incident), 1)
            self.assertEqual(incident[0]["port"], case["source"]["port"])
            for monitor in case["monitors"]:
                if "port" in monitor:
                    self.assertIn(monitor["port"], case["ports"])
                if monitor.get("observable") in bm.POWER_OBSERVABLES:
                    self.assertEqual(
                        monitor["normalization"]["denominator_monitor"],
                        incident[0]["name"],
                    )

    def test_reference_rejects_missing_device_data(self):
        reference = bm.load_reference()
        del reference["devices"]["ring"]["domain_um"]
        with self.assertRaisesRegex(bm.ValidationError, "domain_um"):
            bm.validate_reference(reference)

    def test_case_validation_rejects_unknown_port_and_bad_layer(self):
        cases = bm.load_case_definitions()
        cases["cases"]["coupler"]["source"]["port"] = "missing"
        with self.assertRaisesRegex(bm.ValidationError, "unknown port"):
            bm.validate_case_definitions(cases)
        cases = bm.load_case_definitions()
        cases["cases"]["coupler"]["active_layers"] = ["UNKNOWN"]
        with self.assertRaisesRegex(bm.ValidationError, "active_layers"):
            bm.validate_case_definitions(cases)

    def test_case_validation_requires_power_normalization_and_spectral_policy(self):
        cases = bm.load_case_definitions()
        del cases["cases"]["coupler"]["monitors"][1]["normalization"]
        with self.assertRaisesRegex(bm.ValidationError, "normalization"):
            bm.validate_case_definitions(cases)
        cases = bm.load_case_definitions()
        cases["source_defaults"]["spectral_envelope"][
            "frequency_conversion"
        ] = "center_only"
        with self.assertRaisesRegex(bm.ValidationError, "spectral_envelope"):
            bm.validate_case_definitions(cases)

    def test_malformed_json_is_a_validation_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "bad.json"
            path.write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(bm.ValidationError, "cannot read"):
                bm.load_reference(path)
            path.write_text('{"value": NaN}', encoding="utf-8")
            with self.assertRaisesRegex(bm.ValidationError, "non-finite"):
                bm.load_json_object(path, "fixture")


class ResolutionTests(unittest.TestCase):
    def test_ceil_conversion_does_not_undersample(self):
        result = bm.convert_resolution(15, 3.48, 1.54, "ceil")
        self.assertAlmostEqual(result["exact_resolution_px_per_um"], 33.8961038961)
        self.assertEqual(result["resolution_px_per_um"], 34.0)

    def test_none_conversion_preserves_exact_value(self):
        result = bm.convert_resolution(20, 3.5, 1.54, "none")
        self.assertAlmostEqual(result["resolution_px_per_um"], 20 * 3.5 / 1.54)

    def test_nonfinite_or_nonpositive_inputs_are_rejected(self):
        for value in (0, -1, math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(bm.ValidationError):
                bm.convert_resolution(15, value, 1.54)


class CheckoutTests(unittest.TestCase):
    def setUp(self):
        self.reference = {
            "paper": {"code_commit": "a" * 40},
            "devices": {
                "case": {
                    "gds_relative_path": "inputs/case.gds",
                    "gds_sha256": "0" * 64,
                },
                "unselected": {
                    "gds_relative_path": "inputs/unselected.gds",
                    "gds_sha256": "1" * 64,
                },
            },
        }
        self.cases = {
            "assets": {
                "layer_stack": {
                    "relative_path": "stack.json",
                    "sha256": "2" * 64,
                }
            },
            "cases": {
                "case": {
                    "yaml_relative_path": "inputs/case.yml",
                    "yaml_sha256": "3" * 64,
                },
                "unselected": {
                    "yaml_relative_path": "inputs/unselected.yml",
                    "yaml_sha256": "4" * 64,
                },
            },
        }

    def make_checkout(self, root):
        inputs = root / "inputs"
        inputs.mkdir()
        (inputs / "case.gds").write_bytes(b"gds")
        (inputs / "case.yml").write_bytes(b"yaml")
        (root / "stack.json").write_bytes(b"stack")
        self.reference["devices"]["case"]["gds_sha256"] = bm.sha256_file(
            inputs / "case.gds"
        )
        self.cases["cases"]["case"]["yaml_sha256"] = bm.sha256_file(inputs / "case.yml")
        self.cases["assets"]["layer_stack"]["sha256"] = bm.sha256_file(
            root / "stack.json"
        )

    def git_side_effect(self, _checkout, *args):
        if args == ("rev-parse", "HEAD"):
            return "a" * 40
        if args == ("status", "--porcelain=v1"):
            return " M inputs/case.yml"
        raise AssertionError(args)

    @mock.patch.object(bm, "git_command")
    def test_selected_case_assets_and_dirty_status(self, git_command):
        git_command.side_effect = self.git_side_effect
        with tempfile.TemporaryDirectory() as tmp:
            checkout = pathlib.Path(tmp)
            self.make_checkout(checkout)
            result = bm.validate_checkout(
                checkout, self.reference, self.cases, ["case"]
            )
        self.assertTrue(result["dirty"])
        self.assertEqual(result["validated_cases"], ["case"])
        self.assertIn("case", result["assets"]["yaml"])
        self.assertNotIn("unselected", result["assets"]["gds"])

    @mock.patch.object(bm, "git_command")
    def test_modified_yaml_is_rejected(self, git_command):
        git_command.side_effect = self.git_side_effect
        with tempfile.TemporaryDirectory() as tmp:
            checkout = pathlib.Path(tmp)
            self.make_checkout(checkout)
            (checkout / "inputs" / "case.yml").write_bytes(b"modified")
            with self.assertRaisesRegex(bm.ValidationError, "YAML.*hash mismatch"):
                bm.validate_checkout(checkout, self.reference, self.cases, ["case"])

    @mock.patch.object(bm, "git_command", return_value="b" * 40)
    def test_wrong_commit_is_rejected(self, _git_command):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = pathlib.Path(tmp)
            self.make_checkout(checkout)
            with self.assertRaisesRegex(bm.ValidationError, "pinned commit"):
                bm.validate_checkout(checkout, self.reference, self.cases, ["case"])


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.reference = bm.load_reference()
        self.cases = bm.load_case_definitions()
        self.validation = {
            "checkout": "/external/fdtd-pipeline",
            "expected_commit": self.reference["paper"]["code_commit"],
            "actual_commit": self.reference["paper"]["code_commit"],
            "dirty": False,
            "status_porcelain": [],
            "validated_cases": list(self.reference["devices"]),
            "assets": {
                "gds": {
                    name: {
                        "relative_path": device["gds_relative_path"],
                        "path": f"/external/{name}.gds",
                        "sha256": device["gds_sha256"],
                        "bytes": 1,
                    }
                    for name, device in self.reference["devices"].items()
                },
                "yaml": {
                    name: {
                        "relative_path": case["yaml_relative_path"],
                        "path": f"/external/{name}.yml",
                        "sha256": case["yaml_sha256"],
                        "bytes": 1,
                    }
                    for name, case in self.cases["cases"].items()
                },
                "layer_stack": {
                    "relative_path": "stack_universal.json",
                    "path": "/external/stack_universal.json",
                    "sha256": self.cases["assets"]["layer_stack"]["sha256"],
                    "bytes": 1,
                },
            },
        }

    def manifest(self, mode="smoke", **updates):
        args = dict(
            reference=self.reference,
            reference_path=bm.DEFAULT_REFERENCE,
            case_definitions=self.cases,
            cases_path=bm.DEFAULT_CASES,
            result_schema_path=bm.DEFAULT_RESULT_SCHEMA,
            checkout_validation=self.validation,
            device_name="coupler",
            mode=mode,
            cells_per_material_wavelength=15,
            n_max=3.48,
            lambda_min_um=1.54,
            rounding="ceil",
            bandwidth_nm=20.0,
            material_mode="performance-adaptation",
            material_validation=None,
            precision="native",
            backend="nvidia",
            ranks=1,
            mpi_transport="none",
            steps=None,
            max_steps=None,
            auto_shutoff_threshold=1e-5,
            decay_check_interval_meep=50.0,
        )
        args.update(updates)
        return bm.build_manifest(**args)

    def test_all_cases_emit_structured_runner_inputs(self):
        for name in self.reference["devices"]:
            with self.subTest(name=name):
                manifest = self.manifest(device_name=name)
                json.dumps(manifest, allow_nan=False)
                self.assertTrue(manifest["case"]["ports"])
                self.assertIn("source", manifest["case"])
                self.assertTrue(manifest["case"]["monitors"])
                envelope = manifest["case"]["source"]["spectral_envelope"]
                self.assertGreater(envelope["fwidth_meep"], 0)
                self.assertEqual(
                    envelope["bandwidth_definition"],
                    "full_centered_wavelength_interval",
                )
                self.assertIn("yaml", manifest["input_checkout"])
                self.assertEqual(
                    manifest["execution"]["requested"]["precision"], "native"
                )
                self.assertTrue(manifest["validation_policy"]["required_observables"])
                self.assertEqual(
                    manifest["manifest_schema"]["sha256"],
                    bm.sha256_file(bm.DEFAULT_MANIFEST_SCHEMA),
                )
                if name == "ring":
                    derived = next(
                        monitor
                        for monitor in manifest["case"]["monitors"]
                        if monitor["name"] == "ring_resonance"
                    )
                    self.assertNotIn("frequency_sampling", derived)

    def test_rounding_and_exact_workload_counts_are_distinct(self):
        workload = self.manifest()["paper_comparison"]["workload_25"]["tidy3d"]
        self.assertNotEqual(
            workload["reported_grid_points_rounded"], workload["grid_points_from_shape"]
        )
        self.assertEqual(workload["reported_grid_points_denominator"], 1_000_000)

    def test_mode_argument_conflicts(self):
        with self.assertRaisesRegex(bm.ValidationError, "max-steps"):
            self.manifest(max_steps=5)
        with self.assertRaisesRegex(bm.ValidationError, "steps"):
            self.manifest(mode="end-to-end", steps=5, max_steps=10)
        with self.assertRaisesRegex(bm.ValidationError, "max-steps"):
            self.manifest(mode="end-to-end")
        with self.assertRaisesRegex(bm.ValidationError, "only configurable"):
            self.manifest(decay_check_interval_meep=25.0)
        self.assertEqual(
            self.manifest(mode="fixed-step", steps=20)["stopping"]["steps"], 20
        )

    def test_backend_precision_rank_transport_matrix(self):
        valid = (
            ("cpu", "native", 1, "none"),
            ("cpu", "native", 3, "host"),
            ("nvidia", "native", 1, "none"),
            ("nvidia", "f32", 2, "staged"),
            ("nvidia", "mixed", 4, "direct"),
            ("auto", "native", 2, "auto"),
        )
        for values in valid:
            with self.subTest(values=values):
                self.manifest(
                    backend=values[0],
                    precision=values[1],
                    ranks=values[2],
                    mpi_transport=values[3],
                )
        invalid = (
            ("cpu", "f32", 1, "none"),
            ("cpu", "mixed", 1, "none"),
            ("cpu", "native", 2, "none"),
            ("nvidia", "f32", 2, "host"),
            ("auto", "native", 2, "staged"),
            ("nvidia", "native", 0, "none"),
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(bm.ValidationError):
                self.manifest(
                    backend=values[0],
                    precision=values[1],
                    ranks=values[2],
                    mpi_transport=values[3],
                )

    def test_unknown_enums_and_nonfinite_values_are_rejected_by_api(self):
        for field, value in (
            ("backend", "cuda"),
            ("precision", "double"),
            ("mpi_transport", "magic"),
            ("mode", "profile"),
        ):
            with self.subTest(field=field), self.assertRaises(bm.ValidationError):
                self.manifest(**{field: value})
        for field in (
            "n_max",
            "lambda_min_um",
            "bandwidth_nm",
            "decay_check_interval_meep",
        ):
            kwargs = {field: math.inf}
            if field == "decay_check_interval_meep":
                kwargs.update(mode="end-to-end", max_steps=10)
            with self.subTest(field=field), self.assertRaises(bm.ValidationError):
                self.manifest(**kwargs)

    def material_proof(self, root):
        reference = bm.load_reference()
        proof = {
            "schema_version": 1,
            "paper_reference_sha256": bm.sha256_file(bm.DEFAULT_REFERENCE),
            "pole_convention": reference["material_pole_convention"],
            "wavelength_range_um": [1.52, 1.58],
            "sample_count": 101,
            "error": {
                "norm": "max relative complex epsilon",
                "tolerance": 1e-6,
                "measured": 1e-8,
                "passed": True,
            },
            "materials": {
                "Si": {"max_absolute_error": 1e-8, "max_relative_error": 1e-9},
                "SiO2": {"max_absolute_error": 1e-8, "max_relative_error": 1e-9},
            },
        }
        path = root / "material-validation.json"
        path.write_text(json.dumps(proof), encoding="utf-8")
        return path

    def test_paper_material_mode_requires_passing_proof(self):
        with self.assertRaisesRegex(bm.ValidationError, "requires"):
            self.manifest(material_mode="paper")
        with tempfile.TemporaryDirectory() as tmp:
            proof = self.material_proof(pathlib.Path(tmp))
            manifest = self.manifest(material_mode="paper", material_validation=proof)
            self.assertEqual(manifest["materials"]["validation"]["path"], str(proof))
            data = json.loads(proof.read_text(encoding="utf-8"))
            data["error"]["measured"] = 1.0
            proof.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(bm.ValidationError, "does not prove"):
                self.manifest(material_mode="paper", material_validation=proof)

    def test_paper_material_proof_must_cover_band_and_match_convention(self):
        with tempfile.TemporaryDirectory() as tmp:
            proof = self.material_proof(pathlib.Path(tmp))
            data = json.loads(proof.read_text(encoding="utf-8"))
            data["wavelength_range_um"] = [1.54, 1.56]
            proof.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(bm.ValidationError, "cover"):
                self.manifest(
                    material_mode="paper",
                    material_validation=proof,
                    bandwidth_nm=50.0,
                    lambda_min_um=1.525,
                )

            data = json.loads(proof.read_text(encoding="utf-8"))
            data["wavelength_range_um"] = [1.52, 1.58]
            data["pole_convention"]["frequency_unit"] = "hertz"
            proof.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(bm.ValidationError, "convention"):
                self.manifest(material_mode="paper", material_validation=proof)

            data = json.loads(proof.read_text(encoding="utf-8"))
            data["pole_convention"] = bm.load_reference()["material_pole_convention"]
            data["materials"]["Si"]["max_relative_error"] = 1e-3
            proof.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(bm.ValidationError, "Si.*tolerance"):
                self.manifest(material_mode="paper", material_validation=proof)


class ResultTests(unittest.TestCase):
    def template(
        self,
        *,
        backend="cpu",
        precision="native",
        ranks=1,
        transport="none",
        repetitions=1,
        device="coupler",
    ):
        reference = bm.load_reference()
        cases = bm.load_case_definitions()
        manifest = build_test_manifest(
            reference,
            cases,
            backend=backend,
            precision=precision,
            ranks=ranks,
            mpi_transport=transport,
            device_name=device,
        )
        manifest["execution"]["measured_repetitions"] = repetitions
        temporary = tempfile.TemporaryDirectory()
        root = pathlib.Path(temporary.name)
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        result = bm.build_result_template(path)
        physics_reference = {
            "schema_version": 1,
            "kind": "cpu_native_baseline",
            "paper_reference_sha256": bm.sha256_file(bm.DEFAULT_REFERENCE),
            "case_definitions_sha256": bm.sha256_file(bm.DEFAULT_CASES),
            "case_id": device,
            "physics_configuration_sha256": manifest["validation_policy"]["reference"][
                "physics_configuration_sha256"
            ],
            "observables": [
                {
                    "name": policy["name"],
                    "monitor": policy["monitor"]["name"],
                    "unit": policy["unit"],
                    "value": 0.91 + index,
                }
                for index, policy in enumerate(
                    manifest["validation_policy"]["required_observables"]
                )
            ],
        }
        reference_path = root / "physics-reference.json"
        reference_path.write_text(json.dumps(physics_reference), encoding="utf-8")
        result["physics_reference"] = {
            "path": str(reference_path),
            "sha256": bm.sha256_file(reference_path),
        }
        return temporary, result, path

    def make_successful(self, result, repetitions=1):
        result["status"] = {"exit_code": 0, "succeeded": True, "errors": []}
        result["simulation"] = {
            "grid_shape": [2, 3, 4],
            "grid_points_exact": 24,
            "courant_factor": 0.5,
            "dt_meep": 0.1,
            "steps": 10,
            "physical_time_meep": 1.0,
        }
        result["timing"]["steady_state_repetitions_seconds"] = [2.0] * repetitions
        result["timing"]["steady_state_min_seconds"] = 2.0
        result["timing"]["steady_state_median_seconds"] = 2.0
        result["timing"]["steady_state_max_seconds"] = 2.0
        result["performance"] = {
            "grid_timesteps_per_second": 120.0,
            "component_updates_per_second": 720.0,
            "component_updates_per_grid_timestep": 6.0,
            "rate_basis": "steady_state_median_seconds",
        }
        result["provenance"]["meep"]["commit"] = "1" * 40
        result["provenance"]["invocation"] = {
            "argv": ["meep-benchmark", "--device", "coupler"],
            "cwd": "/tmp/benchmark",
            "environment": {},
        }
        manifest = json.loads(
            pathlib.Path(result["run_manifest"]["path"]).read_text(encoding="utf-8")
        )
        reference = json.loads(
            pathlib.Path(result["physics_reference"]["path"]).read_text(
                encoding="utf-8"
            )
        )
        references = {
            observable["name"]: observable for observable in reference["observables"]
        }
        result["physics_observables"] = [
            {
                "name": policy["name"],
                "value": references[policy["name"]]["value"],
                "unit": policy["unit"],
                "reference_value": references[policy["name"]]["value"],
                "absolute_tolerance": policy["absolute_tolerance"],
                "relative_tolerance": policy["relative_tolerance"],
                "passed": True,
                "monitor": policy["monitor"],
            }
            for policy in manifest["validation_policy"]["required_observables"]
        ]

    def test_template_is_valid_and_schema_is_parseable(self):
        temporary, result, _manifest = self.template()
        self.addCleanup(temporary.cleanup)
        bm.validate_result_document(result)
        schema = bm.load_json_object(bm.DEFAULT_RESULT_SCHEMA, "result schema")
        self.assertEqual(schema["properties"]["schema_version"]["const"], 1)
        self.assertIn(
            "requested_execution", schema["properties"]["provenance"]["required"]
        )
        manifest_schema = bm.load_json_object(
            bm.DEFAULT_MANIFEST_SCHEMA, "run manifest schema"
        )
        self.assertEqual(manifest_schema["properties"]["schema_version"]["const"], 2)

    def test_result_rejects_nonfinite_and_grid_mismatch(self):
        temporary, result, _manifest = self.template()
        self.addCleanup(temporary.cleanup)
        result["timing"]["kernel_seconds"] = math.nan
        with self.assertRaisesRegex(bm.ValidationError, "finite"):
            bm.validate_result_document(result)
        result["timing"]["kernel_seconds"] = 0.0
        result["simulation"]["grid_points_exact"] = 2
        with self.assertRaisesRegex(bm.ValidationError, "product"):
            bm.validate_result_document(result)

    def test_observable_requires_typed_monitor_and_tolerances(self):
        temporary, result, _manifest = self.template()
        self.addCleanup(temporary.cleanup)
        result["physics_observables"] = [
            {
                "name": "transmission",
                "value": 0.9,
                "unit": "1",
                "reference_value": 0.91,
                "absolute_tolerance": 0.02,
                "relative_tolerance": 0.03,
                "passed": True,
                "monitor": {"name": "cross_te0", "port": "o3"},
            }
        ]
        bm.validate_result_document(result)
        result["physics_observables"][0]["relative_tolerance"] = math.inf
        with self.assertRaisesRegex(bm.ValidationError, "finite"):
            bm.validate_result_document(result)

    def test_success_requires_samples_observables_and_normalized_rates(self):
        temporary, result, _manifest = self.template()
        self.addCleanup(temporary.cleanup)
        self.make_successful(result)
        bm.validate_result_document(result)

        result["timing"]["steady_state_repetitions_seconds"] = []
        with self.assertRaisesRegex(bm.ValidationError, "repetition count"):
            bm.validate_result_document(result)
        self.make_successful(result)
        result["physics_observables"] = []
        with self.assertRaisesRegex(bm.ValidationError, "physics observable"):
            bm.validate_result_document(result)
        self.make_successful(result)
        result["performance"]["grid_timesteps_per_second"] = 1.0
        with self.assertRaisesRegex(bm.ValidationError, "grid_timesteps"):
            bm.validate_result_document(result)
        self.make_successful(result)
        result["simulation"]["physical_time_meep"] = 2.0
        with self.assertRaisesRegex(bm.ValidationError, "steps \* dt"):
            bm.validate_result_document(result)

    def test_result_authenticates_manifest_and_enforces_schema_shape(self):
        temporary, result, manifest = self.template()
        self.addCleanup(temporary.cleanup)
        result["unexpected"] = 1
        with self.assertRaisesRegex(bm.ValidationError, "unexpected property"):
            bm.validate_result_document(result)
        del result["unexpected"]
        result["run_manifest"]["case_id"] = "ring"
        with self.assertRaisesRegex(bm.ValidationError, "case_id"):
            bm.validate_result_document(result)
        result["run_manifest"]["case_id"] = "coupler"
        original_path = result["run_manifest"]["path"]
        result["run_manifest"]["path"] = str(manifest.parent / "missing.json")
        with self.assertRaisesRegex(bm.ValidationError, "cannot read run manifest"):
            bm.validate_result_document(result)
        result["run_manifest"]["path"] = original_path
        manifest.write_text(
            manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(bm.ValidationError, "hash mismatch"):
            bm.validate_result_document(result)

    def test_authenticated_manifest_rejects_corpus_and_asset_tampering(self):
        temporary, result, manifest_path = self.template()
        self.addCleanup(temporary.cleanup)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["reference"]["sha256"] = "f" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        result["run_manifest"]["sha256"] = bm.sha256_file(manifest_path)
        with self.assertRaisesRegex(bm.ValidationError, "bundled corpus"):
            bm.validate_result_document(result)

        manifest = build_test_manifest(bm.load_reference(), bm.load_case_definitions())
        manifest["input_checkout"]["yaml"]["sha256"] = "e" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        result["run_manifest"]["sha256"] = bm.sha256_file(manifest_path)
        with self.assertRaisesRegex(bm.ValidationError, "YAML|yaml"):
            bm.validate_result_document(result)

        manifest = build_test_manifest(bm.load_reference(), bm.load_case_definitions())
        manifest["unexpected"] = True
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        result["run_manifest"]["sha256"] = bm.sha256_file(manifest_path)
        with self.assertRaisesRegex(bm.ValidationError, "unexpected property"):
            bm.validate_result_document(result)

        manifest = build_test_manifest(bm.load_reference(), bm.load_case_definitions())
        manifest["case"]["geometry_import"]["gds_scale"] = True
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        result["run_manifest"]["sha256"] = bm.sha256_file(manifest_path)
        with self.assertRaisesRegex(bm.ValidationError, "geometry_import"):
            bm.validate_result_document(result)

    def test_result_rejects_requested_config_mismatch(self):
        temporary, result, _manifest = self.template()
        self.addCleanup(temporary.cleanup)
        result["provenance"]["requested_execution"]["backend"] = "auto"
        with self.assertRaisesRegex(bm.ValidationError, "requested_execution"):
            bm.validate_result_document(result)

    def test_successful_direct_result_requires_unique_devices_and_provider_gate(self):
        temporary, result, _manifest = self.template(
            backend="nvidia", ranks=2, transport="direct"
        )
        self.addCleanup(temporary.cleanup)
        self.make_successful(result)
        result["provenance"]["cuda"] = {
            "driver_version": "13000",
            "runtime_version": "12080",
            "toolkit_version": "12.8",
        }
        result["provenance"]["mpi"] = {
            "provider": "Open MPI",
            "version": "5.0",
            "cuda_aware_query": True,
            "mca": {"opal_cuda_support": "true"},
        }
        device = {
            "visible_device": 0,
            "uuid": "gpu-0",
            "name": "GB200",
            "memory_bytes": 1024,
            "sm_clock_hz": 1.0,
            "memory_clock_hz": 1.0,
        }
        result["provenance"]["rank_devices"] = [
            {**device, "rank": 0},
            {**device, "rank": 1, "visible_device": 1, "uuid": "gpu-1"},
        ]
        bm.validate_result_document(result)
        result["provenance"]["cuda"]["driver_version"] = ""
        with self.assertRaisesRegex(bm.ValidationError, "CUDA version provenance"):
            bm.validate_result_document(result)
        result["provenance"]["cuda"]["driver_version"] = "13000"
        result["provenance"]["rank_devices"][1]["uuid"] = "gpu-0"
        with self.assertRaisesRegex(bm.ValidationError, "duplicate GPU UUID"):
            bm.validate_result_document(result)
        result["provenance"]["rank_devices"][1]["uuid"] = "gpu-1"
        result["provenance"]["mpi"]["cuda_aware_query"] = False
        with self.assertRaisesRegex(bm.ValidationError, "positive CUDA-aware"):
            bm.validate_result_document(result)

    def test_success_rejects_placeholder_provenance(self):
        temporary, result, _manifest = self.template()
        self.addCleanup(temporary.cleanup)
        self.make_successful(result)
        result["provenance"]["meep"]["commit"] = "0" * 40
        with self.assertRaisesRegex(bm.ValidationError, "real Meep commit"):
            bm.validate_result_document(result)
        self.make_successful(result)
        result["provenance"]["invocation"]["argv"] = []
        with self.assertRaisesRegex(bm.ValidationError, "nonempty invocation argv"):
            bm.validate_result_document(result)
        self.make_successful(result)
        result["provenance"]["invocation"]["cwd"] = ""
        with self.assertRaisesRegex(bm.ValidationError, "nonempty invocation cwd"):
            bm.validate_result_document(result)

    def test_success_rejects_self_certified_observables_and_requires_ring_set(self):
        temporary, result, _manifest = self.template()
        self.addCleanup(temporary.cleanup)
        self.make_successful(result)
        result["physics_observables"][0]["reference_value"] = 0.0
        result["physics_observables"][0]["value"] = 0.0
        with self.assertRaisesRegex(bm.ValidationError, "not authenticated"):
            bm.validate_result_document(result)
        self.make_successful(result)
        result["physics_observables"][0]["monitor"] = {"name": "input_incident"}
        with self.assertRaisesRegex(bm.ValidationError, "monitor is not canonical"):
            bm.validate_result_document(result)

        ring_temporary, ring_result, _ring_manifest = self.template(device="ring")
        self.addCleanup(ring_temporary.cleanup)
        self.make_successful(ring_result)
        self.assertEqual(
            {observable["name"] for observable in ring_result["physics_observables"]},
            {"resonance_wavelength_um", "fwhm_um", "quality_factor"},
        )
        ring_result["physics_observables"].pop()
        with self.assertRaisesRegex(bm.ValidationError, "observable set"):
            bm.validate_result_document(ring_result)

    def test_physics_reference_is_bound_to_exact_physics_configuration(self):
        temporary, result, manifest_path = self.template()
        self.addCleanup(temporary.cleanup)
        original_reference = dict(result["physics_reference"])

        different_manifest = build_test_manifest(
            bm.load_reference(),
            bm.load_case_definitions(),
            cells_per_material_wavelength=25,
        )
        manifest_path.write_text(json.dumps(different_manifest), encoding="utf-8")
        result["run_manifest"]["sha256"] = bm.sha256_file(manifest_path)
        result["physics_reference"] = original_reference
        self.make_successful(result)
        with self.assertRaisesRegex(bm.ValidationError, "physics_configuration_sha256"):
            bm.validate_result_document(result)

    def test_physics_configuration_hash_covers_physical_inputs(self):
        manifest = build_test_manifest(bm.load_reference(), bm.load_case_definitions())
        baseline = bm.physics_configuration_sha256(manifest)
        mutations = (
            lambda value: value["discretization"].update(n_max=3.49),
            lambda value: value["discretization"].update(lambda_min_um=1.53),
            lambda value: value["excitation"].update(bandwidth_nm=50.0),
            lambda value: value["materials"].update(validation={"sha256": "1" * 64}),
            lambda value: value["case"]["boundaries"].update(thickness_um=2.0),
            lambda value: value["case"]["monitors"][1].update(mode_order=1),
            lambda value: value["stopping"].update(steps=11),
        )
        for mutate in mutations:
            changed = json.loads(json.dumps(manifest))
            mutate(changed)
            with self.subTest(mutation=mutate):
                self.assertNotEqual(bm.physics_configuration_sha256(changed), baseline)

    def test_schema_scalar_constraints_are_enforced(self):
        for schema in ({"const": 1}, {"enum": [1]}):
            with self.subTest(schema=schema), self.assertRaisesRegex(
                bm.ValidationError, "must equal|one of"
            ):
                bm._validate_schema_structure(True, schema, schema, "boolean")
        mutations = (
            (lambda result: result.update(schema_version=2), "must equal"),
            (lambda result: result.update(schema_version=True), "must equal"),
            (
                lambda result: result["run_manifest"].update(sha256="g" * 64),
                "pattern",
            ),
            (
                lambda result: result["provenance"]["requested_execution"].update(
                    precision="double"
                ),
                "one of",
            ),
            (
                lambda result: result["memory"].update(peak_host_bytes=0.5),
                "integer",
            ),
            (
                lambda result: result["memory"].update(peak_host_bytes=True),
                "integer",
            ),
            (
                lambda result: result["memory"].update(peak_host_bytes=-1),
                "minimum",
            ),
            (
                lambda result: result["simulation"].update(dt_meep=0.0),
                "exclusive minimum",
            ),
            (
                lambda result: result["simulation"].update(grid_shape=[1, 1]),
                "minItems",
            ),
            (
                lambda result: result["simulation"].update(grid_shape=[1, 1, 1, 1]),
                "maxItems",
            ),
            (
                lambda result: result["run_manifest"].update(case_id=""),
                "minLength",
            ),
        )
        for mutate, message in mutations:
            temporary, result, _manifest = self.template()
            self.addCleanup(temporary.cleanup)
            mutate(result)
            with self.subTest(message=message), self.assertRaisesRegex(
                bm.ValidationError, message
            ):
                bm.validate_result_document(result)


class CliTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, str(bm.HERE / "benchmark_manifest.py"), *args],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_cli_reports_missing_checkout(self):
        result = self.run_cli(
            "manifest",
            "--fdtd-pipeline",
            "/does/not/exist",
            "--device",
            "coupler",
            "--mode",
            "smoke",
            "--n-max",
            "3.48",
            "--backend",
            "cpu",
            "--precision",
            "mixed",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not a directory", result.stderr)

    def test_cli_result_template_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            manifest = root / "manifest.json"
            result_path = root / "result.json"
            reference = bm.load_reference()
            cases = bm.load_case_definitions()
            manifest.write_text(
                json.dumps(build_test_manifest(reference, cases)), encoding="utf-8"
            )
            created = self.run_cli(
                "result-template",
                "--manifest",
                str(manifest),
                "--output",
                str(result_path),
            )
            self.assertEqual(created.returncode, 0, created.stderr)
            validated = self.run_cli("validate-result", "--result", str(result_path))
            self.assertEqual(validated.returncode, 0, validated.stderr)
            self.assertTrue(json.loads(validated.stdout)["valid"])


if __name__ == "__main__":
    unittest.main()
