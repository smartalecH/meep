import copy
import json
import math
import pathlib
import tempfile
import unittest
from unittest import mock

import benchmark_manifest as bm
import run_benchmark as runner
from test_benchmark_manifest import build_test_manifest


class RunnerManifestTests(unittest.TestCase):
    def setUp(self):
        self.reference = bm.load_reference()
        self.cases = bm.load_case_definitions()

    def manifest(self, device="crossing", backend="cpu"):
        return build_test_manifest(
            self.reference,
            self.cases,
            device_name=device,
            backend=backend,
            precision="native",
            ranks=1,
            mpi_transport="none",
            cells_per_material_wavelength=6,
        )

    def test_manifest_binds_runtime_materials_and_sampling(self):
        manifest = self.manifest()
        self.assertEqual(
            manifest["materials"]["performance_adaptation"]["Si"]["epsilon"], 12.1104
        )
        sampling = manifest["excitation"]["monitor_sampling"]
        self.assertEqual(sampling["wavelengths_um"], [1.54, 1.55, 1.56])
        self.assertEqual(sampling["dft_decimation_factor"], 1)
        self.assertEqual(
            sampling["frequencies_meep"],
            [1.0 / 1.54, 1.0 / 1.55, 1.0 / 1.56],
        )

    def test_psr_is_explicitly_unsupported(self):
        manifest = self.manifest(device="psr")
        with self.assertRaisesRegex(runner.RunnerError, "top-cladding"):
            runner.validate_executable_manifest(manifest)

    def test_transformed_crossing_planes_fit_non_pml_cell(self):
        manifest = self.manifest()
        ports = runner.transformed_ports(manifest, {"x_um": 0.0, "y_um": 0.0})
        planes = runner.validate_planes(manifest, ports)
        self.assertEqual(planes["source"]["size_um"], [0.0, 0.75, 2.0])
        self.assertEqual(planes["input_incident"]["center_um"], [-3.0, 0.0, 0.0])
        self.assertEqual(planes["through_te0"]["center_um"], [4.0, 0.0, 0.0])

    def test_non_single_rank_and_paper_materials_are_rejected(self):
        manifest = self.manifest(backend="cpu")
        manifest["execution"]["requested"].update(ranks=2, mpi_transport="host")
        with self.assertRaisesRegex(runner.RunnerError, "single-rank"):
            runner.validate_executable_manifest(manifest)
        manifest = self.manifest()
        manifest["materials"]["mode"] = "paper"
        with self.assertRaises((bm.ValidationError, runner.RunnerError)):
            runner.validate_executable_manifest(manifest)

    def test_gds_replacement_after_snapshot_cannot_change_parsed_geometry(self):
        class Polygon:
            points = [(-2.0, -1.0), (4.0, -1.0), (4.0, 3.0), (-2.0, 3.0)]

        class Cell:
            name = "crossing"

            def get_polygons(self, **unused):
                return [Polygon()]

        manifest = self.manifest()
        manifest["case"]["layers"] = [manifest["case"]["layers"][0]]
        with tempfile.TemporaryDirectory() as temporary:
            source = pathlib.Path(temporary) / "source.gds"
            source.write_bytes(b"authenticated geometry")
            manifest["input_checkout"]["gds"].update(
                path=str(source),
                sha256=runner.hashlib.sha256(source.read_bytes()).hexdigest(),
            )

            class Gdstk:
                @staticmethod
                def read_gds(path):
                    source.write_bytes(b"replacement geometry")
                    self.assertEqual(
                        pathlib.Path(path).read_bytes(), b"authenticated geometry"
                    )
                    return type("Library", (), {"cells": [Cell()]})()

            records, translation = runner._geometry_records(manifest, Gdstk())
            self.assertEqual(translation, {"x_um": -1.0, "y_um": -1.0})
            self.assertEqual(records[0]["points_um"][0], [-3.0, -2.0])
            self.assertEqual(source.read_bytes(), b"replacement geometry")


class RunnerResultTests(unittest.TestCase):
    def setUp(self):
        self.reference = bm.load_reference()
        self.cases = bm.load_case_definitions()
        geometry = mock.patch.object(
            runner,
            "_geometry_records",
            return_value=([], {"x_um": 0.0, "y_um": 0.0}),
        )
        gdstk = mock.patch.object(runner, "_load_gdstk_module", return_value=object())
        geometry.start()
        gdstk.start()
        self.addCleanup(geometry.stop)
        self.addCleanup(gdstk.stop)

    def manifest(self, *, backend="cpu", mode="smoke"):
        return build_test_manifest(
            self.reference,
            self.cases,
            device_name="crossing",
            backend=backend,
            precision="native",
            cells_per_material_wavelength=6,
            mode=mode,
            steps=20 if mode == "fixed-step" else None,
        )

    def complete_result(self, manifest_path, manifest, manifest_sha256, *, profiled=False):
        requested = manifest["execution"]["requested"]
        steps = 1 if profiled else int(manifest["stopping"]["steps"])
        warmup = 1 if profiled else int(manifest["execution"]["warmup_steps"])
        repetitions = 1 if profiled else int(manifest["execution"]["measured_repetitions"])
        shape = runner._expected_grid_shape(manifest)
        timestep = (
            float(manifest["case"]["time_stepping"]["courant_factor"])
            / float(manifest["discretization"]["resolution_px_per_um"])
        )
        frequency_count = len(
            manifest["excitation"]["monitor_sampling"]["frequencies_meep"]
        )
        monitors = [
            {
                "name": definition["name"],
                "port": definition["port"],
                "mode_band": int(definition["mode_order"]) + 1,
                "raw_dft_flux": [0.0] * frequency_count,
                "forward_mode_power": [0.0] * frequency_count,
                "backward_mode_power": [0.0] * frequency_count,
            }
            for definition in manifest["case"]["monitors"]
            if definition["kind"] == "mode"
        ]
        runs = [
            {
                "initialization_seconds": 1.0,
                "advance_seconds": 2.0,
                "grid_shape": shape,
                "grid_points_exact": math.prod(shape),
                "dt_meep": timestep,
                "steps": steps,
                "warmup_steps": warmup,
                "total_steps": steps + warmup,
                "physical_time_meep": (steps + warmup) * timestep,
                "monitors": copy.deepcopy(monitors),
            }
            for _ in range(repetitions)
        ]
        translation = {"x_um": 0.0, "y_um": 0.0}
        planes = runner.validate_planes(
            manifest, runner.transformed_ports(manifest, translation)
        )
        devices = []
        if requested["backend"] == "nvidia":
            devices = [
                {
                    "visible_device": 0,
                    "process_device_id": 0,
                    "physical_selector": "0",
                    "uuid": "GPU-test",
                    "name": "test GPU",
                    "memory_bytes": 1024,
                    "sm_clock_hz": 1.0,
                    "memory_clock_hz": 1.0,
                    "driver_version": "test",
                }
            ]
        return {
            "schema_version": 1,
            "kind": "paper_2506_16665_single_rank_diagnostic",
            "generated_at_utc": "2026-08-28T00:00:00+00:00",
            "status": {
                "succeeded": True,
                "diagnostic_only": True,
                "profiled": profiled,
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
                "case_id": "crossing",
                "physics_configuration_sha256": manifest["validation_policy"][
                    "reference"
                ]["physics_configuration_sha256"],
            },
            "provenance": {
                "meep_build_source": "/source",
                "meep_build_commit": "0" * 40,
                "meep_build_dirty": False,
                "runner_commit": "1" * 40,
                "runner_dirty": False,
                "build_directory": None,
                "configure_flags": None,
                "python": "test",
                "meep_module": "/test/meep.py",
                "gdstk_version": "test",
                "cuda_toolkit": None,
                "argv": ["run_benchmark.py"],
                "cwd": "/tmp",
                "environment": {},
                "requested_execution": copy.deepcopy(requested),
                "device_records": devices,
            },
            "execution": {
                "device_id": 0,
                "profile_steps": steps if profiled else None,
                "steps": steps,
                "warmup_steps": warmup,
                "measured_repetitions": repetitions,
            },
            "geometry": {
                "translation_um": translation,
                "planes": planes,
                "material_constants": copy.deepcopy(
                    manifest["materials"]["performance_adaptation"]
                ),
                "manifest_inputs": runner._manifest_geometry_inputs(manifest),
            },
            "sampling": copy.deepcopy(
                manifest["excitation"]["monitor_sampling"]
            ),
            "observable_policy": copy.deepcopy(manifest["validation_policy"]),
            "runs": runs,
            "timing_summary": {
                "samples_seconds": [2.0] * repetitions,
                "minimum_seconds": 2.0,
                "median_seconds": 2.0,
                "maximum_seconds": 2.0,
            },
            "observable_interpretation": (
                "Raw finite DFT flux and mode powers only. These PR3 diagnostics "
                "have no authenticated CPU baseline or verified physics parity and "
                "cannot support speedup or publication claims. Ten-step smoke values "
                "are not normalized transmission, conversion efficiency, resonance, "
                "or loss results."
            ),
        }

    def test_complete_result_is_strictly_bound_to_manifest(self):
        manifest = self.manifest()
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = pathlib.Path(temporary) / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            loaded, digest = runner._manifest_snapshot(manifest_path)
            result = self.complete_result(manifest_path, loaded, digest)
            runner.validate_runner_result(
                result, manifest_path, manifest=loaded, manifest_sha256=digest
            )
            mutations = {
                "missing status": lambda value: value.pop("status"),
                "false success": lambda value: value["status"].update(succeeded=False),
                "speedup claim": lambda value: value["claim_boundary"].update(
                    speedup_claim_permitted=True
                ),
                "backend provenance": lambda value: value["provenance"][
                    "requested_execution"
                ].update(backend="nvidia"),
                "precision provenance": lambda value: value["provenance"][
                    "requested_execution"
                ].update(precision="f32"),
                "rank provenance": lambda value: value["provenance"][
                    "requested_execution"
                ].update(ranks=2),
                "CPU device record": lambda value: value["provenance"][
                    "device_records"
                ].append({}),
                "step override": lambda value: value["execution"].update(steps=1),
                "geometry input": lambda value: value["geometry"]["manifest_inputs"][
                    "paper_domain_um"
                ].__setitem__(0, 999.0),
                "geometry translation": lambda value: value["geometry"][
                    "translation_um"
                ].update(x_um=999.0),
                "material": lambda value: value["geometry"]["material_constants"][
                    "Si"
                ].update(epsilon=1.0),
                "sampling": lambda value: value["sampling"]["frequencies_meep"].__setitem__(
                    0, 1.0
                ),
                "grid": lambda value: value["runs"][0]["grid_shape"].__setitem__(0, 1),
                "duplicate monitor": lambda value: value["runs"][0]["monitors"][1].update(
                    name="input_incident"
                ),
                "monitor port": lambda value: value["runs"][0]["monitors"][0].update(
                    port="o4"
                ),
                "monitor band": lambda value: value["runs"][0]["monitors"][0].update(
                    mode_band=2
                ),
                "extra repetition": lambda value: value["runs"].append(
                    copy.deepcopy(value["runs"][0])
                ),
                "unknown field": lambda value: value.update(untrusted=True),
            }
            for label, mutate in mutations.items():
                with self.subTest(label=label):
                    changed = copy.deepcopy(result)
                    mutate(changed)
                    with self.assertRaises(runner.RunnerError):
                        runner.validate_runner_result(
                            changed,
                            manifest_path,
                            manifest=loaded,
                            manifest_sha256=digest,
                        )

            changed = copy.deepcopy(result)
            changed["geometry"]["translation_um"]["y_um"] = 0.1
            changed["geometry"]["planes"] = runner.validate_planes(
                loaded,
                runner.transformed_ports(
                    loaded, changed["geometry"]["translation_um"]
                ),
            )
            with self.assertRaisesRegex(runner.RunnerError, "authenticated GDS bounds"):
                runner.validate_runner_result(
                    changed,
                    manifest_path,
                    manifest=loaded,
                    manifest_sha256=digest,
                )

    def test_result_validation_rejects_nonfinite_monitor_output(self):
        manifest = self.manifest()
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = pathlib.Path(temporary) / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            loaded, digest = runner._manifest_snapshot(manifest_path)
            result = self.complete_result(manifest_path, loaded, digest)
            result["runs"][0]["monitors"][0]["forward_mode_power"][0] = math.nan
            with self.assertRaisesRegex(runner.RunnerError, "not finite"):
                runner.validate_runner_result(
                    result, manifest_path, manifest=loaded, manifest_sha256=digest
                )

    def test_manifest_snapshot_prevents_path_mutation_from_rebinding_result(self):
        manifest = self.manifest()
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = pathlib.Path(temporary) / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            loaded, digest = runner._manifest_snapshot(manifest_path)
            result = self.complete_result(manifest_path, loaded, digest)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            runner.validate_runner_result(
                result, manifest_path, manifest=loaded, manifest_sha256=digest
            )
            with self.assertRaisesRegex(runner.RunnerError, "manifest hash"):
                runner.validate_runner_result(result, manifest_path)

    def test_nvidia_device_identity_is_bound_to_execution(self):
        manifest = self.manifest(backend="nvidia")
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = pathlib.Path(temporary) / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            loaded, digest = runner._manifest_snapshot(manifest_path)
            result = self.complete_result(manifest_path, loaded, digest)
            runner.validate_runner_result(
                result, manifest_path, manifest=loaded, manifest_sha256=digest
            )
            result["provenance"]["device_records"][0]["process_device_id"] = 1
            with self.assertRaisesRegex(runner.RunnerError, "device_id"):
                runner.validate_runner_result(
                    result, manifest_path, manifest=loaded, manifest_sha256=digest
                )

    def test_fixed_step_without_verified_cpu_baseline_remains_nonpublishable(self):
        manifest = self.manifest(mode="fixed-step")
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = pathlib.Path(temporary) / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            loaded, digest = runner._manifest_snapshot(manifest_path)
            result = self.complete_result(manifest_path, loaded, digest)
            self.assertFalse(result["claim_boundary"]["speedup_claim_permitted"])
            self.assertFalse(result["claim_boundary"]["publication_claim_permitted"])
            runner.validate_runner_result(
                result, manifest_path, manifest=loaded, manifest_sha256=digest
            )


if __name__ == "__main__":
    unittest.main()
