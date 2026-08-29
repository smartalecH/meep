import json
import math
import pathlib
import tempfile
import unittest

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


class RunnerResultTests(unittest.TestCase):
    def test_result_validation_rejects_nonfinite_monitor_output(self):
        reference = bm.load_reference()
        cases = bm.load_case_definitions()
        manifest = build_test_manifest(
            reference,
            cases,
            device_name="crossing",
            backend="cpu",
            precision="native",
            cells_per_material_wavelength=6,
        )
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = pathlib.Path(temporary) / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result = {
                "schema_version": 1,
                "kind": "paper_2506_16665_single_rank_diagnostic",
                "run_manifest": {
                    "sha256": bm.sha256_file(manifest_path),
                    "case_id": "crossing",
                },
                "runs": [
                    {
                        "grid_shape": [2, 3, 4],
                        "grid_points_exact": 24,
                        "initialization_seconds": 1.0,
                        "advance_seconds": 2.0,
                        "dt_meep": 0.1,
                        "steps": 10,
                        "warmup_steps": 0,
                        "total_steps": 10,
                        "physical_time_meep": 1.0,
                        "monitors": [
                            {
                                "name": "input_incident",
                                "raw_dft_flux": [0.0, 0.0, 0.0],
                                "forward_mode_power": [math.nan, 0.0, 0.0],
                                "backward_mode_power": [0.0, 0.0, 0.0],
                            },
                            {
                                "name": "through_te0",
                                "raw_dft_flux": [0.0, 0.0, 0.0],
                                "forward_mode_power": [0.0, 0.0, 0.0],
                                "backward_mode_power": [0.0, 0.0, 0.0],
                            },
                        ],
                    }
                ],
                "timing_summary": {
                    "samples_seconds": [2.0],
                    "minimum_seconds": 2.0,
                    "median_seconds": 2.0,
                    "maximum_seconds": 2.0,
                },
            }
            with self.assertRaisesRegex(runner.RunnerError, "not finite"):
                runner.validate_runner_result(result, manifest_path)


if __name__ == "__main__":
    unittest.main()
