import copy
import json
import math
import pathlib
import sys
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

    def manifest(self, *, backend="cpu", mode="smoke", precision="native"):
        return build_test_manifest(
            self.reference,
            self.cases,
            device_name="crossing",
            backend=backend,
            precision=precision,
            cells_per_material_wavelength=6,
            mode=mode,
            steps=20 if mode == "fixed-step" else None,
        )

    def complete_result(
        self,
        manifest_path,
        manifest,
        manifest_sha256,
        *,
        profiled=False,
        accelerator="cuda"
    ):
        requested = manifest["execution"]["requested"]
        steps = 1 if profiled else int(manifest["stopping"]["steps"])
        warmup = 1 if profiled else int(manifest["execution"]["warmup_steps"])
        repetitions = (
            1 if profiled else int(manifest["execution"]["measured_repetitions"])
        )
        shape = runner._expected_grid_shape(manifest)
        timestep = float(manifest["case"]["time_stepping"]["courant_factor"]) / float(
            manifest["discretization"]["resolution_px_per_um"]
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
        runtime_accelerator = accelerator if requested["backend"] == "nvidia" else None
        device_bytes = 1024 if runtime_accelerator else 0
        pinned_bytes = 128 if runtime_accelerator else 0

        def memory_gauge(name):
            if name in {"process_device_bytes_current", "process_device_bytes_peak"}:
                return device_bytes
            if name in {"process_pinned_bytes_current", "process_pinned_bytes_peak"}:
                return pinned_bytes
            return 0

        counters = {name: 0 for name in runner.RUNTIME_COUNTER_NAMES}
        runs = []
        for index in range(repetitions):
            start_step = warmup + index * steps
            runs.append(
                {
                    "initialization_seconds": 1.0 if index == 0 else 0.0,
                    "warmup_seconds": 1.0 if index == 0 and warmup else 0.0,
                    "advance_seconds": 2.0,
                    "grid_shape": shape,
                    "grid_points_exact": math.prod(shape),
                    "dt_meep": timestep,
                    "steps": steps,
                    "warmup_steps": warmup if index == 0 else 0,
                    "start_step": start_step,
                    "end_step": start_step + steps,
                    "physical_time_meep": (start_step + steps) * timestep,
                    "counter_start": copy.deepcopy(counters),
                    "counter_end": copy.deepcopy(counters),
                    "counter_deltas": copy.deepcopy(counters),
                    "memory_start": {
                        "host_peak_bytes": 4096,
                        **{
                            name: memory_gauge(name)
                            for name in runner.RUNTIME_MEMORY_NAMES
                        },
                    },
                    "memory_end": {
                        "host_peak_bytes": 4096,
                        **{
                            name: memory_gauge(name)
                            for name in runner.RUNTIME_MEMORY_NAMES
                        },
                    },
                    "monitors": copy.deepcopy(monitors),
                }
            )
        translation = {"x_um": 0.0, "y_um": 0.0}
        planes = runner.validate_planes(
            manifest, runner.transformed_ports(manifest, translation)
        )
        devices = []
        if requested["backend"] == "nvidia":
            devices = [
                {
                    "accelerator": accelerator,
                    "visible_device": 0,
                    "visible_devices": ["0"],
                    "process_device_id": 0,
                    "physical_selector": "0",
                    "inventory_index": 0,
                    "uuid": "GPU-11111111-2222-3333-4444-555555555555",
                    "name": "test GPU",
                    "pci_bus_id": "0000:01:00.0",
                    "memory_bytes": 1024,
                    "core_clock": "1 MHz",
                    "memory_clock": "1 MHz",
                    "driver_version": "test",
                }
            ]
            if accelerator == "hip":
                devices[0].pop("inventory_index")
                devices[0].update(
                    inventory_card="card0",
                    physical_unique_id="0x1234",
                    numa_node=0,
                    architecture="gfx950",
                )
        runtime = {
            **{name: "" for name in runner.RUNTIME_STRING_NAMES},
            **{name: False for name in runner.RUNTIME_BOOL_NAMES},
            **{name: 0 for name in runner.RUNTIME_INTEGER_NAMES},
            **{name: 0 for name in runner.RUNTIME_COUNTER_NAMES},
            **{name: 0 for name in runner.RUNTIME_MEMORY_NAMES},
        }
        runtime.update(
            requested_backend=requested["backend"],
            resolved_backend=requested["backend"],
            requested_precision=requested["precision"],
            resolved_precision=requested["precision"],
            requested_transport="staged" if runtime_accelerator else "none",
            resolved_transport="none",
            requested_overlap=requested["overlap"],
            resolved_overlap="off",
            captured_requested_transport="none",
            captured_overlap_policy="off",
            requested_graph=requested["graph"],
            resolved_graph="eager",
            mpi_provider="test",
            counter_scope="rank_local_current_epoch",
            backend_counter_scope="rank_local_backend_lifetime",
            memory_gauge_scope="rank_local_process_lifetime",
            setup_counter_scope="rank_local_current_backend_state",
            transport_timing_scope="rank_local_current_transport_epoch_host_elapsed",
            allocation_counter_scope="rank_local_process_lifetime",
            communicator_size=1,
            device_id=0 if runtime_accelerator else -1,
            device_owner=runtime_accelerator is not None,
            device_uuid=(
                "GPU-11111111-2222-3333-4444-555555555555"
                if runtime_accelerator
                else ""
            ),
            executable_build_count=1 if runtime_accelerator else 0,
            process_device_bytes_current=device_bytes,
            process_device_bytes_peak=device_bytes,
            process_pinned_bytes_current=pinned_bytes,
            process_pinned_bytes_peak=pinned_bytes,
        )
        executable = pathlib.Path(sys.executable).resolve()
        worktree = pathlib.Path(runner.__file__).resolve().parents[3]
        source_state = runner._source_tree_state(worktree)
        environment = {
            "MEEP_FINITE_CHECK": (
                "off" if profiled or requested["mode"] == "fixed-step" else "step"
            )
        }
        if runtime_accelerator:
            environment.update(
                {
                    "MEEP_ACCELERATOR_RUNTIME": runtime_accelerator,
                    "MEEP_GPU_AWARE_MPI": "no",
                    "MEEP_NVIDIA_MPI_OVERLAP": requested["overlap"],
                    "MEEP_NVIDIA_GRAPH_MODE": requested["graph"],
                }
            )
            if runtime_accelerator == "hip":
                environment["ROCR_VISIBLE_DEVICES"] = "0"
        hip_inventory_output = json.dumps(
            {
                "card0": {
                    "Unique ID": "0x1234",
                    "PCI Bus": "0000:01:00.0",
                    "VRAM Total Memory (B)": "1024",
                    "Card Series": "test GPU",
                    "GFX Version": "gfx950",
                    "(Topology) Numa Node": "0",
                    "sclk clock speed:": "1 MHz",
                    "mclk clock speed:": "1 MHz",
                },
                "system": {"Driver version": "test"},
            }
        )
        return {
            "schema_version": 2,
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
                "meep_build_source": str(worktree),
                "meep_build_commit": source_state["commit"],
                "meep_build_dirty": source_state["dirty"],
                "meep_build_diff_sha256": source_state["diff_sha256"],
                "runner_commit": source_state["commit"],
                "runner_dirty": source_state["dirty"],
                "runner_diff_sha256": source_state["diff_sha256"],
                "runner_source_sha256": runner.bm.sha256_file(
                    pathlib.Path(runner.__file__).resolve()
                ),
                "build_directory": None,
                "configure_flags": None,
                "python": "test",
                "meep_module": {
                    "path": str(executable),
                    "sha256": runner.bm.sha256_file(executable),
                },
                "meep_extension": {
                    "path": str(executable),
                    "sha256": runner.bm.sha256_file(executable),
                },
                "gdstk_version": "test",
                "accelerator": {
                    "runtime": runtime_accelerator,
                    "toolchain": (
                        {
                            "path": str(executable),
                            "sha256": runner.bm.sha256_file(executable),
                            "version": "test",
                        }
                        if runtime_accelerator
                        else None
                    ),
                    "inventory_tool": (
                        {
                            "path": str(executable),
                            "sha256": runner.bm.sha256_file(executable),
                        }
                        if runtime_accelerator == "hip"
                        else None
                    ),
                    "inventory_snapshot": (
                        {
                            "output": hip_inventory_output,
                            "sha256": runner.hashlib.sha256(
                                hip_inventory_output.encode("utf-8")
                            ).hexdigest(),
                        }
                        if runtime_accelerator == "hip"
                        else None
                    ),
                },
                "argv": ["run_benchmark.py"],
                "cwd": "/tmp",
                "environment": environment,
                "requested_execution": copy.deepcopy(requested),
                "device_records": devices,
            },
            "execution": {
                "device_id": 0,
                "accelerator": runtime_accelerator,
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
            "sampling": copy.deepcopy(manifest["excitation"]["monitor_sampling"]),
            "observable_policy": copy.deepcopy(manifest["validation_policy"]),
            "runtime": runtime,
            "memory": {
                "host_peak_bytes": 4096,
                **{name: runtime[name] for name in runner.RUNTIME_MEMORY_NAMES},
            },
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
                "old schema": lambda value: value.update(schema_version=1),
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
                "sampling": lambda value: value["sampling"][
                    "frequencies_meep"
                ].__setitem__(0, 1.0),
                "grid": lambda value: value["runs"][0]["grid_shape"].__setitem__(0, 1),
                "duplicate monitor": lambda value: value["runs"][0]["monitors"][
                    1
                ].update(name="input_incident"),
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
                runner.transformed_ports(loaded, changed["geometry"]["translation_um"]),
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
            with self.assertRaisesRegex(runner.RunnerError, "ownership"):
                runner.validate_runner_result(
                    result, manifest_path, manifest=loaded, manifest_sha256=digest
                )

    def test_hip_device_identity_and_tool_provenance_are_bound(self):
        manifest = self.manifest(backend="nvidia")
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = pathlib.Path(temporary) / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            loaded, digest = runner._manifest_snapshot(manifest_path)
            result = self.complete_result(
                manifest_path, loaded, digest, accelerator="hip"
            )
            runner.validate_runner_result(
                result, manifest_path, manifest=loaded, manifest_sha256=digest
            )
            mutations = {
                "runtime UUID": lambda value: value["runtime"].update(
                    device_uuid="GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
                ),
                "PCI BDF": lambda value: value["provenance"]["device_records"][
                    0
                ].update(pci_bus_id="not-a-bdf"),
                "valid wrong PCI BDF": lambda value: value["provenance"][
                    "device_records"
                ][0].update(pci_bus_id="0000:88:00.0"),
                "physical unique ID": lambda value: value["provenance"][
                    "device_records"
                ][0].update(physical_unique_id="forged"),
                "inventory hash": lambda value: value["provenance"]["accelerator"][
                    "inventory_tool"
                ].update(sha256="0" * 64),
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

    def test_rocm_provenance_joins_runtime_bdf_not_card_ordinal(self):
        inventory = {
            "card7": {
                "Unique ID": "0x1234",
                "PCI Bus": "0000:88:00.0",
                "VRAM Total Memory (B)": "309220868096",
                "Card Series": "AMD Radeon Graphics",
                "GFX Version": "gfx950",
                "(Topology) Numa Node": "1",
                "sclk clock speed:": "(1400Mhz)",
                "mclk clock speed:": "(2000Mhz)",
            },
            "system": {"Driver version": "test-driver"},
        }
        completed = type("Completed", (), {"stdout": json.dumps(inventory)})()
        with mock.patch.object(
            runner, "_hip_pci_bus_id", return_value="0000:88:00.0"
        ), mock.patch.object(runner.subprocess, "run", return_value=completed):
            records, snapshot = runner._rocm_device_provenance(
                0,
                "GPU-11111111-2222-3333-4444-555555555555",
                ["5"],
                pathlib.Path("/opt/rocm/bin/rocm-smi"),
            )
            record = records[0]
        self.assertEqual(record["physical_selector"], "5")
        self.assertEqual(record["inventory_card"], "card7")
        self.assertEqual(record["pci_bus_id"], "0000:88:00.0")
        self.assertEqual(record["memory_bytes"], 309220868096)
        self.assertEqual(
            snapshot["sha256"],
            runner.hashlib.sha256(snapshot["output"].encode()).hexdigest(),
        )

    def test_runtime_counters_memory_and_source_tampering_are_rejected(self):
        manifest = self.manifest(backend="nvidia", mode="fixed-step")
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = pathlib.Path(temporary) / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            loaded, digest = runner._manifest_snapshot(manifest_path)
            result = self.complete_result(manifest_path, loaded, digest)
            mutations = {
                "counter delta": lambda value: value["runs"][0]["counter_end"].update(
                    graph_launch_count=1
                ),
                "measured allocation": lambda value: (
                    value["runs"][0]["counter_end"].update(steady_allocation_count=1),
                    value["runs"][0]["counter_deltas"].update(
                        steady_allocation_count=1
                    ),
                ),
                "measured full copy": lambda value: (
                    value["runs"][0]["counter_end"].update(full_field_copy_count=1),
                    value["runs"][0]["counter_deltas"].update(full_field_copy_count=1),
                ),
                "measured graph recapture": lambda value: (
                    value["runs"][0]["counter_end"].update(graph_recapture_count=1),
                    value["runs"][0]["counter_deltas"].update(graph_recapture_count=1),
                ),
                "fallback": lambda value: value["runtime"].update(
                    host_fallback_count=1
                ),
                "material warning": lambda value: value["runtime"].update(
                    material_fallback_warning_count=1
                ),
                "memory": lambda value: value["memory"].update(
                    process_device_bytes_peak=2048
                ),
                "cross-window host peak": lambda value: value["runs"][1][
                    "memory_start"
                ].update(host_peak_bytes=4095),
                "cross-window device peak": lambda value: value["runs"][1][
                    "memory_start"
                ].update(process_device_bytes_peak=1023),
                "final device peak": lambda value: (
                    value["memory"].update(process_device_bytes_peak=1023),
                    value["runtime"].update(process_device_bytes_peak=1023),
                ),
                "final pinned peak": lambda value: (
                    value["memory"].update(process_pinned_bytes_peak=127),
                    value["runtime"].update(process_pinned_bytes_peak=127),
                ),
                "resolved precision": lambda value: value["runtime"].update(
                    resolved_precision="f32"
                ),
                "resolved backend": lambda value: value["runtime"].update(
                    resolved_backend="cpu"
                ),
                "eager graph enabled": lambda value: value["runtime"].update(
                    graph_enabled=True
                ),
                "missing finite check": lambda value: value["provenance"][
                    "environment"
                ].pop("MEEP_FINITE_CHECK"),
                "wrong finite check": lambda value: value["provenance"][
                    "environment"
                ].update(MEEP_FINITE_CHECK="step"),
                "second initialization": lambda value: value["runs"][1].update(
                    initialization_seconds=1.0
                ),
                "nonsequential window": lambda value: value["runs"][1].update(
                    start_step=value["runs"][1]["start_step"] + 1
                ),
                "runner hash": lambda value: value["provenance"].update(
                    runner_source_sha256="0" * 64
                ),
                "missing Meep module": lambda value: value["provenance"][
                    "meep_module"
                ].update(path="/tmp/nonexistent/meep.py"),
                "source state": lambda value: value["provenance"].update(
                    runner_diff_sha256="0" * 64
                ),
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

    def test_fixed_step_uses_one_initialization_and_sequential_windows(self):
        manifest = self.manifest(mode="fixed-step")
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = pathlib.Path(temporary) / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            loaded, digest = runner._manifest_snapshot(manifest_path)
            runtime = self.complete_result(manifest_path, loaded, digest)["runtime"]

        class Fields:
            def __init__(self):
                self.t = 0
                self.dt = 0.5 / 14.0
                self.advances = []

            def advance(self, steps):
                self.advances.append(steps)
                self.t += steps

        class Simulation:
            def __init__(self):
                self.fields = Fields()
                self.reset_count = 0

            def get_execution_runtime_report(self):
                value = copy.deepcopy(runtime)
                value["graph_launch_count"] = self.fields.t
                return value

            def reset_meep(self):
                self.reset_count += 1

        class Mp:
            @staticmethod
            def all_wait():
                pass

        simulation = Simulation()
        with mock.patch.object(
            runner, "_build_simulation", return_value=(simulation, {})
        ) as build, mock.patch.object(
            runner, "_grid_shape", return_value=runner._expected_grid_shape(manifest)
        ), mock.patch.object(
            runner, "_monitor_output", return_value=[]
        ), mock.patch.object(
            runner, "_host_peak_bytes", return_value=4096
        ):
            runs, final_runtime, memory = runner._run_session(
                Mp(),
                manifest,
                [],
                {},
                device_id=0,
                steps=20,
                warmup_steps=100,
                repetitions=5,
                profile=False,
            )
        build.assert_called_once()
        self.assertEqual(simulation.fields.advances, [100, 20, 20, 20, 20, 20])
        self.assertEqual([run["start_step"] for run in runs], [100, 120, 140, 160, 180])
        self.assertEqual([run["end_step"] for run in runs], [120, 140, 160, 180, 200])
        self.assertEqual([run["warmup_steps"] for run in runs], [100, 0, 0, 0, 0])
        self.assertEqual(
            [run["initialization_seconds"] > 0 for run in runs],
            [True, False, False, False, False],
        )
        self.assertEqual(
            [run["counter_deltas"]["graph_launch_count"] for run in runs], [20] * 5
        )
        self.assertEqual(final_runtime["graph_launch_count"], 200)
        self.assertEqual(memory["host_peak_bytes"], 4096)
        self.assertEqual(simulation.reset_count, 1)

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
