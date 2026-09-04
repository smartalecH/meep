#!/usr/bin/env python3

import copy
import contextlib
import importlib.util
import io
import json
import math
import os
import pathlib
import subprocess
import tempfile
import unittest
from unittest import mock


SCRIPT = pathlib.Path(__file__).with_name("run_transport_acceptance.py")
SPEC = importlib.util.spec_from_file_location("run_transport_acceptance", SCRIPT)
acceptance = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(acceptance)


def runtime(route, rank, size):
    gpu = route != "cpu"
    return {
        "resolved_backend": "nvidia" if gpu else "cpu",
        "requested_transport": route if gpu else "none",
        "resolved_transport": route if gpu else "none",
        "captured_requested_transport": route if gpu else "none",
        "mpi_provider": "Open MPI/UCX",
        "mpi_query_available": route == "direct",
        "mpi_cuda_aware": route == "direct",
        "communicator_rank": rank,
        "communicator_size": size,
        "device_owner": gpu,
        "captured_transport_epoch_active": gpu,
        "captured_transport_epoch_fresh": gpu,
        "transport_pinned_bytes": 64 if route == "staged" else 0,
        "device_id": rank if gpu else -1,
        "device_uuid": f"GPU-{rank}" if gpu else "",
        "host_fallback_count": 0,
        "host_fallback_device_to_host_bytes": 0,
        "host_fallback_host_to_device_bytes": 0,
        "host_fallback_steady_capacity_growths": 0,
        "material_fallback_warning_count": 0,
    }


def counters(route):
    result = {name: 0 for name in acceptance.mpi_runner.COUNTER_AGGREGATIONS}
    if route != "cpu":
        result.update(
            messages_sent=1, messages_received=1, bytes_sent=16, bytes_received=16
        )
    if route == "staged":
        result.update(
            device_to_host_calls=1,
            device_to_host_bytes=16,
            host_to_device_calls=1,
            host_to_device_bytes=16,
            slot_reuses=1,
        )
    elif route == "direct":
        result["direct_bytes"] = 32
    return result


def artifact(route):
    size = 1 if route == "cpu" else 2
    records = []
    for rank in range(size):
        records.append(
            {
                "rank": rank,
                "runtime": runtime(route, rank, size),
                "steps": 10,
                "stop_reason": "field_energy_decay",
                "observables": {
                    "samples": [
                        [float(index), 0.0]
                        for index in range(len(acceptance.CASE["sample_z"]))
                    ]
                },
                "device": (
                    {
                        "logical_device_id": rank,
                        "visibility_selector": str(rank),
                        "physical_selector": str(rank),
                        "runtime_uuid": f"GPU-{rank}",
                        "physical_unique_id": f"physical-{rank}",
                        "pci_bus_id": f"0000:{rank + 8:02x}:00.0",
                        "numa_node": rank,
                        "name": "test GPU",
                        "architecture": "",
                    }
                    if route != "cpu"
                    else None
                ),
                "cpu_affinity": [rank],
                "cpu_numa_nodes": [rank],
                "counter_deltas": counters(route),
            }
        )
    return {
        "schema_version": 2,
        "kind": "meep_builtin_transport_acceptance",
        "case": copy.deepcopy(acceptance.CASE),
        "case_sha256": acceptance.canonical_hash(acceptance.CASE),
        "route": route,
        "build_identity": {
            "source": "/tmp/meep",
            "commit": "a" * 40,
            "dirty": False,
            "python": "/tmp/python",
            "meep_module": "/tmp/meep/__init__.py",
            "meep_extension": "/tmp/_meep.so",
        },
        "rank_records": records,
    }


def comparison():
    reference = {"path": "/tmp/result.json", "sha256": "a" * 64}
    return {
        "schema_version": 2,
        "kind": "meep_builtin_transport_acceptance_comparison",
        "case_sha256": acceptance.canonical_hash(acceptance.CASE),
        "cpu": copy.deepcopy(reference),
        "staged": copy.deepcopy(reference),
        "cpu_tolerance_passed": True,
    }


class AcceptanceArtifactValidationTest(unittest.TestCase):
    def test_valid_cpu_staged_and_direct_artifacts(self):
        for route in ("cpu", "staged", "direct"):
            with self.subTest(route=route):
                acceptance.validate_acceptance_artifact(artifact(route), route)

    def test_semantically_empty_direct_artifact_is_rejected(self):
        value = artifact("direct")
        for record in value["rank_records"]:
            record["counter_deltas"] = counters("cpu")
        with self.assertRaisesRegex(acceptance.mpi_runner.RunnerError, "do not prove"):
            acceptance.validate_acceptance_artifact(value, "direct")

    def test_case_and_hash_tampering_are_rejected(self):
        value = artifact("direct")
        value["case"]["resolution"] += 1
        with self.assertRaisesRegex(acceptance.mpi_runner.RunnerError, "exactly match"):
            acceptance.validate_acceptance_artifact(value, "direct")
        value = artifact("direct")
        value["case_sha256"] = "b" * 64
        with self.assertRaisesRegex(acceptance.mpi_runner.RunnerError, "case hash"):
            acceptance.validate_acceptance_artifact(value, "direct")

    def test_runtime_route_provider_and_pinned_tampering_are_rejected(self):
        mutations = (
            (
                lambda value: value["rank_records"][0]["runtime"].update(
                    resolved_transport="staged"
                ),
                "resolved route",
            ),
            (
                lambda value: value["rank_records"][0]["runtime"].update(
                    mpi_provider=""
                ),
                "MPI provider",
            ),
            (
                lambda value: value["rank_records"][0]["runtime"].update(
                    mpi_query_available=False
                ),
                "GPU-aware MPI",
            ),
            (
                lambda value: value["rank_records"][0]["runtime"].update(
                    transport_pinned_bytes=1
                ),
                "GPU-aware MPI",
            ),
        )
        for mutate, message in mutations:
            with self.subTest(message=message):
                value = artifact("direct")
                mutate(value)
                with self.assertRaisesRegex(acceptance.mpi_runner.RunnerError, message):
                    acceptance.validate_acceptance_artifact(value, "direct")

    def test_rank_count_and_order_are_exact(self):
        value = artifact("direct")
        value["rank_records"].pop()
        with self.assertRaisesRegex(acceptance.mpi_runner.RunnerError, "2 ordered"):
            acceptance.validate_acceptance_artifact(value, "direct")
        value = artifact("direct")
        value["rank_records"].reverse()
        with self.assertRaisesRegex(acceptance.mpi_runner.RunnerError, "2 ordered"):
            acceptance.validate_acceptance_artifact(value, "direct")

    def test_observable_and_counter_tampering_are_rejected(self):
        value = artifact("direct")
        value["rank_records"][0]["observables"]["samples"].pop()
        with self.assertRaisesRegex(
            acceptance.mpi_runner.RunnerError, "fixed observable"
        ):
            acceptance.validate_acceptance_artifact(value, "direct")
        value = artifact("direct")
        value["rank_records"][0]["observables"]["samples"][0][0] = math.nan
        with self.assertRaisesRegex(
            acceptance.mpi_runner.RunnerError, "invalid observable"
        ):
            acceptance.validate_acceptance_artifact(value, "direct")
        value = artifact("direct")
        value["rank_records"][0]["counter_deltas"]["direct_bytes"] = -1
        with self.assertRaisesRegex(
            acceptance.mpi_runner.RunnerError, "counter direct_bytes"
        ):
            acceptance.validate_acceptance_artifact(value, "direct")
        value = artifact("direct")
        del value["rank_records"][0]["counter_deltas"]["slot_reuses"]
        with self.assertRaisesRegex(
            acceptance.mpi_runner.RunnerError, "required counter set"
        ):
            acceptance.validate_acceptance_artifact(value, "direct")

    def test_steady_state_side_effect_counters_are_rejected(self):
        for name in (
            "steady_allocation_count",
            "graph_recapture_count",
            "full_field_copy_count",
            "host_fallback_count",
            "host_fallback_device_to_host_bytes",
            "host_fallback_host_to_device_bytes",
            "host_fallback_steady_capacity_growths",
            "material_fallback_warning_count",
        ):
            with self.subTest(counter=name):
                value = artifact("direct")
                value["rank_records"][0]["counter_deltas"][name] = 1
                with self.assertRaisesRegex(
                    acceptance.mpi_runner.RunnerError, f"steady-state counter {name}"
                ):
                    acceptance.validate_acceptance_artifact(value, "direct")

    def test_gpu_identity_and_numa_placement_are_enforced(self):
        value = artifact("staged")
        value["rank_records"][1]["device"]["physical_unique_id"] = "physical-0"
        with self.assertRaisesRegex(
            acceptance.mpi_runner.RunnerError, "unique physical GPUs"
        ):
            acceptance.validate_acceptance_artifact(value, "staged")
        value = artifact("staged")
        value["rank_records"][1]["cpu_numa_nodes"] = [0]
        with self.assertRaisesRegex(
            acceptance.mpi_runner.RunnerError, "not exclusively local"
        ):
            acceptance.validate_acceptance_artifact(value, "staged")

        for nodes in ([], [0, 1], [1, 1], [1, 0]):
            with self.subTest(nodes=nodes):
                value = artifact("staged")
                value["rank_records"][1]["cpu_numa_nodes"] = nodes
                with self.assertRaises(acceptance.mpi_runner.RunnerError):
                    acceptance.validate_acceptance_artifact(value, "staged")

    def test_rocm_provenance_matches_runtime_bdf_not_visibility_ordinal(self):
        runtime = {"device_id": 0, "device_uuid": "GPU-runtime"}
        inventory = {
            "card0": {
                "Unique ID": "physical-0",
                "PCI Bus": "0000:08:00.0",
                "Card Series": "test GPU",
                "GFX Version": "gfx950",
                "(Topology) Numa Node": "0",
            },
            "card3": {
                "Unique ID": "physical-3",
                "PCI Bus": "0000:78:00.0",
                "Card Series": "test GPU",
                "GFX Version": "gfx950",
                "(Topology) Numa Node": "0",
            },
        }
        completed = subprocess.CompletedProcess(
            args=["rocm-smi"], returncode=0, stdout=json.dumps(inventory), stderr=""
        )
        environment = {
            "OMPI_COMM_WORLD_LOCAL_RANK": "0",
            "MEEP_ACCEPTANCE_VISIBLE_DEVICES": "0,4",
            "MEEP_ROCM_SMI": "/test/rocm-smi",
        }
        with mock.patch.dict(
            acceptance.os.environ, environment, clear=True
        ), mock.patch.object(
            acceptance, "_hip_pci_bus_id", return_value="0000:78:00.0"
        ), mock.patch.object(
            acceptance.subprocess, "run", return_value=completed
        ):
            device = acceptance._device_provenance(runtime)
        self.assertEqual(device["visibility_selector"], "0")
        self.assertEqual(device["physical_selector"], "3")
        self.assertEqual(device["physical_unique_id"], "physical-3")
        self.assertEqual(device["pci_bus_id"], "0000:78:00.0")

    def test_rocm_provenance_rejects_ambiguous_runtime_bdf(self):
        entry = {
            "Unique ID": "physical",
            "PCI Bus": "0000:78:00.0",
            "Card Series": "test GPU",
            "GFX Version": "gfx950",
            "(Topology) Numa Node": "0",
        }
        completed = subprocess.CompletedProcess(
            args=["rocm-smi"],
            returncode=0,
            stdout=json.dumps({"card0": entry, "card3": entry}),
            stderr="",
        )
        environment = {
            "OMPI_COMM_WORLD_LOCAL_RANK": "0",
            "MEEP_ACCEPTANCE_VISIBLE_DEVICES": "0,4",
            "MEEP_ROCM_SMI": "/test/rocm-smi",
        }
        with mock.patch.dict(
            acceptance.os.environ, environment, clear=True
        ), mock.patch.object(
            acceptance, "_hip_pci_bus_id", return_value="0000:78:00.0"
        ), mock.patch.object(
            acceptance.subprocess, "run", return_value=completed
        ), self.assertRaisesRegex(
            acceptance.mpi_runner.RunnerError, "matched 2"
        ):
            acceptance._device_provenance(
                {"device_id": 0, "device_uuid": "GPU-runtime"}
            )

    def test_comparison_direct_evidence_is_all_or_nothing(self):
        direct_fields = {
            "direct": {"path": "/tmp/direct.json", "sha256": "b" * 64},
            "staged_direct_bitwise": True,
            "device_buffer_mpi_positive": True,
        }
        value = comparison()
        acceptance.validate_comparison_artifact(value)
        value.update(direct_fields)
        acceptance.validate_comparison_artifact(value)
        for name in direct_fields:
            with self.subTest(missing=name):
                partial = copy.deepcopy(value)
                del partial[name]
                with self.assertRaisesRegex(
                    acceptance.mpi_runner.RunnerError, "entirely present or absent"
                ):
                    acceptance.validate_comparison_artifact(partial)

    def test_provider_zero_copy_is_separate_complete_evidence(self):
        evidence = {
            "provider": "ucx",
            "memory_type": "rocm",
            "transport": "rocm_ipc",
            "evidence": {"path": "/tmp/ucx.log", "sha256": "c" * 64},
        }
        value = comparison()
        value["provider_zero_copy"] = evidence
        with self.assertRaisesRegex(
            acceptance.mpi_runner.RunnerError, "requires complete device-buffer"
        ):
            acceptance.validate_comparison_artifact(value)

        value.update(
            {
                "direct": {"path": "/tmp/direct.json", "sha256": "b" * 64},
                "staged_direct_bitwise": True,
                "device_buffer_mpi_positive": True,
            }
        )
        acceptance.validate_comparison_artifact(value)

        partial = copy.deepcopy(value)
        del partial["provider_zero_copy"]["evidence"]
        with self.assertRaises(acceptance.bm.ValidationError):
            acceptance.validate_comparison_artifact(partial)

        inconsistent = copy.deepcopy(value)
        inconsistent["provider_zero_copy"]["transport"] = "cuda_ipc"
        with self.assertRaisesRegex(
            acceptance.mpi_runner.RunnerError, "memory type and transport"
        ):
            acceptance.validate_comparison_artifact(inconsistent)

    def test_staged_wire_balance_reuse_and_direct_zero_are_enforced(self):
        for name, value, message in (
            ("bytes_received", 8, "host-staged"),
            ("slot_reuses", 0, "host-staged"),
            ("direct_bytes", 1, "host-staged"),
        ):
            with self.subTest(counter=name):
                result = artifact("staged")
                result["rank_records"][0]["counter_deltas"][name] = value
                with self.assertRaisesRegex(acceptance.mpi_runner.RunnerError, message):
                    acceptance.validate_acceptance_artifact(result, "staged")

    def test_measurement_window_warms_and_fences_before_reports(self):
        events = []

        class Fields:
            t = 0

            def advance(self, steps):
                events.append(("advance", steps))
                self.t += steps

        class Simulation:
            fields = Fields()

            def get_execution_runtime_report(self):
                events.append(("report", self.fields.t))
                return {"step": self.fields.t}

        class Meep:
            @staticmethod
            def all_wait():
                events.append(("barrier", Simulation.fields.t))

        before, start = acceptance._steady_measurement_start(Meep, Simulation())
        self.assertEqual(before, {"step": 1})
        self.assertEqual(start, 1)
        self.assertEqual(events, [("advance", 1), ("barrier", 1), ("report", 1)])
        events.clear()
        after = acceptance._steady_measurement_end(Meep, Simulation())
        self.assertEqual(after, {"step": 1})
        self.assertEqual(events, [("barrier", 1), ("report", 1)])

    def test_loader_validates_before_returning_artifact(self):
        value = artifact("direct")
        value["rank_records"] = []
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "direct.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(acceptance.bm.ValidationError):
                acceptance.load_acceptance_artifact(path, "direct")

    def test_cli_orders_build_and_private_mpi_loader_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            output_dir = root / "out"
            python_path = root / "python"
            library_path = root / "lib"
            mpi_prefix = root / "private-mpi"
            calls = []

            def fake_run(command, env, timeout):
                calls.append((command, env, timeout))
                route = command[command.index("--worker") + 1]
                output = pathlib.Path(command[command.index("--output") + 1])
                output.write_text(json.dumps(artifact(route)), encoding="utf-8")

            with mock.patch.dict(
                acceptance.os.environ,
                {"LD_LIBRARY_PATH": str(root / "wrong-mpi" / "lib")},
            ), mock.patch.object(acceptance, "_run", side_effect=fake_run):
                status = acceptance.main(
                    [
                        "--output-dir",
                        str(output_dir),
                        "--prefix",
                        str(mpi_prefix),
                        "--pythonpath",
                        str(python_path),
                        "--library-path",
                        str(library_path),
                    ]
                )
            self.assertEqual(status, 0)
            self.assertEqual(len(calls), 3)
            for _command, environment, _timeout in calls:
                self.assertEqual(environment["PYTHONPATH"], str(python_path.resolve()))
                self.assertEqual(
                    environment["LD_LIBRARY_PATH"],
                    os.pathsep.join(
                        (str(library_path.resolve()), str(mpi_prefix.resolve() / "lib"))
                    ),
                )

    def test_cli_rejects_implicit_import_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(acceptance, "_run") as run:
                with contextlib.redirect_stderr(io.StringIO()):
                    status = acceptance.main(["--output-dir", directory])
            self.assertEqual(status, 2)
            run.assert_not_called()

    def test_staged_only_cli_uses_explicit_tools_and_rocr_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            calls = []

            def fake_run(command, env, timeout):
                calls.append((command, env, timeout))
                route = command[command.index("--worker") + 1]
                output = pathlib.Path(command[command.index("--output") + 1])
                output.write_text(json.dumps(artifact(route)), encoding="utf-8")

            python = root / "branch-python"
            mpiexec = root / "branch-mpiexec"
            mpi_prefix = root / "private-mpi"
            rocm_smi = root / "rocm-smi"
            with mock.patch.object(acceptance, "_run", side_effect=fake_run):
                status = acceptance.main(
                    [
                        "--output-dir",
                        str(root / "out"),
                        "--routes",
                        "cpu,staged",
                        "--prefix",
                        str(mpi_prefix),
                        "--python",
                        str(python),
                        "--mpiexec",
                        str(mpiexec),
                        "--visible-devices",
                        "0,1",
                        "--rocm-smi",
                        str(rocm_smi),
                        "--pythonpath",
                        str(root / "python"),
                        "--library-path",
                        str(root / "lib"),
                    ]
                )
            self.assertEqual(status, 0)
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0][0][0], str(python.resolve()))
            self.assertEqual(calls[1][0][0], str(mpiexec.resolve()))
            self.assertEqual(
                calls[1][1]["LD_LIBRARY_PATH"],
                os.pathsep.join(
                    (str((root / "lib").resolve()), str(mpi_prefix.resolve() / "lib"))
                ),
            )
            self.assertEqual(calls[1][1]["ROCR_VISIBLE_DEVICES"], "0,1")
            self.assertEqual(calls[1][1]["MEEP_GPU_AWARE_MPI"], "no")
            self.assertNotIn("CUDA_VISIBLE_DEVICES", calls[1][1])
            self.assertNotIn("HIP_VISIBLE_DEVICES", calls[1][1])
            self.assertNotIn("OMPI_MCA_opal_cuda_support", calls[1][1])
            self.assertFalse((root / "out" / "direct.json").exists())


if __name__ == "__main__":
    unittest.main()
