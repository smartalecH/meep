#!/usr/bin/env python3

import copy
import json
import pathlib
import tempfile
import unittest
from unittest import mock

import run_mpi_benchmark as runner


HASH = "a" * 64
REQUESTED = {"backend": "nvidia", "precision": "native", "route": "direct",
             "overlap": "required", "graph": "required", "ranks": 2}


def rank_record(rank, samples=(2.0, 4.0)):
    normalized_uuid = f"{rank + 1:032x}"
    runtime = {
        "requested_backend": "nvidia", "resolved_backend": "nvidia",
        "requested_precision": "native", "resolved_precision": "native",
        "requested_transport": "direct", "resolved_transport": "direct",
        "requested_overlap": "required", "resolved_overlap": "overlap",
        "captured_requested_transport": "direct", "captured_overlap_policy": "required",
        "requested_graph": "required", "resolved_graph": "graph",
        "mpi_provider": "Open MPI/UCX", "counter_scope": "rank_local_current_epoch",
        "backend_counter_scope": "rank_local_backend_lifetime",
        "memory_gauge_scope": "rank_local_process_lifetime",
        "setup_counter_scope": "rank_local_current_backend_state",
        "transport_timing_scope": "rank_local_current_transport_epoch_host_elapsed",
        "allocation_counter_scope": "rank_local_process_lifetime",
        "communicator_rank": rank,
        "communicator_size": 2, "communicator_generation": 7,
        "captured_provider_signature": 9,
        "mpi_query_available": True, "mpi_cuda_aware": True,
        "device_owner": True, "device_id": rank, "device_uuid": normalized_uuid,
        "captured_transport_epoch_active": True, "captured_transport_epoch_fresh": True,
        "graph_enabled": True, "graph_valid": True,
    }
    runtime.update({name: rank + 1 for name in runner.COUNTER_AGGREGATIONS})
    runtime.update({name: 100 + rank for name in runner.GAUGE_AGGREGATIONS})
    runtime["transport_pinned_bytes"] = 0
    runtime["device_to_host_calls"] = runtime["device_to_host_bytes"] = 0
    runtime["host_to_device_calls"] = runtime["host_to_device_bytes"] = 0
    return {"rank": rank, "communicator_size": 2, "communicator_generation": 7,
            "manifest_sha256": HASH, "launch_sha256": "b" * 64,
            "provenance_sha256": "c" * 64, "hostname": "node", "local_rank": rank,
            "role": "owner", "device": {"visible_device": rank, "process_device_id": rank,
                "physical_selector": str(rank), "uuid": f"GPU-{normalized_uuid}", "name": "GB200",
                "memory_bytes": 1000, "sm_clock_hz": 1.0, "memory_clock_hz": 1.0,
                "driver_version": "1"}, "runtime": runtime,
            "module_paths": {"meep": "/tmp/meep/__init__.py", "extension": "/tmp/_meep.so"},
            "module_sha256": {"meep": "e" * 64, "extension": "f" * 64},
            "cpu_affinity": [rank],
            "repetitions": [{"initialization_seconds": 0.1, "warmup_seconds": 0.2,
                             "steady_seconds": value + rank, "monitor_seconds": 0.3,
                             "steps": 100, "dt_meep": 0.1, "grid_shape": [2, 3, 4],
                             "stop_reason": "fixed_steps",
                             "counter_start": {name: 0 for name in runner.COUNTER_AGGREGATIONS},
                             "counter_end": {name: (2 * (rank + 1) if name == "direct_bytes" else
                                                    0 if name in {"device_to_host_calls", "device_to_host_bytes", "host_to_device_calls", "host_to_device_bytes"}
                                                    else rank + 1) for name in runner.COUNTER_AGGREGATIONS},
                             "counter_deltas": {name: (2 * (rank + 1) if name == "direct_bytes" else
                                                         0 if name in {"device_to_host_calls", "device_to_host_bytes", "host_to_device_calls", "host_to_device_bytes"}
                                                         else rank + 1) for name in runner.COUNTER_AGGREGATIONS},
                             "monitors": []} for value in samples]}


def result_document():
    reconciled = runner.reconcile_rank_records([rank_record(0), rank_record(1)], HASH, REQUESTED)
    result = {"schema_version": 2, "kind": "paper_2506_16665_multi_rank_benchmark",
            "generated_at_utc": "2026-09-02T00:00:00+00:00",
            "run_manifest": {"path": "/tmp/manifest.json", "sha256": HASH, "case_id": "coupler"},
            "physics_reference": {"path": "/tmp/reference.json", "sha256": "d" * 64},
            "status": {"succeeded": True, "errors": []},
            "launch": {"argv": [str(runner.MPIEXEC), "-np", "2", str(runner.PYTHON)], "cwd": "/tmp",
                       "environment": {"OMPI_MCA_opal_cuda_support": "true", "OMPI_MCA_pml": "ucx",
                           "UCX_TLS": runner.UCX_TLS, "MEEP_GPU_AWARE_MPI": "yes",
                           "MEEP_NVIDIA_MPI_OVERLAP": "required", "MEEP_NVIDIA_GRAPH_MODE": "required",
                           "MEEP_PRECISION": "native", "CUDA_VISIBLE_DEVICES": "0,1",
                           "PYTHONPATH": "/tmp/python", "LD_LIBRARY_PATH": "/tmp/lib"}, "timeout_seconds": 30,
                       "mpiexec": str(runner.MPIEXEC), "python": str(runner.PYTHON),
                       "mpi_version": "Open MPI", "ucx_version": "UCX"},
            "provenance": {"meep_source": "/tmp/meep", "meep_commit": "1" * 40,
                           "meep_dirty": False, "runner_commit": "2" * 40,
                           "runner_dirty": True, "build_directory": "/tmp/build",
                           "cuda_toolkit": "CUDA 13", "configure_arguments": "--with-cuda",
                           "compiler": "mpic++ 15", "compiler_flags": {"CXXFLAGS": "-O3"},
                           "executable_sha256": {"mpiexec": "3" * 64,
                                                 "python": "4" * 64}},
            "requested_execution": REQUESTED,
            "physics_observables": [{"name": "x", "monitor": {}, "unit": "1", "evaluation": "linear_center_wavelength_mode_power_ratio",
                                     "value": 1.0, "values_by_repetition": [1.0, 1.0],
                                     "reference_value": 1.0, "absolute_tolerance": 0.0,
                                     "relative_tolerance": 0.0, "passed": True}],
            **reconciled}
    launch_hash = __import__("hashlib").sha256(json.dumps(result["launch"], sort_keys=True).encode()).hexdigest()
    provenance_hash = __import__("hashlib").sha256(json.dumps(result["provenance"], sort_keys=True).encode()).hexdigest()
    for record in result["rank_records"]:
        record["launch_sha256"] = launch_hash
        record["provenance_sha256"] = provenance_hash
    return result


class LaunchEnvironmentTest(unittest.TestCase):
    def test_provider_positive_ucx_is_forced(self):
        env = runner.launch_environment({"OMPI_MCA_btl": "stale"}, "direct", "required", "required", "native")
        self.assertEqual(env["OMPI_MCA_opal_cuda_support"], "true")
        self.assertEqual(env["OMPI_MCA_pml"], "ucx")
        self.assertEqual(env["UCX_TLS"], "self,sm,cuda_copy,cuda_ipc")
        self.assertEqual(env["MEEP_GPU_AWARE_MPI"], "yes")
        self.assertNotIn("OMPI_MCA_btl", env)

    def test_nvidia_smi_uuid_is_normalized_to_runtime_form(self):
        self.assertEqual(
            runner.normalize_gpu_uuid("GPU-51B2C4CE-1234-5678-9ABC-DEF012345678"),
            "51b2c4ce123456789abcdef012345678",
        )

    def test_python_purelib_is_queried_from_selected_interpreter(self):
        path = runner.python_purelib(runner.PYTHON)
        self.assertTrue(path.is_absolute())
        self.assertIn("site-packages", str(path))


class CollectiveStopTest(unittest.TestCase):
    def test_fixed_step_uses_exact_count(self):
        class Fields:
            t = 4
            def advance(self, steps): self.t += steps
        class Simulation:
            fields = Fields()
        steps, reason = runner.advance_with_collective_stop(None, Simulation(),
                                                            {"kind": "fixed_steps", "steps": 7})
        self.assertEqual((steps, reason), (7, "fixed_steps"))

    def test_total_energy_mode_is_rejected_instead_of_risking_rank_divergence(self):
        class Fields: t = 0
        class Simulation: fields = Fields()
        with self.assertRaisesRegex(runner.RunnerError, "total_field_energy"):
            runner.advance_with_collective_stop(None, Simulation(),
                {"kind": "field_energy_decay", "observable": "total_field_energy"})

    def test_timed_interval_is_bracketed_by_active_communicator_barriers(self):
        events = []

        class MP:
            @staticmethod
            def all_wait():
                events.append("barrier")

        class Fields:
            t = 0

            def advance(self, steps):
                events.append("advance")
                self.t += steps

        class Simulation:
            fields = Fields()

            @staticmethod
            def get_execution_runtime_report():
                events.append("report")
                return {"counter": len(events)}

        with mock.patch.object(runner.time, "perf_counter", side_effect=(10.0, 13.5)):
            steps, reason, elapsed, before, after = runner.timed_advance_with_collective_stop(
                MP(), Simulation(), {"kind": "fixed_steps", "steps": 7}
            )
        self.assertEqual(events, ["barrier", "report", "advance", "barrier", "report"])
        self.assertEqual((steps, reason, elapsed), (7, "fixed_steps", 3.5))
        self.assertEqual((before["counter"], after["counter"]), (2, 5))


class ReconciliationTest(unittest.TestCase):
    def test_global_timing_and_declared_counter_aggregation(self):
        result = runner.reconcile_rank_records([rank_record(0), rank_record(1)], HASH, REQUESTED)
        self.assertEqual(result["timing"]["critical_path_samples_seconds"], [3.0, 5.0])
        self.assertEqual(result["timing"]["median_seconds"], 4.0)
        self.assertEqual(result["timing"]["repetitions"][0]["rank_median_seconds"], 2.5)
        self.assertEqual(result["timing"]["repetitions"][0]["imbalance_ratio"], 1.2)
        counters = {item["name"]: item for item in result["counter_aggregates"]}
        self.assertEqual(counters["direct_bytes"]["aggregation"], "sum")
        self.assertEqual(counters["direct_bytes"]["value"], 12)

    def test_manifest_disagreement_is_rejected(self):
        records = [rank_record(0), rank_record(1)]
        records[1]["manifest_sha256"] = "b" * 64
        with self.assertRaisesRegex(runner.RunnerError, "manifest hash"):
            runner.reconcile_rank_records(records, HASH, REQUESTED)

    def test_communicator_disagreement_is_rejected(self):
        records = [rank_record(0), rank_record(1)]
        records[1]["communicator_generation"] = 8
        with self.assertRaisesRegex(runner.RunnerError, "communicator identity"):
            runner.reconcile_rank_records(records, HASH, REQUESTED)

    def test_loaded_module_hash_disagreement_is_rejected(self):
        records = [rank_record(0), rank_record(1)]
        records[1]["module_sha256"]["extension"] = "0" * 64
        with self.assertRaisesRegex(runner.RunnerError, "module hashes"):
            runner.reconcile_rank_records(records, HASH, REQUESTED)

    def test_direct_and_required_modes_cannot_downgrade(self):
        records = [rank_record(0), rank_record(1)]
        for record in records:
            record["runtime"]["resolved_transport"] = "staged"
        with self.assertRaisesRegex(runner.RunnerError, "route was downgraded"):
            runner.reconcile_rank_records(records, HASH, REQUESTED)

        records = [rank_record(0), rank_record(1)]
        for record in records:
            record["runtime"]["resolved_graph"] = "eager"
        with self.assertRaisesRegex(runner.RunnerError, "required graph"):
            runner.reconcile_rank_records(records, HASH, REQUESTED)


class PublicationTest(unittest.TestCase):
    @staticmethod
    def authenticated_context(result):
        manifest = {
            "execution": {
                "requested": {
                    "backend": "nvidia", "precision": "native",
                    "mpi_transport": "direct", "overlap": "required",
                    "graph": "required", "ranks": 2,
                },
                "measured_repetitions": 2,
            },
            "validation_policy": {"required_observables": [{
                "name": "x", "monitor": {"name": "out"}, "unit": "1",
            }]},
        }
        references = {"x": {"monitor": "out", "unit": "1", "value": 1.0}}
        derived = copy.deepcopy(result["physics_observables"][0])
        del derived["values_by_repetition"]
        return manifest, references, derived

    def test_schema_validated_atomic_publication(self):
        result = result_document()
        runner.validate_result(result)
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "result.json"
            runner.atomic_publish(result, path)
            self.assertEqual(json.loads(path.read_text()), result)
            self.assertEqual(list(path.parent.glob(f".{path.name}.*")), [])

    def test_invalid_result_never_replaces_existing_artifact(self):
        result = result_document()
        invalid = copy.deepcopy(result)
        invalid["schema_version"] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "result.json"
            path.write_text('{"old": true}\n')
            with self.assertRaises(runner.bm.ValidationError):
                runner.atomic_publish(invalid, path)
            self.assertEqual(json.loads(path.read_text()), {"old": True})

    def test_direct_result_requires_positive_provider_query(self):
        result = result_document()
        for record in result["rank_records"]:
            record["runtime"]["mpi_cuda_aware"] = False
        with self.assertRaisesRegex(runner.RunnerError, "positive CUDA-aware"):
            runner.validate_result(result)

    def test_provider_environment_tampering_is_rejected(self):
        result = result_document()
        result["launch"]["environment"]["OMPI_MCA_pml"] = "ob1"
        with self.assertRaisesRegex(runner.RunnerError, "authoritative UCX"):
            runner.validate_result(result)

    def test_public_aggregate_tampering_is_rejected(self):
        result = result_document()
        result["timing"]["maximum_seconds"] += 1
        with self.assertRaisesRegex(runner.RunnerError, "published timing"):
            runner.validate_result(result)

    def test_simulation_and_performance_tampering_are_rejected(self):
        for section, key in (("simulation", "steps"),
                             ("performance", "grid_timesteps_per_second")):
            with self.subTest(section=section):
                result = result_document()
                result[section][key] += 1
                with self.assertRaisesRegex(runner.RunnerError, f"published {section}"):
                    runner.validate_result(result)

    def test_manifest_requested_execution_and_repetitions_are_bound(self):
        result = result_document()
        manifest, references, derived = self.authenticated_context(result)
        with mock.patch.object(runner, "derive_observables", return_value=[derived]):
            runner.validate_result(result, manifest=manifest, references=references)
            changed = copy.deepcopy(manifest)
            changed["execution"]["requested"]["backend"] = "cpu"
            with self.assertRaisesRegex(runner.RunnerError, "requested execution"):
                runner.validate_result(result, manifest=changed, references=references)
            changed = copy.deepcopy(manifest)
            changed["execution"]["measured_repetitions"] = 3
            with self.assertRaisesRegex(runner.RunnerError, "repetition count"):
                runner.validate_result(result, manifest=changed, references=references)

    def test_authenticated_reference_and_derived_observables_are_bound(self):
        result = result_document()
        manifest, references, derived = self.authenticated_context(result)
        with mock.patch.object(runner, "derive_observables", return_value=[derived]):
            changed_references = copy.deepcopy(references)
            changed_references["x"]["monitor"] = "other"
            with self.assertRaisesRegex(runner.RunnerError, "monitor/unit"):
                runner.validate_result(result, manifest=manifest,
                                       references=changed_references)
            changed_result = copy.deepcopy(result)
            changed_result["physics_observables"][0]["value"] = 2.0
            with self.assertRaisesRegex(runner.RunnerError, "not derived"):
                runner.validate_result(changed_result, manifest=manifest,
                                       references=references)

    def test_per_rank_route_accounting_cannot_cancel_across_ranks(self):
        result = result_document()
        first = result["rank_records"][0]["repetitions"][0]["counter_deltas"]
        second = result["rank_records"][1]["repetitions"][0]["counter_deltas"]
        first["direct_bytes"] = 0
        second["direct_bytes"] += 2
        for record in result["rank_records"]:
            repetition = record["repetitions"][0]
            repetition["counter_end"] = {
                name: repetition["counter_start"][name] + value
                for name, value in repetition["counter_deltas"].items()
            }
        reconciled = runner.reconcile_rank_records(
            result["rank_records"], HASH, result["requested_execution"]
        )
        result["counter_aggregates"] = reconciled["counter_aggregates"]
        with self.assertRaisesRegex(runner.RunnerError, "rank 0 direct"):
            runner.validate_result(result)

    def test_resolved_graph_requires_valid_positive_execution_per_rank(self):
        for mutate in (
            lambda runtime, counters: runtime.update(graph_enabled=False),
            lambda runtime, counters: runtime.update(graph_valid=False),
            lambda runtime, counters: counters.update(graph_launch_count=0),
        ):
            with self.subTest(mutate=mutate):
                result = result_document()
                record = result["rank_records"][0]
                for repetition in record["repetitions"]:
                    mutate(record["runtime"], repetition["counter_deltas"])
                    repetition["counter_end"] = {
                        name: repetition["counter_start"][name] + value
                        for name, value in repetition["counter_deltas"].items()
                    }
                reconciled = runner.reconcile_rank_records(
                    result["rank_records"], HASH, result["requested_execution"]
                )
                result["counter_aggregates"] = reconciled["counter_aggregates"]
                with self.assertRaisesRegex(runner.RunnerError, "valid graph"):
                    runner.validate_result(result)


if __name__ == "__main__":
    unittest.main()
