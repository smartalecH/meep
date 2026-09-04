#!/usr/bin/env python3

import copy
import hashlib
import json
import math
import pathlib
import tempfile
import unittest
from unittest import mock

import run_mpi_benchmark as mpi_runner
import run_self_contained_benchmark as runner


def source_provenance():
    source = pathlib.Path(runner.__file__).resolve().parents[3]
    return {"source": str(source), **runner.benchmark_source_provenance(source)}


def bind_records(value):
    launch_hash = hashlib.sha256(
        json.dumps(value["launch"], sort_keys=True).encode()
    ).hexdigest()
    provenance_hash = hashlib.sha256(
        json.dumps(value["provenance"], sort_keys=True).encode()
    ).hexdigest()
    for record in value["rank_records"]:
        record["launch_sha256"] = launch_hash
        record["provenance_sha256"] = provenance_hash


def cpu_result():
    launch = {
        "argv": ["mpirun", "-np", "1", "python", "runner.py"],
        "cwd": "/tmp",
        "environment": {},
        "timeout_seconds": 30,
        "mpiexec": "/mpi/bin/mpirun",
        "python": "/venv/bin/python",
        "backend": "cpu",
        "accelerator": "hip",
        "route": "host",
        "ranks": 1,
        "omp_threads": 1,
        "visible_devices": [],
        "map_by": "ppr:1:node",
        "rank_by": None,
        "bind_to": "core",
        "mpi_version": "Open MPI 5.0.10",
        "ucx_version": "UCX 1.19.1",
    }
    provenance = source_provenance()
    launch_hash = hashlib.sha256(
        json.dumps(launch, sort_keys=True).encode()
    ).hexdigest()
    provenance_hash = hashlib.sha256(
        json.dumps(provenance, sort_keys=True).encode()
    ).hexdigest()
    counters = {name: 0 for name in mpi_runner.COUNTER_AGGREGATIONS}
    repetitions = [
        {
            "initialization_seconds": 0.1 if index == 0 else 0.0,
            "warmup_seconds": 0.2 if index == 0 else 0.0,
            "steady_seconds": 1.0 + index / 10,
            "steps": runner.CASE["measured_steps"],
            "dt_meep": 0.1,
            "grid_shape": [2, 3, 4],
            "counter_start": dict(counters),
            "counter_end": dict(counters),
            "counter_deltas": dict(counters),
            "samples": [[float(index), 0.0] for _ in runner.CASE["sample_points"]],
        }
        for index in range(runner.CASE["measured_repetitions"])
    ]
    critical = [rep["steady_seconds"] for rep in repetitions]
    return {
        "schema_version": 1,
        "kind": "meep_self_contained_fixed_step_benchmark",
        "generated_at_utc": "2026-09-03T00:00:00+00:00",
        "case": runner.CASE,
        "case_sha256": runner.canonical_hash(runner.CASE),
        "launch": launch,
        "provenance": provenance,
        "rank_records": [
            {
                "rank": 0,
                "communicator_size": 1,
                "local_rank": 0,
                "hostname": "node",
                "role": "cpu",
                "device": None,
                "cpu_affinity": [0],
                "cpu_numa_nodes": [0],
                "runtime": {},
                "repetitions": repetitions,
                "module_paths": {},
                "module_sha256": {},
                "launch_sha256": launch_hash,
                "provenance_sha256": provenance_hash,
            }
        ],
        "timing": {
            "critical_path_samples_seconds": critical,
            "minimum_seconds": min(critical),
            "median_seconds": 1.2,
            "maximum_seconds": max(critical),
            "grid_timesteps_per_second": 2000.0,
        },
        "validation": {
            "fixed_step_protocol": True,
            "steady_state_clean": True,
            "transport_accounting": True,
            "topology_attested": True,
            "cpu_reference": None,
            "cpu_tolerance_passed": None,
            "peer_route": None,
            "peer_route_bitwise": None,
        },
    }


def gpu_result(route="staged"):
    value = cpu_result()
    value["launch"].update(
        {
            "backend": "nvidia",
            "accelerator": "hip",
            "route": route,
            "ranks": 2,
            "visible_devices": ["1", "5"],
            "map_by": "ppr:1:package",
            "rank_by": "fill",
        }
    )
    records = []
    for rank, (uuid, bdf, physical_id, numa) in enumerate(
        (
            ("1" * 32, "0000:08:00.0", "physical0", 0),
            ("2" * 32, "0000:88:00.0", "physical1", 1),
        )
    ):
        record = copy.deepcopy(value["rank_records"][0])
        record.update(
            {
                "rank": rank,
                "communicator_size": 2,
                "local_rank": rank,
                "role": "owner",
                "device": {
                    "visible_devices": ["1", "5"],
                    "physical_selector": ("1", "5")[rank],
                    "uuid": uuid,
                    "pci_bus_id": bdf,
                    "physical_unique_id": physical_id,
                    "numa_node": numa,
                },
                "cpu_affinity": [rank],
                "cpu_numa_nodes": [numa],
                "runtime": {
                    "device_owner": True,
                    "resolved_backend": "nvidia",
                    "device_uuid": uuid,
                    "resolved_transport": route,
                    "mpi_query_available": True,
                    "mpi_cuda_aware": True,
                    "transport_pinned_bytes": 1 if route == "staged" else 0,
                },
            }
        )
        for repetition in record["repetitions"]:
            deltas = {name: 0 for name in mpi_runner.COUNTER_AGGREGATIONS}
            deltas.update(
                {
                    "messages_sent": 1,
                    "messages_received": 1,
                    "bytes_sent": 8,
                    "bytes_received": 8,
                }
            )
            if route == "staged":
                deltas.update(
                    {
                        "device_to_host_calls": 1,
                        "device_to_host_bytes": 8,
                        "host_to_device_calls": 1,
                        "host_to_device_bytes": 8,
                    }
                )
            else:
                deltas["direct_bytes"] = 16
            repetition["counter_start"] = {
                name: 0 for name in mpi_runner.COUNTER_AGGREGATIONS
            }
            repetition["counter_end"] = dict(deltas)
            repetition["counter_deltas"] = dict(deltas)
        records.append(record)
    value["rank_records"] = records
    value["validation"].update(
        {
            "cpu_reference": None,
            "cpu_tolerance_passed": None,
            "peer_route": None,
            "peer_route_bitwise": None,
        }
    )
    bind_records(value)
    return value


def write_result(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


class ValidationTest(unittest.TestCase):
    def test_valid_cpu_artifact_retains_five_raw_windows(self):
        value = cpu_result()
        runner.validate_result(value)
        self.assertEqual(len(value["timing"]["critical_path_samples_seconds"]), 5)

    def test_steady_fallback_is_rejected(self):
        value = copy.deepcopy(cpu_result())
        value["rank_records"][0]["repetitions"][2]["counter_deltas"][
            "host_fallback_count"
        ] = 1
        value["rank_records"][0]["repetitions"][2]["counter_end"][
            "host_fallback_count"
        ] = 1
        with self.assertRaisesRegex(runner.SelfContainedError, "fell back"):
            runner.validate_result(value)

    def test_counter_delta_must_match_snapshots(self):
        value = cpu_result()
        value["rank_records"][0]["repetitions"][0]["counter_end"][
            "steady_allocation_count"
        ] = 99
        with self.assertRaisesRegex(
            runner.SelfContainedError, "disagrees with its snapshots"
        ):
            runner.validate_result(value)

    def test_all_published_timing_fields_are_recomputed(self):
        for name, forged in (
            ("critical_path_samples_seconds", [999.0] * 5),
            ("minimum_seconds", 998.0),
            ("median_seconds", 999.0),
            ("maximum_seconds", 1000.0),
            ("grid_timesteps_per_second", 1.0),
        ):
            with self.subTest(name=name):
                value = cpu_result()
                value["timing"][name] = forged
                with self.assertRaisesRegex(runner.SelfContainedError, "timing/rate"):
                    runner.validate_result(value)

    def test_nonfinite_raw_field_sample_is_rejected(self):
        value = cpu_result()
        value["rank_records"][0]["repetitions"][4]["samples"][0][0] = math.nan
        with self.assertRaisesRegex(runner.SelfContainedError, "field sample"):
            runner.validate_result(value)

    def test_every_measured_window_must_match_cpu_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            reference_path = pathlib.Path(directory) / "cpu.json"
            write_result(reference_path, cpu_result())
            value = gpu_result()
            for record in value["rank_records"]:
                record["repetitions"][4]["samples"][0] = [999.0, 0.0]
            with self.assertRaisesRegex(runner.SelfContainedError, "measured window"):
                runner._compare_reference(value, reference_path)

    def test_each_physical_identity_field_must_be_unique(self):
        for name in ("uuid", "pci_bus_id", "physical_unique_id"):
            with self.subTest(name=name):
                value = gpu_result()
                value["rank_records"][1]["device"][name] = value["rank_records"][0][
                    "device"
                ][name]
                if name == "uuid":
                    value["rank_records"][1]["runtime"]["device_uuid"] = value[
                        "rank_records"
                    ][0]["runtime"]["device_uuid"]
                with self.assertRaisesRegex(
                    runner.SelfContainedError, "duplicate physical"
                ):
                    runner.validate_result(value, allow_unbound_references=True)

    def test_cpu_reference_flags_and_content_hash_are_authenticated(self):
        with tempfile.TemporaryDirectory() as directory:
            reference_path = pathlib.Path(directory) / "cpu.json"
            write_result(reference_path, cpu_result())
            valid_reference = {
                "path": str(reference_path),
                "sha256": runner.bm.sha256_file(reference_path),
            }
            for reference, passed, message in (
                (valid_reference, False, "validated CPU reference"),
                ({**valid_reference, "sha256": "0" * 64}, True, "hash is invalid"),
                (
                    {**valid_reference, "sha256": "1" * 64},
                    True,
                    "does not match its content",
                ),
            ):
                with self.subTest(message=message):
                    value = gpu_result()
                    value["validation"]["cpu_reference"] = reference
                    value["validation"]["cpu_tolerance_passed"] = passed
                    with self.assertRaisesRegex(runner.SelfContainedError, message):
                        runner.validate_result(value)

    def test_peer_flags_and_content_hash_are_authenticated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            cpu_path = root / "cpu.json"
            write_result(cpu_path, cpu_result())
            cpu_reference = {
                "path": str(cpu_path),
                "sha256": runner.bm.sha256_file(cpu_path),
            }
            staged = gpu_result()
            staged["validation"]["cpu_reference"] = cpu_reference
            staged["validation"]["cpu_tolerance_passed"] = True
            staged_path = root / "staged.json"
            write_result(staged_path, staged)
            valid_peer = {
                "path": str(staged_path),
                "sha256": runner.bm.sha256_file(staged_path),
            }
            for peer, passed, message in (
                (valid_peer, False, "bitwise staged peer"),
                ({**valid_peer, "sha256": "0" * 64}, True, "hash is invalid"),
                (
                    {**valid_peer, "sha256": "1" * 64},
                    True,
                    "does not match its content",
                ),
            ):
                with self.subTest(message=message):
                    value = gpu_result(route="direct")
                    value["validation"].update(
                        {
                            "cpu_reference": cpu_reference,
                            "cpu_tolerance_passed": True,
                            "peer_route": peer,
                            "peer_route_bitwise": passed,
                        }
                    )
                    with self.assertRaisesRegex(runner.SelfContainedError, message):
                        runner.validate_result(value)

    def test_valid_staged_and_direct_reference_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            cpu_path = root / "cpu.json"
            write_result(cpu_path, cpu_result())
            cpu_reference = {
                "path": str(cpu_path),
                "sha256": runner.bm.sha256_file(cpu_path),
            }
            staged = gpu_result()
            staged["validation"].update(
                {"cpu_reference": cpu_reference, "cpu_tolerance_passed": True}
            )
            runner.validate_result(staged)
            staged_path = root / "staged.json"
            write_result(staged_path, staged)
            direct = gpu_result(route="direct")
            direct["validation"].update(
                {
                    "cpu_reference": cpu_reference,
                    "cpu_tolerance_passed": True,
                    "peer_route": {
                        "path": str(staged_path),
                        "sha256": runner.bm.sha256_file(staged_path),
                    },
                    "peer_route_bitwise": True,
                }
            )
            runner.validate_result(direct)

    def test_runner_and_dirty_source_state_are_authenticated(self):
        for field, forged, message in (
            (
                "runner_source",
                {
                    "path": str(pathlib.Path(runner.__file__).resolve()),
                    "sha256": "0" * 64,
                },
                "runner source hash",
            ),
            ("source_diff_sha256", "0" * 64, "source-tree state"),
            (
                "source_clean",
                not source_provenance()["source_clean"],
                "source-tree state",
            ),
        ):
            with self.subTest(field=field):
                value = cpu_result()
                value["provenance"][field] = forged
                with self.assertRaisesRegex(runner.SelfContainedError, message):
                    runner.validate_result(value)

    def test_clean_source_state_uses_no_dirty_diff_hash(self):
        value = cpu_result()
        clean = {"commit": "a" * 40, "source_clean": True, "source_diff_sha256": None}
        value["provenance"].update({**clean, "dirty": False})
        bind_records(value)
        with mock.patch.object(runner, "source_tree_state", return_value=clean):
            runner.validate_result(value)


if __name__ == "__main__":
    unittest.main()
