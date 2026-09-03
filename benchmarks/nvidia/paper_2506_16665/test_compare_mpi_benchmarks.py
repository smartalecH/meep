#!/usr/bin/env python3

import copy
import hashlib
import json
import pathlib
import unittest

import compare_mpi_benchmarks as compare
import run_mpi_benchmark as runner
from test_run_mpi_benchmark import HASH, REQUESTED, result_document


def route_result(route):
    result = result_document()
    requested = dict(REQUESTED); requested["route"] = route
    result["requested_execution"] = requested
    result["launch"]["environment"]["MEEP_GPU_AWARE_MPI"] = "no" if route == "staged" else "yes"
    for rank in result["rank_records"]:
        rank["runtime"]["requested_transport"] = route
        rank["runtime"]["captured_requested_transport"] = route
        rank["runtime"]["resolved_transport"] = route
        for repetition in rank["repetitions"]:
            counters = repetition["counter_deltas"]
            counters["bytes_sent"] = counters["bytes_received"] = 10
            counters["messages_sent"] = counters["messages_received"] = 1
            if route == "staged":
                counters["direct_bytes"] = 0
                counters["device_to_host_bytes"] = counters["host_to_device_bytes"] = 10
                counters["device_to_host_calls"] = counters["host_to_device_calls"] = 1
                rank["runtime"]["transport_pinned_bytes"] = 64
            else:
                counters["direct_bytes"] = 20
                counters["device_to_host_bytes"] = counters["host_to_device_bytes"] = 0
                counters["device_to_host_calls"] = counters["host_to_device_calls"] = 0
                rank["runtime"]["transport_pinned_bytes"] = 0
            repetition["counter_end"] = {
                name: repetition["counter_start"][name] + value for name, value in counters.items()
            }
    launch_hash = hashlib.sha256(json.dumps(result["launch"], sort_keys=True).encode()).hexdigest()
    for rank in result["rank_records"]:
        rank["launch_sha256"] = launch_hash
    reconciled = runner.reconcile_rank_records(result["rank_records"], HASH, requested)
    for key in ("resolved_execution", "communicator", "rank_records", "timing",
                "counter_aggregates", "simulation", "performance"):
        result[key] = reconciled[key]
    return result


def refresh_record_hashes(result):
    launch_hash = hashlib.sha256(
        json.dumps(result["launch"], sort_keys=True).encode()
    ).hexdigest()
    provenance_hash = hashlib.sha256(
        json.dumps(result["provenance"], sort_keys=True).encode()
    ).hexdigest()
    for rank in result["rank_records"]:
        rank["launch_sha256"] = launch_hash
        rank["provenance_sha256"] = provenance_hash


class ObservableTest(unittest.TestCase):
    def test_center_wavelength_ratio_and_loss(self):
        monitor = lambda name, values: {"name": name, "forward_mode_power": values,
            "backward_mode_power": [0, 0], "raw_dft_flux": values, "port": "x", "mode_band": 1}
        manifest = {"excitation": {"center_wavelength_um": 1.55,
                    "monitor_sampling": {"wavelengths_um": [1.54, 1.56]}},
                    "case": {"monitors": [{"name": "input_incident"}, {"name": "out",
                    "normalization": {"kind": "mode_power_ratio"}}]},
                    "validation_policy": {"required_observables": [
                    {"name": "target_mode_transmission", "monitor": {"name": "out"}, "unit": "1",
                     "evaluation": "linear_center_wavelength_mode_power_ratio", "absolute_tolerance": 0.0, "relative_tolerance": 0.0},
                    {"name": "excess_loss_db", "monitor": {"name": "out"}, "unit": "dB",
                     "evaluation": "negative_ten_log10_linear_center_wavelength_mode_power_ratio", "absolute_tolerance": 0.0, "relative_tolerance": 0.0}]}}
        refs = {"target_mode_transmission": {"value": 0.5},
                "excess_loss_db": {"value": 3.010299956639812}}
        values = runner.derive_observables(manifest,
            [monitor("input_incident", [2.0, 4.0]), monitor("out", [1.0, 2.0])], refs)
        self.assertTrue(all(item["passed"] for item in values))


class RouteComparisonTest(unittest.TestCase):
    def test_staged_and_direct_match(self):
        staged, direct = route_result("staged"), route_result("direct")
        value = compare.compare(staged, direct, pathlib.Path("staged.json"),
                                pathlib.Path("direct.json"), "e" * 64, "f" * 64)
        compare.validate(value)
        self.assertTrue(value["bitwise_monitor_match"])

    def test_monitor_difference_rejected(self):
        staged, direct = route_result("staged"), route_result("direct")
        for rank in direct["rank_records"]:
            rank["repetitions"][0]["monitors"] = [{"different": True}]
        with self.assertRaisesRegex(runner.RunnerError, "monitor arrays"):
            compare.compare(staged, direct, pathlib.Path("s"), pathlib.Path("d"),
                            "e" * 64, "f" * 64)

    def test_full_provenance_difference_rejected(self):
        staged, direct = route_result("staged"), route_result("direct")
        direct["provenance"]["cuda_toolkit"] = "different CUDA"
        refresh_record_hashes(direct)
        with self.assertRaisesRegex(runner.RunnerError, "build provenance"):
            compare.compare(staged, direct, pathlib.Path("s"), pathlib.Path("d"),
                            "e" * 64, "f" * 64)

    def test_resolved_nonroute_mode_difference_rejected(self):
        staged, direct = route_result("staged"), route_result("direct")
        for result in (staged, direct):
            result["requested_execution"]["graph"] = "auto"
            result["launch"]["environment"]["MEEP_NVIDIA_GRAPH_MODE"] = "auto"
            for rank in result["rank_records"]:
                rank["runtime"]["requested_graph"] = "auto"
            refresh_record_hashes(result)
        for rank in direct["rank_records"]:
            rank["runtime"]["resolved_graph"] = "eager"
        direct["resolved_execution"]["graph"] = "eager"
        with self.assertRaisesRegex(runner.RunnerError, "resolved graph"):
            compare.compare(staged, direct, pathlib.Path("s"), pathlib.Path("d"),
                            "e" * 64, "f" * 64)

    def test_module_path_difference_rejected(self):
        staged, direct = route_result("staged"), route_result("direct")
        for rank in direct["rank_records"]:
            rank["module_paths"]["extension"] = "/tmp/other/_meep.so"
        with self.assertRaisesRegex(runner.RunnerError, "module/library paths"):
            compare.compare(staged, direct, pathlib.Path("s"), pathlib.Path("d"),
                            "e" * 64, "f" * 64)

    def test_module_hash_and_cpu_affinity_differences_are_rejected(self):
        for field, message in (("module_sha256", "paths or hashes"),
                               ("cpu_affinity", "CPU affinity")):
            with self.subTest(field=field):
                staged, direct = route_result("staged"), route_result("direct")
                if field == "module_sha256":
                    for rank in direct["rank_records"]:
                        rank[field]["extension"] = "0" * 64
                else:
                    for rank in direct["rank_records"]:
                        rank[field] = [rank["rank"] + 10]
                with self.assertRaisesRegex(runner.RunnerError, message):
                    compare.compare(staged, direct, pathlib.Path("s"), pathlib.Path("d"),
                                    "e" * 64, "f" * 64)

    def test_mpi_and_ucx_version_differences_are_rejected(self):
        for key in ("mpi_version", "ucx_version"):
            with self.subTest(key=key):
                staged, direct = route_result("staged"), route_result("direct")
                direct["launch"][key] = "different"
                refresh_record_hashes(direct)
                with self.assertRaisesRegex(runner.RunnerError, key):
                    compare.compare(staged, direct, pathlib.Path("s"), pathlib.Path("d"),
                                    "e" * 64, "f" * 64)

    def test_staged_accounting_cannot_cancel_across_ranks(self):
        staged, direct = route_result("staged"), route_result("direct")
        first = staged["rank_records"][0]["repetitions"][0]
        second = staged["rank_records"][1]["repetitions"][0]
        first["counter_deltas"]["device_to_host_bytes"] = 0
        second["counter_deltas"]["device_to_host_bytes"] = 20
        for repetition in (first, second):
            repetition["counter_end"] = {
                name: repetition["counter_start"][name] + value
                for name, value in repetition["counter_deltas"].items()
            }
        staged["counter_aggregates"] = runner.reconcile_rank_records(
            staged["rank_records"], HASH, staged["requested_execution"]
        )["counter_aggregates"]
        with self.assertRaisesRegex(runner.RunnerError, "rank 0 staged"):
            compare.compare(staged, direct, pathlib.Path("s"), pathlib.Path("d"),
                            "e" * 64, "f" * 64)


if __name__ == "__main__":
    unittest.main()
