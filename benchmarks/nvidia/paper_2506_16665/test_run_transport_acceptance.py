#!/usr/bin/env python3

import copy
import contextlib
import importlib.util
import io
import json
import math
import pathlib
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
    }


def counters(route):
    result = {name: 0 for name in acceptance.mpi_runner.COUNTER_AGGREGATIONS}
    if route != "cpu":
        result.update(messages_sent=1, messages_received=1, bytes_sent=16,
                      bytes_received=16)
    if route == "staged":
        result.update(device_to_host_calls=1, device_to_host_bytes=16,
                      host_to_device_calls=1, host_to_device_bytes=16)
    elif route == "direct":
        result["direct_bytes"] = 32
    return result


def artifact(route):
    size = 1 if route == "cpu" else 2
    records = []
    for rank in range(size):
        records.append({
            "rank": rank,
            "runtime": runtime(route, rank, size),
            "steps": 10,
            "stop_reason": "field_energy_decay",
            "observables": {"samples": [[float(index), 0.0]
                                          for index in range(len(acceptance.CASE["sample_z"]))]},
            "counter_deltas": counters(route),
        })
    return {
        "schema_version": 1,
        "kind": "meep_builtin_transport_acceptance",
        "case": copy.deepcopy(acceptance.CASE),
        "case_sha256": acceptance.canonical_hash(acceptance.CASE),
        "route": route,
        "build_identity": {
            "source": "/tmp/meep", "commit": "a" * 40, "dirty": False,
            "python": "/tmp/python", "meep_module": "/tmp/meep/__init__.py",
            "meep_extension": "/tmp/_meep.so",
        },
        "rank_records": records,
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
            (lambda value: value["rank_records"][0]["runtime"].update(resolved_transport="staged"), "resolved route"),
            (lambda value: value["rank_records"][0]["runtime"].update(mpi_provider=""), "MPI provider"),
            (lambda value: value["rank_records"][0]["runtime"].update(mpi_query_available=False), "GPU-aware MPI"),
            (lambda value: value["rank_records"][0]["runtime"].update(transport_pinned_bytes=1), "GPU-aware MPI"),
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
        with self.assertRaisesRegex(acceptance.mpi_runner.RunnerError, "fixed observable"):
            acceptance.validate_acceptance_artifact(value, "direct")
        value = artifact("direct")
        value["rank_records"][0]["observables"]["samples"][0][0] = math.nan
        with self.assertRaisesRegex(acceptance.mpi_runner.RunnerError, "invalid observable"):
            acceptance.validate_acceptance_artifact(value, "direct")
        value = artifact("direct")
        value["rank_records"][0]["counter_deltas"]["direct_bytes"] = -1
        with self.assertRaisesRegex(acceptance.mpi_runner.RunnerError, "counter direct_bytes"):
            acceptance.validate_acceptance_artifact(value, "direct")
        value = artifact("direct")
        del value["rank_records"][0]["counter_deltas"]["slot_reuses"]
        with self.assertRaisesRegex(acceptance.mpi_runner.RunnerError, "required counter set"):
            acceptance.validate_acceptance_artifact(value, "direct")

    def test_steady_state_side_effect_counters_are_rejected(self):
        for name in (
            "steady_allocation_count", "graph_recapture_count", "full_field_copy_count"
        ):
            with self.subTest(counter=name):
                value = artifact("direct")
                value["rank_records"][0]["counter_deltas"][name] = 1
                with self.assertRaisesRegex(
                    acceptance.mpi_runner.RunnerError, f"steady-state counter {name}"
                ):
                    acceptance.validate_acceptance_artifact(value, "direct")

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

    def test_cli_requires_and_propagates_explicit_import_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            output_dir = root / "out"
            python_path = root / "python"
            library_path = root / "lib"
            calls = []

            def fake_run(command, env, timeout):
                calls.append((command, env, timeout))
                route = command[command.index("--worker") + 1]
                output = pathlib.Path(command[command.index("--output") + 1])
                output.write_text(json.dumps(artifact(route)), encoding="utf-8")

            with mock.patch.object(acceptance, "_run", side_effect=fake_run):
                status = acceptance.main([
                    "--output-dir", str(output_dir),
                    "--prefix", str(root / "prefix"),
                    "--pythonpath", str(python_path),
                    "--library-path", str(library_path),
                ])
            self.assertEqual(status, 0)
            self.assertEqual(len(calls), 3)
            for _command, environment, _timeout in calls:
                self.assertEqual(environment["PYTHONPATH"], str(python_path.resolve()))
                self.assertTrue(environment["LD_LIBRARY_PATH"].startswith(
                    str(library_path.resolve()) + ":"
                ))

    def test_cli_rejects_implicit_import_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(acceptance, "_run") as run:
                with contextlib.redirect_stderr(io.StringIO()):
                    status = acceptance.main(["--output-dir", directory])
            self.assertEqual(status, 2)
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
