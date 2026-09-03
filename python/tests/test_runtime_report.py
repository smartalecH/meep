import json
import unittest

import meep as mp


class TestRuntimeReport(unittest.TestCase):
    def test_cpu_report_is_native_dict(self):
        sim = mp.Simulation(cell_size=mp.Vector3(2, 2), resolution=4, backend="cpu")
        sim.init_sim()
        report = sim.get_execution_runtime_report()
        self.assertIsInstance(report, dict)
        self.assertEqual(report["resolved_backend"], "cpu")
        self.assertFalse(report["device_owner"])
        self.assertEqual(report["device_uuid"], "")
        self.assertEqual(report["direct_bytes"], 0)
        self.assertEqual(report["counter_scope"], "rank_local_current_epoch")
        self.assertEqual(report["backend_counter_scope"], "rank_local_backend_lifetime")
        self.assertEqual(report["memory_gauge_scope"], "rank_local_process_lifetime")
        self.assertEqual(report["setup_counter_scope"], "rank_local_current_backend_state")
        self.assertEqual(
            report["transport_timing_scope"],
            "rank_local_current_transport_epoch_host_elapsed",
        )
        self.assertEqual(report["allocation_counter_scope"], "rank_local_process_lifetime")
        self.assertNotIn("\x00", report["mpi_provider"])
        timing_and_steady_keys = (
            "material_recipe_prepare_nanoseconds",
            "material_initialize_nanoseconds",
            "graph_build_nanoseconds",
            "gather_pack_nanoseconds",
            "device_to_host_nanoseconds",
            "mpi_progress_nanoseconds",
            "mpi_wait_nanoseconds",
            "host_to_device_nanoseconds",
            "scatter_unpack_nanoseconds",
            "steady_allocation_count",
            "graph_recapture_count",
            "full_field_copy_count",
        )
        for key in timing_and_steady_keys:
            self.assertIsInstance(report[key], int)
            self.assertGreaterEqual(report[key], 0)

    def test_active_communicator_allgather_is_rank_ordered(self):
        payload = json.dumps({"rank": mp.my_rank()}, sort_keys=True)
        records = mp.active_communicator_allgather_json(payload)
        self.assertEqual(len(records), mp.count_processors())
        self.assertEqual(
            [json.loads(record)["rank"] for record in records],
            list(range(mp.count_processors())),
        )

    def test_empty_payload(self):
        self.assertEqual(
            mp.active_communicator_allgather_json(""),
            [""] * mp.count_processors(),
        )

    def test_asymmetric_embedded_nul_is_rejected_collectively(self):
        if mp.count_processors() < 2:
            self.skipTest("requires at least two MPI ranks")
        payload = "bad\x00payload" if mp.my_rank() == mp.count_processors() - 1 else "{}"
        with self.assertRaisesRegex(RuntimeError, "invalid|NUL"):
            mp.active_communicator_allgather_json(payload)


if __name__ == "__main__":
    unittest.main()
