#!/usr/bin/env python3

import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).with_name("run_mpi_acceptance.py")
SPEC = importlib.util.spec_from_file_location("run_mpi_acceptance", SCRIPT)
acceptance = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(acceptance)


class ProviderEnvironmentTest(unittest.TestCase):
    def test_ucx_environment_is_explicit(self):
        environment = acceptance.provider_environment(
            "ucx", {"OMPI_MCA_btl": "stale", "UNCHANGED": "yes"}
        )
        self.assertEqual(environment["OMPI_MCA_opal_cuda_support"], "true")
        self.assertEqual(environment["OMPI_MCA_pml"], "ucx")
        self.assertEqual(environment["UCX_TLS"], "self,sm,cuda_copy,cuda_ipc")
        self.assertNotIn("OMPI_MCA_btl", environment)
        self.assertEqual(environment["UNCHANGED"], "yes")

    def test_ob1_environment_is_an_explicit_fallback(self):
        environment = acceptance.provider_environment(
            "ob1", {"UCX_TLS": "stale", "UNCHANGED": "yes"}
        )
        self.assertEqual(environment["OMPI_MCA_opal_cuda_support"], "true")
        self.assertEqual(environment["OMPI_MCA_pml"], "ob1")
        self.assertEqual(environment["OMPI_MCA_btl"], "self,smcuda,tcp")
        self.assertNotIn("UCX_TLS", environment)
        self.assertEqual(environment["UNCHANGED"], "yes")

    def test_unknown_route_rejected(self):
        with self.assertRaisesRegex(acceptance.AcceptanceError, "ucx or ob1"):
            acceptance.provider_environment("auto", {})


if __name__ == "__main__":
    unittest.main()
