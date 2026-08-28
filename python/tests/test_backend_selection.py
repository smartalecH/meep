# Copyright (C) 2005-2026 Massachusetts Institute of Technology
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2, or (at your option)
# any later version.

"""PR 7 acceptance tests: backend and precision selection from Python.

Also covers what the plan calls the primary risk of PR 7 -- that every
pre-run material query keeps working with no run() call.
"""

import os
import unittest

import numpy as np

import meep as mp


def _sim(**kwargs):
    return mp.Simulation(
        cell_size=mp.Vector3(3, 3),
        resolution=10,
        geometry=[mp.Block(mp.Vector3(1, 1, mp.inf), material=mp.Medium(epsilon=12))],
        boundary_layers=[mp.PML(0.5)],
        sources=[
            mp.Source(
                mp.GaussianSource(0.3, fwidth=0.1), component=mp.Ez, center=mp.Vector3()
            )
        ],
        **kwargs,
    )


class TestBackendSelection(unittest.TestCase):
    def test_default_is_cpu(self):
        sim = _sim()
        sim.init_sim()
        sim.fields.advance(3)
        self.assertEqual(sim.fields.t, 3)

    def test_explicit_cpu(self):
        sim = _sim(backend="cpu", precision="native")
        sim.init_sim()
        sim.fields.advance(3)
        self.assertEqual(sim.fields.t, 3)

    def test_cpu_and_default_agree(self):
        """Selecting cpu explicitly must be the same simulation as the default."""
        a, b = _sim(), _sim(backend="cpu")
        for s in (a, b):
            s.init_sim()
            s.fields.advance(6)
        self.assertTrue(
            np.array_equal(a.get_array(component=mp.Ez), b.get_array(component=mp.Ez))
        )

    def test_nvidia_is_rejected(self):
        sim = _sim(backend="nvidia")
        with self.assertRaises(Exception) as cm:
            sim.init_sim()
        self.assertIn("nvidia", str(cm.exception).lower())

    def test_unsupported_precision_is_rejected(self):
        for p in ("mixed", "f32"):
            with self.subTest(precision=p):
                sim = _sim(precision=p)
                with self.assertRaises(Exception) as cm:
                    sim.init_sim()
                self.assertIn("native", str(cm.exception).lower())

    def test_bad_names_are_rejected_early(self):
        for kwargs in (dict(backend="gpu"), dict(precision="float16")):
            with self.subTest(**kwargs):
                sim = _sim(**kwargs)
                with self.assertRaises(ValueError):
                    sim.init_sim()

    def test_device_id_on_cpu_is_rejected(self):
        sim = _sim(device_id=0)
        with self.assertRaises(Exception):
            sim.init_sim()


class TestEnvironmentOverrides(unittest.TestCase):
    """MEEP_BACKEND / MEEP_PRECISION must behave like the constructor args."""

    def setUp(self):
        self._saved = {
            k: os.environ.get(k) for k in ("MEEP_BACKEND", "MEEP_PRECISION")
        }

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_env_backend_overrides(self):
        os.environ["MEEP_BACKEND"] = "nvidia"
        sim = _sim(backend="cpu")  # the environment wins
        with self.assertRaises(Exception) as cm:
            sim.init_sim()
        self.assertIn("nvidia", str(cm.exception).lower())

    def test_env_precision_overrides(self):
        os.environ["MEEP_PRECISION"] = "f32"
        sim = _sim()
        with self.assertRaises(Exception) as cm:
            sim.init_sim()
        self.assertIn("native", str(cm.exception).lower())


class TestPreRunMaterialQueries(unittest.TestCase):
    """The primary risk the plan flags for PR 7.

    Deferring coefficient construction to preparation would break every one of
    these. The CPU material path stays eager precisely so it does not.

    "Without run" means exactly that: init_sim() is called where meep already
    required it before this stack, and no timestep is ever taken.
    """

    def test_get_epsilon_without_run(self):
        sim = _sim()
        sim.init_sim()  # no run(); the arrays must already hold their values
        eps = sim.get_epsilon()
        self.assertGreater(float(np.max(eps)), 11.0)

    def test_get_array_dielectric_without_run(self):
        sim = _sim()
        sim.init_sim()
        eps = sim.get_array(mp.Dielectric)
        self.assertGreater(float(np.max(eps)), 11.0)

    def test_geps_without_run(self):
        sim = _sim()
        sim.init_sim()
        self.assertIsNotNone(sim.geps)

    def test_structure_dump_without_run(self):
        import tempfile

        sim = _sim()
        sim.init_sim()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "structure.h5")
            sim.dump_structure(path)
            self.assertTrue(os.path.exists(path))

    def test_plot2D_without_run(self):
        import matplotlib

        matplotlib.use("Agg")
        sim = _sim()
        sim.plot2D()


if __name__ == "__main__":
    unittest.main()
