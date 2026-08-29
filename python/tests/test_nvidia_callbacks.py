# Copyright (C) 2005-2026 Massachusetts Institute of Technology

"""Single-GPU acceptance tests for Python host boundaries in Phase 2 PR3."""

import tempfile
import unittest

import numpy as np

import meep as mp


@unittest.skipUnless(
    mp.count_processors() == 1,
    "NVIDIA callback acceptance requires the supported single-rank backend",
)
class TestNvidiaCallbacks(unittest.TestCase):
    def _simulation(self, backend, source_time, output_dir=None):
        sim = mp.Simulation(
            cell_size=mp.Vector3(3, 3),
            resolution=8,
            sources=[mp.Source(source_time, component=mp.Ez, center=mp.Vector3(0.3, 0.4))],
            backend=backend,
            accelerator_strict=True,
        )
        if output_dir:
            sim.use_output_directory(output_dir)
        return sim

    def test_dft_data_roundtrip_and_callbacks(self):
        cpu = self._simulation("cpu", mp.GaussianSource(0.31, fwidth=0.18))
        gpu = self._simulation("nvidia", mp.GaussianSource(0.31, fwidth=0.18))
        monitors = []
        for sim in (cpu, gpu):
            flux = sim.add_flux(
                0.31,
                0.08,
                3,
                mp.FluxRegion(center=mp.Vector3(x=0.55), size=mp.Vector3(y=1.4)),
                decimation_factor=1,
            )
            force = sim.add_force(
                0.31,
                0.08,
                3,
                mp.ForceRegion(
                    direction=mp.X,
                    center=mp.Vector3(x=0.45),
                    size=mp.Vector3(y=1.2),
                ),
                decimation_factor=1,
            )
            near2far = sim.add_near2far(
                0.31,
                0.08,
                3,
                mp.Near2FarRegion(center=mp.Vector3(y=0.6), size=mp.Vector3(x=1.2)),
                decimation_factor=1,
            )
            monitors.append((flux, force, near2far))
            sim.run(until=6)

        cpu_flux, cpu_force, cpu_n2f = monitors[0]
        gpu_flux, gpu_force, gpu_n2f = monitors[1]
        cpu_data = cpu.get_flux_data(cpu_flux)
        gpu_data = gpu.get_flux_data(gpu_flux)
        np.testing.assert_allclose(gpu_data.E, cpu_data.E, rtol=3e-4, atol=3e-6)
        np.testing.assert_allclose(gpu_data.H, cpu_data.H, rtol=3e-4, atol=3e-6)

        force_data = gpu.get_force_data(gpu_force)
        self.assertEqual(len(force_data.offdiag1), len(cpu.get_force_data(cpu_force).offdiag1))
        self.assertEqual(len(force_data.offdiag2), len(cpu.get_force_data(cpu_force).offdiag2))
        self.assertEqual(len(force_data.diag), len(cpu.get_force_data(cpu_force).diag))
        n2f_data = gpu.get_near2far_data(gpu_n2f)
        self.assertEqual(len(n2f_data.F), len(cpu.get_near2far_data(cpu_n2f).F))

        gpu.load_minus_flux_data(gpu_flux, cpu_data)
        negative = gpu.get_flux_data(gpu_flux)
        np.testing.assert_allclose(negative.E, -cpu_data.E, rtol=3e-4, atol=3e-6)
        np.testing.assert_allclose(negative.H, -cpu_data.H, rtol=3e-4, atol=3e-6)
        gpu.load_flux_data(gpu_flux, cpu_data)
        cpu.run(until=cpu.meep_time() + cpu.fields.dt)
        gpu.run(until=gpu.meep_time() + gpu.fields.dt)
        np.testing.assert_allclose(mp.get_fluxes(gpu_flux), mp.get_fluxes(cpu_flux), rtol=5e-4, atol=3e-6)

        cpu_calls = []
        gpu_calls = []

        def field_callback(calls):
            def callback(point, ez):
                calls.append((point.x, point.y, complex(ez)))
                return ez

            return callback

        cpu_integral = cpu.integrate_field_function([mp.Ez], field_callback(cpu_calls))
        gpu_integral = gpu.integrate_field_function([mp.Ez], field_callback(gpu_calls))
        self.assertEqual(len(gpu_calls), len(cpu_calls))
        self.assertGreater(len(gpu_calls), 0)
        self.assertAlmostEqual(gpu_integral.real, cpu_integral.real, places=5)
        self.assertAlmostEqual(
            gpu.max_abs_field_function([mp.Ez], lambda point, ez: ez),
            cpu.max_abs_field_function([mp.Ez], lambda point, ez: ez),
            places=5,
        )
        with tempfile.TemporaryDirectory() as cpu_dir, tempfile.TemporaryDirectory() as gpu_dir:
            cpu.use_output_directory(cpu_dir)
            gpu.use_output_directory(gpu_dir)
            cpu.output_field_function("callback-field", [mp.Ez], lambda point, ez: ez)
            gpu.output_field_function("callback-field", [mp.Ez], lambda point, ez: ez)

        with self.assertRaisesRegex(RuntimeError, "magnetic synchronization"):
            gpu.run(mp.synchronized_magnetic(lambda sim: None), until=gpu.meep_time() + gpu.fields.dt)
        before = gpu.fields.t
        gpu.fields.advance(1)
        self.assertEqual(gpu.fields.t, before + 1)

    def test_custom_source_gil_timing_and_exception(self):
        for integrated in (False, True):
            traces = {}
            for backend in ("cpu", "nvidia"):
                calls = []

                def source_time(t, calls=calls):
                    calls.append(t)
                    return np.exp(-0.2 * t) * np.exp(-2j * np.pi * 0.27 * t)

                sim = self._simulation(
                    backend,
                    mp.CustomSource(
                        src_func=source_time,
                        start_time=0,
                        end_time=3,
                        is_integrated=integrated,
                    ),
                )
                sim.run(until=1)
                traces[backend] = calls
            self.assertGreater(len(traces["nvidia"]), 0)
            self.assertEqual(traces["nvidia"], traces["cpu"])

        calls = []

        def failing_source(t):
            calls.append(t)
            raise ValueError("intentional custom-source failure")

        sim = self._simulation(
            "nvidia", mp.CustomSource(src_func=failing_source, start_time=0, end_time=3)
        )
        with self.assertRaises(RuntimeError):
            sim.run(until=1)
        self.assertGreater(len(calls), 0)


if __name__ == "__main__":
    unittest.main()
