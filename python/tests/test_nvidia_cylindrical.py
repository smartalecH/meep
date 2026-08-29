import unittest

import numpy as np

import meep as mp


@unittest.skipUnless(
    mp.count_processors() == 1,
    "NVIDIA cylindrical acceptance requires the supported single-rank backend",
)
class TestNvidiaCylindrical(unittest.TestCase):
    def _ring_mode(self, backend):
        index, width, radius = 3.4, 1.0, 1.0
        padding, pml_thickness = 4.0, 2.0
        cell = mp.Vector3(radius + width + padding + pml_thickness)
        fcen, df = 0.15, 0.1
        probe = mp.Vector3(radius + 0.1)
        sim = mp.Simulation(
            cell_size=cell,
            geometry=[
                mp.Block(
                    center=mp.Vector3(radius + 0.5 * width),
                    size=mp.Vector3(width, mp.inf, mp.inf),
                    material=mp.Medium(index=index),
                )
            ],
            boundary_layers=[mp.PML(pml_thickness)],
            resolution=10,
            sources=[
                mp.Source(
                    mp.GaussianSource(fcen, fwidth=df),
                    component=mp.Ez,
                    center=probe,
                )
            ],
            dimensions=mp.CYLINDRICAL,
            m=3,
            Courant=0.25,
            split_chunks_evenly=False,
            backend=backend,
            precision="native",
        )
        modes = mp.Harminv(mp.Ez, probe, fcen, df)
        sim.run(mp.after_sources(modes), until_after_sources=200)
        mode = modes.modes[0]
        result = np.array(
            [mode.freq, mode.decay, mode.Q, abs(mode.amp), mode.amp.real, mode.amp.imag]
        )
        sim.reset_meep()
        return result

    def test_m3_ring_resonance(self):
        expected = np.array(
            [
                0.11834617482364607,
                -6.905650667712696e-4,
                85.68792465637777,
                0.025686111790547618,
                -0.023901645997549816,
                -0.009406787843058874,
            ]
        )
        cpu = self._ring_mode("cpu")
        gpu = self._ring_mode("nvidia")
        np.testing.assert_allclose(gpu, cpu, rtol=2e-5, atol=2e-7)
        np.testing.assert_allclose(gpu, expected, rtol=1e-6, atol=2e-8)


if __name__ == "__main__":
    unittest.main()
