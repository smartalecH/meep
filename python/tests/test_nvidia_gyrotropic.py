import unittest

import numpy as np

import meep as mp

from test_faraday_rotation import kgyro_drude, kgyro_llg, kgyro_lorentzian


@unittest.skipUnless(
    mp.count_processors() == 1,
    "NVIDIA gyrotropic acceptance requires the supported single-rank backend",
)
class TestNvidiaGyrotropic(unittest.TestCase):
    def _run(self, susceptibility, backend):
        length, source_z, output_z = 12.0, -4.5, 4.0
        frequency, run_time = 0.8, 100.0
        medium = mp.Medium(
            epsilon=1.5, mu=1.0, E_susceptibilities=[susceptibility]
        )
        simulation = mp.Simulation(
            cell_size=mp.Vector3(0, 0, length),
            sources=[
                mp.Source(
                    mp.ContinuousSource(frequency=frequency),
                    component=mp.Ex,
                    center=mp.Vector3(0, 0, source_z),
                )
            ],
            boundary_layers=[mp.PML(thickness=1.0, direction=mp.Z)],
            default_material=medium,
            resolution=24,
            backend=backend,
            precision="native",
        )
        ex, ey, times = [], [], []

        def sample(sim):
            ex.append(sim.get_field_point(mp.Ex, mp.Vector3(0, 0, output_z)))
            ey.append(sim.get_field_point(mp.Ey, mp.Vector3(0, 0, output_z)))
            times.append(sim.meep_time())

        simulation.run(mp.after_time(0.5 * run_time, mp.at_every(0.05, sample)), until=run_time)
        phase = np.exp(2j * np.pi * frequency * np.asarray(times))
        ex_amp = np.sum(np.asarray(ex) * phase)
        ey_amp = np.sum(np.asarray(ey) * phase)
        signed = np.degrees(
            np.arctan2(np.real(ey_amp * np.conj(ex_amp)), np.abs(ex_amp) ** 2)
        )
        simulation.reset_meep()
        return signed, ex_amp, ey_amp

    def _check_model(self, susceptibility, wave_number):
        distance = 8.5
        expected = np.degrees(
            np.arctan2(
                abs(np.sin(wave_number * distance).real),
                abs(np.cos(wave_number * distance).real),
            )
        )
        cpu_angle, _, _ = self._run(susceptibility, "cpu")
        gpu_angle, _, _ = self._run(susceptibility, "nvidia")
        self.assertLessEqual(abs(abs(gpu_angle) - expected), 1.5)
        self.assertLessEqual(abs(gpu_angle - cpu_angle), 0.05)

    def test_models_and_handedness(self):
        frequency, epsilon, resonance, gamma, sigma, bias = 0.8, 1.5, 1.0, 1e-3, 0.1, 0.15
        self._check_model(
            mp.GyrotropicLorentzianSusceptibility(
                frequency=resonance,
                gamma=gamma,
                sigma=sigma,
                bias=mp.Vector3(0, 0, bias),
            ),
            kgyro_lorentzian(frequency, epsilon, resonance, gamma, sigma, bias),
        )
        self._check_model(
            mp.GyrotropicDrudeSusceptibility(
                frequency=resonance,
                gamma=gamma,
                sigma=sigma,
                bias=mp.Vector3(0, 0, bias),
            ),
            kgyro_drude(frequency, epsilon, resonance, gamma, sigma, bias),
        )
        alpha = 1e-5
        self._check_model(
            mp.GyrotropicSaturatedSusceptibility(
                frequency=resonance,
                gamma=gamma,
                sigma=sigma,
                alpha=alpha,
                bias=mp.Vector3(0, 0, 1),
            ),
            kgyro_llg(frequency, epsilon, resonance, gamma, sigma, alpha),
        )

        positive = mp.GyrotropicLorentzianSusceptibility(
            frequency=resonance,
            gamma=gamma,
            sigma=sigma,
            bias=mp.Vector3(0, 0, bias),
        )
        negative = mp.GyrotropicLorentzianSusceptibility(
            frequency=resonance,
            gamma=gamma,
            sigma=sigma,
            bias=mp.Vector3(0, 0, -bias),
        )
        plus_angle, plus_ex, plus_ey = self._run(positive, "nvidia")
        minus_angle, minus_ex, minus_ey = self._run(negative, "nvidia")
        self.assertLess(plus_angle * minus_angle, 0.0)
        self.assertAlmostEqual(abs(plus_angle), abs(minus_angle), delta=0.05)
        np.testing.assert_allclose(plus_ex, minus_ex, rtol=2e-4, atol=2e-6)
        np.testing.assert_allclose(plus_ey, -minus_ey, rtol=2e-4, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
