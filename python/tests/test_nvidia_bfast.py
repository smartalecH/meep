import math
import unittest

import numpy as np

import meep as mp


@unittest.skipUnless(
    mp.count_processors() == 1,
    "NVIDIA BFAST acceptance requires the supported single-rank backend",
)
class TestNvidiaBfast(unittest.TestCase):
    def _reflectance(self, backend, theta_deg):
        n1, n2 = 1.4, 3.5
        theta = math.radians(theta_deg)
        scaled_k = (n1 * math.sin(theta), 0, 0)
        pml_thickness, length_z = 0.8, 5.0
        size_z = length_z + 2 * pml_thickness
        fmin, fmax = 1 / 0.80, 1 / 0.70
        fcen, df, nfreq = 0.5 * (fmin + fmax), fmax - fmin, 3
        source_df = 1 / 0.45 - 1 / 0.75
        courant = (1 - abs(scaled_k[0])) / math.sqrt(3)
        source_z = -0.5 * size_z + pml_thickness
        monitor_z = source_z + 0.25 * length_z

        def make_sim(geometry):
            return mp.Simulation(
                resolution=100,
                cell_size=mp.Vector3(z=size_z),
                dimensions=3,
                default_material=mp.Medium(index=n1),
                geometry=geometry,
                sources=[
                    mp.Source(
                        mp.GaussianSource(fcen, fwidth=source_df),
                        component=mp.Ex,
                        center=mp.Vector3(z=source_z),
                    )
                ],
                boundary_layers=[mp.PML(pml_thickness)],
                bfast_scaled_k=scaled_k,
                Courant=courant,
                backend=backend,
                precision="native",
            )

        monitor_region = mp.FluxRegion(center=mp.Vector3(z=monitor_z))
        def stop():
            return mp.stop_when_fields_decayed(
                30, mp.Ex, mp.Vector3(z=monitor_z), 1e-6
            )

        empty = make_sim([])
        incident = empty.add_flux(fcen, df, nfreq, monitor_region)
        empty.run(until_after_sources=stop())
        incident_data = empty.get_flux_data(incident)
        incident_flux = np.asarray(mp.get_fluxes(incident))
        empty.reset_meep()

        interface = make_sim(
            [
                mp.Block(
                    size=mp.Vector3(mp.inf, mp.inf, 0.5 * size_z),
                    center=mp.Vector3(z=0.25 * size_z),
                    material=mp.Medium(index=n2),
                )
            ]
        )
        reflected = interface.add_flux(fcen, df, nfreq, monitor_region)
        interface.init_sim()
        interface.load_minus_flux_data(reflected, incident_data)
        interface.run(until_after_sources=stop())
        result = -np.asarray(mp.get_fluxes(reflected)) / incident_flux
        frequencies = np.asarray(mp.get_flux_freqs(reflected))
        interface.reset_meep()
        return frequencies, result

    def test_fixed_angle_reflectance(self):
        theta = 35.7
        frequencies, cpu = self._reflectance("cpu", theta)
        gpu_frequencies, gpu = self._reflectance("nvidia", theta)
        _, gpu_negative = self._reflectance("nvidia", -theta)
        np.testing.assert_allclose(gpu_frequencies, frequencies, rtol=0, atol=0)
        np.testing.assert_allclose(gpu, cpu, rtol=3e-4, atol=3e-6)
        np.testing.assert_allclose(gpu_negative, gpu, rtol=3e-4, atol=3e-6)

        theta_out = math.asin(1.4 * math.sin(math.radians(theta)) / 3.5)
        fresnel = abs(
            (1.4 * math.cos(theta_out) - 3.5 * math.cos(math.radians(theta)))
            / (1.4 * math.cos(theta_out) + 3.5 * math.cos(math.radians(theta)))
        ) ** 2
        np.testing.assert_allclose(gpu, fresnel, rtol=0.03, atol=0)


if __name__ == "__main__":
    unittest.main()
