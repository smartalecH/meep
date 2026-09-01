"""The 3d adjoint gradient must not depend on how the cell is split into chunks.

`material_grids_addgradient` assembles the gradient by walking the forward and
adjoint DFT chunks. Before the chunk-independent rewrite it (a) paired forward
and adjoint chunks by list position, which is not a spatial correspondence once
a component is missing from some chunk, (b) skipped cross terms when the two
lists had different lengths, and (c) looked up the neighboring forward node with
chunk-local index arithmetic guarded only against the flat index leaving
[0, N) -- a 3d point outside the chunk can still have an in-range flat index, so
the lookup either aliased to an unrelated voxel or substituted zero.

All three only bite where a chunk boundary crosses the design region, so the
error appears with MPI but is *not* an MPI bug: forcing the same split in a
serial run reproduces it exactly. That is what this test does, so it runs in
every CI job rather than only the MPI ones.

A directional finite-difference check compresses the gradient onto a single
number, and this error survives that: measured on the pre-fix build, probing
along g/|g| gave fd/adjoint = 1.0008 while the gradient vector itself was 44%
off in L2. Comparing the gradient vectors directly is what catches it.

The tolerance is therefore what the *field solve* can reproduce across a
different split, not what the gradient assembly contributes; see the comment on
`tol` below.
"""

import os
import unittest

import numpy as np
from autograd import numpy as npa

import meep as mp
import meep.adjoint as mpa

RESOLUTION = 12
DESIGN_N = 6
SEED = 0


def _gradient(
    num_chunks,
    retry_after_failure=False,
    stale_after_forward=False,
    nvidia_failures=(),
    nvidia_auto_failure=None,
    mutate_frequency_after_forward=False,
    use_symmetry=False,
    k_point=False,
    weights_override=None,
    need_gradient=True,
):
    """Adjoint gradient of |alpha|^2 for a fixed problem at a forced chunk count."""
    si, clad = mp.Medium(index=3.48), mp.Medium(index=1.44)
    pml, port_pad, side_pad = 0.5, 0.8, 0.5
    design, thickness = 1.0, 0.22

    sx = 2 * pml + 2 * port_pad + design
    sy = 2 * pml + 2 * side_pad + design
    sz = 2 * pml + 2 * side_pad + thickness

    weights = (
        np.asarray(weights_override, dtype=np.float64).reshape(-1)
        if weights_override is not None
        else np.random.default_rng(SEED).uniform(0.2, 0.8, size=DESIGN_N * DESIGN_N)
    )
    grid = mp.MaterialGrid(
        mp.Vector3(DESIGN_N, DESIGN_N, 1),
        clad,
        si,
        weights=weights.reshape(DESIGN_N, DESIGN_N, 1),
        do_averaging=False,
    )
    region = mpa.DesignRegion(
        grid,
        volume=mp.Volume(
            center=mp.Vector3(), size=mp.Vector3(design, design, thickness)
        ),
    )

    fcen = 1 / 1.55
    port = mp.Vector3(0, sy - 2 * pml, sz - 2 * pml)
    sim = mp.Simulation(
        cell_size=mp.Vector3(sx, sy, sz),
        resolution=RESOLUTION,
        boundary_layers=[mp.PML(pml)],
        default_material=clad,
        geometry=[
            mp.Block(
                center=mp.Vector3(),
                size=mp.Vector3(mp.inf, 0.5, thickness),
                material=si,
            ),
            mp.Block(center=region.center, size=region.size, material=grid),
        ],
        sources=[
            mp.EigenModeSource(
                mp.GaussianSource(fcen, fwidth=0.1 * fcen),
                center=mp.Vector3(-(design / 2 + port_pad / 2)),
                size=port,
                eig_band=1,
            )
        ],
        eps_averaging=False,
        num_chunks=num_chunks,
        symmetries=[mp.Mirror(mp.Y)] if use_symmetry else [],
        k_point=k_point,
    )
    monitor = mpa.EigenmodeCoefficient(
        sim,
        mp.Volume(center=mp.Vector3(design / 2 + port_pad / 2), size=port),
        mode=1,
    )
    opt = mpa.OptimizationProblem(
        simulation=sim,
        objective_functions=[lambda c: npa.abs(c) ** 2],
        objective_arguments=[monitor],
        design_regions=[region],
        frequencies=[fcen],
        decay_by=1e-6,
    )
    if (
        retry_after_failure
        or stale_after_forward
        or nvidia_failures
        or nvidia_auto_failure
        or mutate_frequency_after_forward
    ):
        opt.update_design([weights])
        opt.forward_run()
        if stale_after_forward:
            region.update_design_parameters(np.clip(weights + 0.01, 0.0, 1.0))
            sim.set_materials()
        if mutate_frequency_after_forward:
            opt.frequencies = np.asarray([fcen * 1.01])
        opt.adjoint_run()
        if stale_after_forward:
            stale_rejected = False
            try:
                opt.calculate_gradient()
            except RuntimeError:
                stale_rejected = True
            return float(np.squeeze(opt.f0)), stale_rejected
        if mutate_frequency_after_forward:
            frequency_rejected = False
            try:
                opt.calculate_gradient()
            except RuntimeError:
                frequency_rejected = True
            return float(np.squeeze(opt.f0)), frequency_rejected
        if nvidia_auto_failure:
            previous_mode = os.environ.get("MEEP_NVIDIA_ADJOINT_MODE")
            os.environ["MEEP_NVIDIA_ADJOINT_MODE"] = "auto"
            mp._set_nvidia_adjoint_failure_for_testing(nvidia_auto_failure)
            try:
                opt.calculate_gradient()
            finally:
                mp._set_nvidia_adjoint_failure_for_testing("clear")
                if previous_mode is None:
                    os.environ.pop("MEEP_NVIDIA_ADJOINT_MODE", None)
                else:
                    os.environ["MEEP_NVIDIA_ADJOINT_MODE"] = previous_mode
            f0, gradient = opt.f0, opt.gradient
            return (
                float(np.squeeze(f0)),
                np.asarray(np.real(np.squeeze(gradient)), dtype=np.float64).reshape(-1),
            )
        if retry_after_failure:
            for checkpoint in (0, 1, 2):
                mp._set_adjoint_failure_after_for_testing(checkpoint)
                failed = False
                try:
                    opt.calculate_gradient()
                except RuntimeError:
                    failed = True
                finally:
                    mp._set_adjoint_failure_after_for_testing(-1)
                if not failed:
                    raise AssertionError(
                        f"injected adjoint checkpoint {checkpoint} did not propagate"
                    )
        for point in nvidia_failures:
            mp._set_nvidia_adjoint_failure_for_testing(point)
            failed = False
            try:
                opt.calculate_gradient()
            except RuntimeError:
                failed = True
            finally:
                mp._set_nvidia_adjoint_failure_for_testing("clear")
            if not failed:
                raise AssertionError(f"injected NVIDIA adjoint {point} failure did not propagate")
        # The failed transaction must leave the same snapshots and adjoint
        # chunks available for a clean retry.
        previous_mode = os.environ.get("MEEP_NVIDIA_ADJOINT_MODE")
        if nvidia_failures:
            os.environ["MEEP_NVIDIA_ADJOINT_MODE"] = "host"
        opt.calculate_gradient()
        if nvidia_failures:
            if previous_mode is None:
                os.environ.pop("MEEP_NVIDIA_ADJOINT_MODE", None)
            else:
                os.environ["MEEP_NVIDIA_ADJOINT_MODE"] = previous_mode
        f0, gradient = opt.f0, opt.gradient
    else:
        f0, gradient = opt([weights], need_gradient=need_gradient)
        if not need_gradient:
            return float(np.squeeze(f0)), None
    return (
        float(np.squeeze(f0)),
        np.asarray(np.real(np.squeeze(gradient)), dtype=np.float64).reshape(-1),
    )


def _nonzero_m_gradient():
    """Small cylindrical host-oracle regression for the reversible m flip."""
    fcen = 0.8
    weights = np.asarray([0.25, 0.45, 0.65, 0.75], dtype=np.float64)
    grid = mp.MaterialGrid(
        mp.Vector3(2, 0, 2),
        mp.Medium(index=1.4),
        mp.Medium(index=2.4),
        weights=weights.reshape(2, 1, 2),
        do_averaging=False,
    )
    region = mpa.DesignRegion(
        grid,
        volume=mp.Volume(center=mp.Vector3(0.55, 0, 0), size=mp.Vector3(0.7, 0, 0.7)),
    )
    sim = mp.Simulation(
        cell_size=mp.Vector3(1.5, 0, 1.5),
        dimensions=mp.CYLINDRICAL,
        m=1.0,
        resolution=8,
        boundary_layers=[mp.PML(0.2)],
        geometry=[mp.Block(center=region.center, size=region.size, material=grid)],
        sources=[
            mp.Source(
                mp.GaussianSource(fcen, fwidth=0.3),
                component=mp.Er,
                center=mp.Vector3(0.45, 0, -0.45),
            )
        ],
    )
    monitor = mpa.FourierFields(
        sim,
        mp.Volume(center=mp.Vector3(0.55, 0, 0.35), size=mp.Vector3(0.5, 0, 0)),
        mp.Er,
    )
    opt = mpa.OptimizationProblem(
        simulation=sim,
        objective_functions=lambda values: npa.sum(npa.abs(values) ** 2),
        objective_arguments=[monitor],
        design_regions=[region],
        frequencies=[fcen],
        decay_by=1e-3,
        maximum_run_time=30,
    )
    _, gradient = opt([weights])
    return np.asarray(np.real(np.squeeze(gradient)), dtype=np.float64).reshape(-1)


def _small_cartesian_gradient():
    """Compact native-CUDA adjoint integration fixture."""
    fcen = 0.8
    weights = np.asarray([0.25, 0.45, 0.65, 0.75], dtype=np.float64)
    grid = mp.MaterialGrid(
        mp.Vector3(2, 2),
        mp.Medium(index=1.4),
        mp.Medium(index=2.4),
        weights=weights.reshape(2, 2),
        do_averaging=False,
    )
    region = mpa.DesignRegion(
        grid, volume=mp.Volume(center=mp.Vector3(), size=mp.Vector3(0.7, 0.7))
    )
    sim = mp.Simulation(
        cell_size=mp.Vector3(1.5, 1.5),
        resolution=8,
        accelerator_strict=False,
        boundary_layers=[mp.PML(0.2)],
        geometry=[mp.Block(center=region.center, size=region.size, material=grid)],
        sources=[
            mp.Source(
                mp.GaussianSource(fcen, fwidth=0.3),
                component=mp.Ez,
                center=mp.Vector3(-0.45, 0),
            )
        ],
    )
    monitor = mpa.FourierFields(
        sim, mp.Volume(center=mp.Vector3(0.35, 0), size=mp.Vector3(0, 0.5)), mp.Ez
    )
    opt = mpa.OptimizationProblem(
        simulation=sim,
        objective_functions=lambda values: npa.sum(npa.abs(values) ** 2),
        objective_arguments=[monitor],
        design_regions=[region],
        frequencies=[fcen],
        decay_by=1e-3,
        maximum_run_time=30,
    )
    objective, gradient = opt([weights])
    return float(np.squeeze(objective)), np.asarray(
        np.real(np.squeeze(gradient)), dtype=np.float64
    ).reshape(-1)


class TestAdjointChunks(unittest.TestCase):
    def test_gradient_independent_of_chunk_division(self):
        # Both counts are >= the process count, so this is a genuine change of
        # chunk division in serial and under MPI alike.
        np_ = mp.count_processors()
        f0_a, g_a = _gradient(num_chunks=np_)
        f0_b, g_b = _gradient(num_chunks=3 * np_)

        # A single-precision build cannot resolve either check to 1e-9. The
        # objective alone -- which no gradient code touches -- already moves by
        # one float ulp (1.2e-7) when the cell is split differently under MPI,
        # and the gradient inherits that. The looser bound is still four orders
        # of magnitude below the defect this test guards against.
        tol = 1e-5 if mp.is_single_precision() else 1e-9

        # the forward solve was never chunk-dependent; if this trips, the test
        # problem itself changed rather than the gradient assembly
        self.assertAlmostEqual(f0_a / f0_b, 1.0, delta=tol)

        rel = np.linalg.norm(g_a - g_b) / np.linalg.norm(g_a)
        # Pre-fix this comparison read 4.7e-1. Post-fix the only difference is
        # summation order in the collective reductions: measured 3e-16 (serial,
        # where there are none) to 2e-12 (np 8) in double precision, 1.5e-7 in
        # single.
        self.assertLess(
            rel, tol, f"gradient changed by {rel:.3e} under a different chunk split"
        )

    def test_gradient_failure_preserves_snapshot_and_adjoint_chunks(self):
        _, expected = _gradient(num_chunks=mp.count_processors())
        _, retried = _gradient(
            num_chunks=mp.count_processors(), retry_after_failure=True
        )
        tol = 1e-5 if mp.is_single_precision() else 1e-12
        self.assertLess(
            np.linalg.norm(expected - retried) / np.linalg.norm(expected), tol
        )

    def test_material_mutation_rejects_stale_forward_snapshot(self):
        _, stale_rejected = _gradient(
            num_chunks=mp.count_processors(), stale_after_forward=True
        )
        self.assertTrue(stale_rejected)

    def test_frequency_mutation_rejects_stale_forward_snapshot(self):
        _, stale_rejected = _gradient(
            num_chunks=mp.count_processors(), mutate_frequency_after_forward=True
        )
        self.assertTrue(stale_rejected)

    def test_nonzero_bloch_k_restores_host_fallback_snapshot(self):
        previous_mode = os.environ.get("MEEP_NVIDIA_ADJOINT_MODE")
        try:
            os.environ["MEEP_NVIDIA_ADJOINT_MODE"] = "host"
            _, gradient = _gradient(
                num_chunks=mp.count_processors(), k_point=mp.Vector3(0.07, 0.03, 0.0)
            )
            self.assertTrue(np.all(np.isfinite(gradient)))
        finally:
            if previous_mode is None:
                os.environ.pop("MEEP_NVIDIA_ADJOINT_MODE", None)
            else:
                os.environ["MEEP_NVIDIA_ADJOINT_MODE"] = previous_mode

    def test_nonzero_cylindrical_m_restores_host_fallback_snapshot(self):
        previous_mode = os.environ.get("MEEP_NVIDIA_ADJOINT_MODE")
        try:
            os.environ["MEEP_NVIDIA_ADJOINT_MODE"] = "host"
            gradient = _nonzero_m_gradient()
            self.assertTrue(np.all(np.isfinite(gradient)))
        finally:
            if previous_mode is None:
                os.environ.pop("MEEP_NVIDIA_ADJOINT_MODE", None)
            else:
                os.environ["MEEP_NVIDIA_ADJOINT_MODE"] = previous_mode

    @unittest.skipUnless(
        os.environ.get("MEEP_TEST_ADJOINT_FD"), "expensive finite-difference gate"
    )
    def test_directional_finite_difference(self):
        weights = np.random.default_rng(SEED).uniform(
            0.2, 0.8, size=DESIGN_N * DESIGN_N
        )
        direction = np.random.default_rng(SEED + 1).normal(size=weights.size)
        direction /= np.linalg.norm(direction)
        step = 1e-4
        f0, gradient = _gradient(
            num_chunks=mp.count_processors(), weights_override=weights
        )
        perturbed, _ = _gradient(
            num_chunks=mp.count_processors(),
            weights_override=weights + step * direction,
            need_gradient=False,
        )
        adjoint_delta = step * np.dot(gradient, direction)
        finite_delta = perturbed - f0
        relative = abs(adjoint_delta - finite_delta) / max(
            abs(adjoint_delta), abs(finite_delta), 1e-15
        )
        self.assertLess(relative, 0.03)

    @unittest.skipUnless(
        os.environ.get("MEEP_TEST_NVIDIA_ADJOINT_FAILURES"),
        "requires the NVIDIA adjoint backend",
    )
    def test_nvidia_failures_preserve_retry_state(self):
        expected = None
        for failures in (("allocation", "upload", "download"), ("launch",)):
            _, retried = _gradient(
                num_chunks=mp.count_processors(), nvidia_failures=failures
            )
            tol = 1e-5 if mp.is_single_precision() else 1e-12
            if expected is None:
                expected = retried
            else:
                self.assertLess(
                    np.linalg.norm(expected - retried) / np.linalg.norm(expected), tol
                )
        _, automatic = _gradient(
            num_chunks=mp.count_processors(), nvidia_auto_failure="allocation"
        )
        self.assertLess(
            np.linalg.norm(expected - automatic) / np.linalg.norm(expected), tol
        )

    @unittest.skipUnless(
        os.environ.get("MEEP_TEST_NVIDIA_ADJOINT_FAILURES"),
        "requires the NVIDIA adjoint backend",
    )
    def test_nvidia_support_policy_is_explicit(self):
        previous_mode = os.environ.get("MEEP_NVIDIA_ADJOINT_MODE")
        try:
            os.environ["MEEP_NVIDIA_ADJOINT_MODE"] = "required"
            with self.assertRaises(RuntimeError):
                _gradient(num_chunks=mp.count_processors(), use_symmetry=True)
            os.environ["MEEP_NVIDIA_ADJOINT_MODE"] = "auto"
            _, gradient = _gradient(
                num_chunks=mp.count_processors(), use_symmetry=True
            )
            self.assertTrue(np.all(np.isfinite(gradient)))
        finally:
            if previous_mode is None:
                os.environ.pop("MEEP_NVIDIA_ADJOINT_MODE", None)
            else:
                os.environ["MEEP_NVIDIA_ADJOINT_MODE"] = previous_mode

    @unittest.skipUnless(
        os.environ.get("MEEP_TEST_NVIDIA_ADJOINT_REQUIRED"),
        "requires the NVIDIA adjoint backend",
    )
    def test_nvidia_required_compact_gradient(self):
        previous_mode = os.environ.get("MEEP_NVIDIA_ADJOINT_MODE")
        try:
            os.environ["MEEP_NVIDIA_ADJOINT_MODE"] = "required"
            objective, gradient = _small_cartesian_gradient()
            self.assertTrue(np.isfinite(objective))
            self.assertTrue(np.all(np.isfinite(gradient)))
        finally:
            if previous_mode is None:
                os.environ.pop("MEEP_NVIDIA_ADJOINT_MODE", None)
            else:
                os.environ["MEEP_NVIDIA_ADJOINT_MODE"] = previous_mode


if __name__ == "__main__":
    unittest.main()
