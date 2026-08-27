# Copyright (C) 2005-2026 Massachusetts Institute of Technology
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2, or (at your option)
# any later version.

"""Configuration matrix for the bitwise-neutrality harness.

Phase 1 of the backend-neutral refactor changes no arithmetic, so every
configuration here must produce bitwise-identical raw state before and after
each PR in the stack.  The full cross product of the axes in section 4 of the
plan is far too large; this is a covering set that hits every axis value at
least twice.

Determinism is not optional -- an unpinned degree of freedom turns the harness
into a random-failure generator and destroys confidence in the whole stack.
Every knob that can perturb reduction order or storage layout is pinned here
or in ``dump_state.py``:

  * ``split_chunks_evenly=True`` and an explicit ``num_chunks`` so the
    decomposition never varies with the host core count,
  * ``loop_tile_base_db`` / ``loop_tile_base_eh`` fixed so tiling never varies,
  * ``mp.set_random_seed`` before every run, for noisy susceptibilities,
  * ``OMP_NUM_THREADS`` pinned by the driver.

Coverage caveats are recorded in README.md; keep the two in sync.
"""

import math

import meep as mp

# Pinned everywhere. Small on purpose: the harness runs the whole matrix many
# times, and neutrality is not a convergence property -- a 3-voxel difference
# is as fatal as a 3000-voxel one.
RESOLUTION_1D = 20
RESOLUTION_2D = 12
RESOLUTION_3D = 7
RESOLUTION_CYL = 12

LOOP_TILE_BASE_DB = 10000
LOOP_TILE_BASE_EH = 10000
RANDOM_SEED = 20260825


def _src(freq=0.3, width=0.5, component=mp.Ez, center=None, size=None, integrated=False):
    kwargs = dict(
        src=mp.GaussianSource(freq, fwidth=width, is_integrated=integrated),
        component=component,
        center=center if center is not None else mp.Vector3(),
    )
    if size is not None:
        kwargs["size"] = size
    return mp.Source(**kwargs)


def _lorentzian():
    return [mp.LorentzianSusceptibility(frequency=1.1, gamma=1e-5, sigma=0.5)]


def _drude():
    return [mp.DrudeSusceptibility(frequency=1.1, gamma=0.02, sigma=0.4)]


def _noisy():
    return [
        mp.NoisyLorentzianSusceptibility(
            noise_amp=0.01, frequency=1.1, gamma=0.05, sigma=0.5
        )
    ]


def _gyrotropic():
    return [
        mp.GyrotropicLorentzianSusceptibility(
            frequency=1.1, gamma=1e-5, sigma=0.5, bias=mp.Vector3(0, 0, 1.0)
        )
    ]


def _multilevel():
    # Same shape as tests/test_multilevel_atom.py, but only a handful of steps.
    omega_a = 40.0
    freq_21 = omega_a / (2 * math.pi)
    gamma_21 = (2 * 4.0) / (2 * math.pi)
    t1 = mp.Transition(
        1,
        2,
        pumping_rate=0.0051,
        frequency=freq_21,
        gamma=gamma_21,
        sigma_diag=mp.Vector3(2 * omega_a, 2 * omega_a, 2 * omega_a),
    )
    t2 = mp.Transition(2, 1, transition_rate=0.005)
    return [mp.MultilevelAtom(sigma=1, transitions=[t1, t2], initial_populations=[28])]


# --- the matrix ------------------------------------------------------------
#
# Each entry is a plain dict consumed by build().  Keys not listed default as
# documented in build().  `steps` is the list of timestep counts at which state
# is dumped; the run advances cumulatively through them.

CONFIGS = [
    # ---- D1 ---------------------------------------------------------------
    dict(
        name="d1_vacuum_pml_real",
        dims=1,
        boundaries="pml",
        steps=[0, 25, 60],
    ),
    dict(
        name="d1_lorentzian_pml_complex",
        dims=1,
        boundaries="pml",
        complex_fields=True,
        eps=2.5,
        susceptibilities=_lorentzian,
        steps=[0, 25, 60],
    ),
    dict(
        name="d1_conductivity_absorber",
        dims=1,
        boundaries="absorber",
        eps=4.0,
        conductivity=0.3,
        steps=[0, 25, 60],
    ),
    dict(
        name="d1_multilevel_metallic",
        dims=1,
        boundaries="none",
        eps=2.25,
        susceptibilities=_multilevel,
        steps=[0, 20, 45],
    ),
    # ---- D2 ---------------------------------------------------------------
    dict(
        name="d2_isotropic_pml",
        dims=2,
        boundaries="pml",
        eps=12.0,
        monitors="flux",
        steps=[0, 20, 45],
    ),
    dict(
        name="d2_anisotropic_pml",
        dims=2,
        boundaries="pml",
        eps_diag=mp.Vector3(9, 11, 13),
        eps_offdiag=mp.Vector3(0.4, 0.0, 0.0),
        steps=[0, 20, 45],
    ),
    dict(
        name="d2_bloch_periodic",
        dims=2,
        boundaries="bloch",
        k_point=mp.Vector3(0.31, 0.17),
        eps=6.0,
        complex_fields=True,
        steps=[0, 20, 45],
    ),
    dict(
        name="d2_mirror_symmetry",
        dims=2,
        boundaries="pml",
        symmetries=[mp.Mirror(mp.Y)],
        eps=8.0,
        monitors="flux",
        steps=[0, 20, 45],
    ),
    dict(
        name="d2_rotate4_symmetry",
        dims=2,
        boundaries="pml",
        symmetries=[mp.Rotate4(mp.Z)],
        eps=5.0,
        steps=[0, 20, 45],
    ),
    dict(
        name="d2_metallic_chi3",
        dims=2,
        boundaries="none",
        eps=4.0,
        chi3=0.02,
        steps=[0, 20, 45],
    ),
    dict(
        name="d2_chi2_absorber",
        dims=2,
        boundaries="absorber",
        eps=4.0,
        chi2=0.05,
        steps=[0, 20, 45],
    ),
    dict(
        name="d2_drude_near2far",
        dims=2,
        boundaries="pml",
        eps=3.0,
        susceptibilities=_drude,
        monitors="near2far",
        steps=[0, 20, 45],
    ),
    dict(
        name="d2_noisy_lorentzian",
        dims=2,
        boundaries="pml",
        eps=2.0,
        susceptibilities=_noisy,
        steps=[0, 20, 45],
    ),
    dict(
        name="d2_gyrotropic",
        dims=2,
        boundaries="pml",
        eps=2.0,
        susceptibilities=_gyrotropic,
        complex_fields=True,
        steps=[0, 20, 45],
    ),
    dict(
        name="d2_material_grid",
        dims=2,
        boundaries="pml",
        material_grid=True,
        monitors="flux",
        steps=[0, 20, 45],
    ),
    dict(
        name="d2_special_kz",
        dims=2,
        boundaries="pml",
        eps=6.0,
        k_point=mp.Vector3(z=0.4),
        kz_2d="complex",
        steps=[0, 20, 45],
    ),
    dict(
        name="d2_bfast",
        dims=2,
        boundaries="pml",
        eps=4.0,
        bfast_scaled_k=mp.Vector3(0.2, 0, 0),
        complex_fields=True,
        steps=[0, 20, 45],
    ),
    dict(
        name="d2_integrated_source",
        dims=2,
        boundaries="pml",
        eps=4.0,
        integrated_source=True,
        monitors="flux",
        steps=[0, 20, 45],
    ),
    dict(
        name="d2_volume_source_dft",
        dims=2,
        boundaries="pml",
        eps=4.0,
        source_size=mp.Vector3(0, 2),
        monitors="dft_fields",
        steps=[0, 20, 45],
    ),
    dict(
        name="d2_custom_source",
        dims=2,
        boundaries="pml",
        eps=4.0,
        custom_source=True,
        steps=[0, 20, 45],
    ),
    dict(
        name="d2_solve_cw",
        dims=2,
        boundaries="pml",
        eps=6.0,
        complex_fields=True,
        solver="cw",
        steps=[0],
    ),
    dict(
        name="d2_magnetic_sync",
        dims=2,
        boundaries="pml",
        eps=6.0,
        monitors="flux",
        synchronize=True,
        steps=[0, 20, 45],
    ),
    # ---- D3 ---------------------------------------------------------------
    dict(
        name="d3_isotropic_pml",
        dims=3,
        boundaries="pml",
        eps=6.0,
        steps=[0, 12, 25],
    ),
    dict(
        name="d3_mirror_lorentzian",
        dims=3,
        boundaries="pml",
        eps=2.0,
        susceptibilities=_lorentzian,
        symmetries=[mp.Mirror(mp.Y)],
        steps=[0, 12, 25],
    ),
    # ---- Dcyl -------------------------------------------------------------
    dict(
        name="dcyl_m0",
        dims="cyl",
        m=0,
        boundaries="pml",
        eps=9.0,
        steps=[0, 20, 45],
    ),
    dict(
        name="dcyl_m1",
        dims="cyl",
        m=1,
        boundaries="pml",
        eps=9.0,
        complex_fields=True,
        steps=[0, 20, 45],
    ),
    dict(
        # |m| > 1 exercises the origin rules; zero_fields_near_cylorigin itself
        # is not reachable from the Python Simulation constructor (see
        # README.md, "coverage gaps").
        name="dcyl_m3",
        dims="cyl",
        m=3,
        boundaries="pml",
        eps=9.0,
        complex_fields=True,
        steps=[0, 20, 45],
    ),
]

CONFIG_BY_NAME = {c["name"]: c for c in CONFIGS}


def _cell_and_geometry(cfg):
    dims = cfg["dims"]
    eps = cfg.get("eps", 1.0)

    if cfg.get("material_grid"):
        n = 8
        import numpy as np

        design = np.linspace(0.0, 1.0, n * n).reshape(n, n)
        grid = mp.MaterialGrid(
            mp.Vector3(n, n),
            mp.Medium(index=1.0),
            mp.Medium(index=3.5),
            weights=design,
            grid_type="U_MEAN",
        )
        material = grid
    elif "eps_diag" in cfg:
        material = mp.Medium(
            epsilon_diag=cfg["eps_diag"],
            epsilon_offdiag=cfg.get("eps_offdiag", mp.Vector3()),
        )
    else:
        kwargs = dict(epsilon=eps)
        if cfg.get("conductivity"):
            kwargs["D_conductivity"] = cfg["conductivity"]
        if cfg.get("chi2"):
            kwargs["chi2"] = cfg["chi2"]
        if cfg.get("chi3"):
            kwargs["chi3"] = cfg["chi3"]
        if cfg.get("susceptibilities"):
            kwargs["E_susceptibilities"] = cfg["susceptibilities"]()
        material = mp.Medium(**kwargs)

    if dims == 1:
        return mp.Vector3(0, 0, 8), [
            mp.Block(mp.Vector3(mp.inf, mp.inf, 3), center=mp.Vector3(), material=material)
        ]
    if dims == 2:
        return mp.Vector3(6, 6), [
            mp.Block(mp.Vector3(2, 2, mp.inf), center=mp.Vector3(), material=material)
        ]
    if dims == 3:
        return mp.Vector3(4, 4, 4), [
            mp.Sphere(radius=1.0, center=mp.Vector3(), material=material)
        ]
    if dims == "cyl":
        return mp.Vector3(4, 0, 6), [
            mp.Block(
                mp.Vector3(1.5, mp.inf, 2),
                center=mp.Vector3(1.0),
                material=material,
            )
        ]
    raise ValueError(f"unknown dims {dims!r}")


def _resolution(cfg):
    return {
        1: RESOLUTION_1D,
        2: RESOLUTION_2D,
        3: RESOLUTION_3D,
        "cyl": RESOLUTION_CYL,
    }[cfg["dims"]]


def _boundary_layers(cfg):
    kind = cfg.get("boundaries", "pml")
    if kind == "pml":
        return [mp.PML(1.0)]
    if kind == "absorber":
        return [mp.Absorber(1.0)]
    # "none" leaves the default metallic walls; "bloch" is periodic via k_point.
    return []


def build(cfg):
    """Return an un-initialized ``mp.Simulation`` for ``cfg``."""
    cell, geometry = _cell_and_geometry(cfg)
    dims = cfg["dims"]

    # A 1d cell propagates along z with Ex/Hy, so Ez does not exist there.
    src_component = mp.Ex if dims == 1 else mp.Ez
    if dims == "cyl" and cfg.get("m", 0) != 0:
        src_component = mp.Er
    src_center = mp.Vector3(1.0) if dims == "cyl" else mp.Vector3()

    source = _src(
        component=src_component,
        center=src_center,
        size=cfg.get("source_size"),
        integrated=cfg.get("integrated_source", False),
    )
    if cfg.get("custom_source"):
        source = mp.Source(
            src=mp.CustomSource(
                src_func=lambda t: math.exp(-((t - 3.0) ** 2)) * math.cos(2 * math.pi * 0.3 * t),
                end_time=12.0,
            ),
            component=src_component,
            center=src_center,
        )

    kwargs = dict(
        cell_size=cell,
        resolution=_resolution(cfg),
        geometry=geometry,
        sources=[source],
        boundary_layers=_boundary_layers(cfg),
        # --- determinism pins ---
        split_chunks_evenly=True,
        num_chunks=cfg.get("num_chunks", 0),
        loop_tile_base_db=LOOP_TILE_BASE_DB,
        loop_tile_base_eh=LOOP_TILE_BASE_EH,
        force_complex_fields=cfg.get("complex_fields", False),
        eps_averaging=cfg.get("eps_averaging", True),
    )
    if cfg.get("symmetries"):
        kwargs["symmetries"] = cfg["symmetries"]
    if cfg.get("k_point") is not None:
        kwargs["k_point"] = cfg["k_point"]
    if cfg.get("kz_2d"):
        kwargs["kz_2d"] = cfg["kz_2d"]
    if cfg.get("bfast_scaled_k") is not None:
        kwargs["bfast_scaled_k"] = cfg["bfast_scaled_k"]
    if dims == "cyl":
        kwargs["dimensions"] = mp.CYLINDRICAL
        kwargs["m"] = cfg.get("m", 0)
    elif dims == 1:
        kwargs["dimensions"] = 1
    return mp.Simulation(**kwargs)


def add_monitors(sim, cfg):
    """Attach monitors and return a dict of named reduction callables."""
    kind = cfg.get("monitors")
    reductions = {}
    if kind == "flux":
        fr = mp.FluxRegion(center=mp.Vector3(1.5), size=mp.Vector3(0, 2))
        flux = sim.add_flux(0.3, 0.2, 3, fr)
        reductions["flux"] = lambda: list(mp.get_fluxes(flux))
    elif kind == "near2far":
        n2f = sim.add_near2far(
            0.3, 0, 1, mp.Near2FarRegion(center=mp.Vector3(1.5), size=mp.Vector3(0, 2))
        )
        reductions["near2far_norm"] = lambda: [sim.get_dft_array(n2f, mp.Ez, 0).sum().real]
    elif kind == "dft_fields":
        dft = sim.add_dft_fields([mp.Ez], 0.3, 0, 1, where=mp.Volume(size=mp.Vector3(2, 2)))
        reductions["dft_fields_norm"] = lambda: [
            float(abs(sim.get_dft_array(dft, mp.Ez, 0)).sum())
        ]
    return reductions
