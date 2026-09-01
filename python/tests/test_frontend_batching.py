import os
import signal
import threading
import time
import unittest

import numpy as np

import meep as mp
import meep.simulation as simulation_module
from meep.simulation import (
    Simulation,
    _first_step_at_or_after_time,
    after_time,
    at_beginning,
    at_end,
    at_every,
    at_time,
    before_time,
    combine_step_funcs,
    display_progress,
    during_sources,
    synchronized_magnetic,
)


class FakeFields:
    def __init__(self, dt=0.25, last_source_time=1.0):
        self.dt = dt
        self.t = 0
        self._last_source_time = last_source_time
        self.calls = []
        self.magnetic = []

    def time(self):
        return self.t * self.dt

    def round_time(self):
        return float(np.float32(self.t * self.dt))

    def last_source_time(self):
        return self._last_source_time

    def step(self):
        self.calls.append(1)
        self.t += 1

    def advance(self, count):
        self.calls.append(count)
        self.t += count

    def synchronize_magnetic_fields(self):
        self.magnetic.append(("sync", self.t))

    def restore_magnetic_fields(self):
        self.magnetic.append(("restore", self.t))


class FakeSimulation:
    _run_until = Simulation._run_until
    round_time = Simulation.round_time
    meep_time = Simulation.meep_time

    def __init__(self, dt=0.25, sources=()):
        self.fields = FakeFields(dt=dt)
        self.sources = list(sources)
        self.progress_interval = 4
        self.progress = False
        self.interactive = True
        self.run_index = 0


def recorder(name, trace):
    def _record(sim, todo):
        trace.append((sim.fields.t, sim.round_time(), todo, name))

    return _record


SERIAL_ONLY = unittest.skipIf(
    mp.count_processors() > 1, "frontend batching is deliberately disabled under MPI"
)


class RecordingFields:
    def __init__(self, fields, frontend_dt=None):
        self._fields = fields
        self._frontend_dt = frontend_dt
        self.calls = []

    def __getattr__(self, name):
        return getattr(self._fields, name)

    @property
    def dt(self):
        return self._fields.dt if self._frontend_dt is None else self._frontend_dt

    def step(self):
        self.calls.append(1)
        return self._fields.step()

    def advance(self, count):
        self.calls.append(count)
        return self._fields.advance(count)


class TestFrontendBatching(unittest.TestCase):
    def setUp(self):
        self.old_verbosity = simulation_module.verbosity.meep
        self.old_progress = simulation_module.do_progress
        simulation_module.verbosity.meep = 0
        simulation_module.do_progress = False

    def tearDown(self):
        simulation_module.verbosity.meep = self.old_verbosity
        simulation_module.do_progress = self.old_progress

    @SERIAL_ONLY
    def test_numeric_run_uses_one_batch(self):
        sim = FakeSimulation()
        sim._run_until(2.0, [])
        self.assertEqual(sim.fields.t, 8)
        self.assertEqual(sim.fields.calls, [8])

    @SERIAL_ONLY
    def test_event_trace_matches_forced_one_step(self):
        def run(force_one_step):
            sim = FakeSimulation()
            trace = []
            scheduled = combine_step_funcs(
                at_beginning(recorder("begin", trace)),
                at_time(0.75, recorder("time", trace)),
                at_every(0.5, recorder("every", trace)),
                at_end(recorder("end", trace)),
            )
            funcs = [scheduled]
            if force_one_step:
                funcs.append(lambda _sim: None)
            sim._run_until(2.0, funcs)
            return sim, trace

        batched, batched_trace = run(False)
        stepped, stepped_trace = run(True)
        self.assertEqual(batched_trace, stepped_trace)
        self.assertEqual(batched.fields.t, stepped.fields.t)
        self.assertTrue(any(count > 1 for count in batched.fields.calls))
        self.assertTrue(all(count == 1 for count in stepped.fields.calls))

    @SERIAL_ONLY
    def test_active_interval_wrappers_remain_per_step(self):
        before = FakeSimulation()
        before._run_until(2.0, [before_time(0.75, lambda _sim: None)])
        self.assertEqual(before.fields.calls[:3], [1, 1, 1])
        self.assertGreater(before.fields.calls[-1], 1)

        after = FakeSimulation()
        after._run_until(2.0, [after_time(0.75, lambda _sim: None)])
        self.assertEqual(after.fields.calls[0], 3)
        self.assertTrue(all(count == 1 for count in after.fields.calls[1:]))

        during = FakeSimulation()
        during._run_until(2.0, [during_sources(lambda _sim: None)])
        self.assertEqual(during.fields.calls[:4], [1, 1, 1, 1])
        self.assertGreater(during.fields.calls[-1], 1)

    def test_dynamic_callback_progress_sync_and_custom_source_force_steps(self):
        cases = []
        cases.append(
            (FakeSimulation(), [lambda _sim: None], lambda sim: sim.fields.t >= 5)
        )
        cases.append((FakeSimulation(), [display_progress(0, 2, 1000)], 1.25))
        cases.append(
            (FakeSimulation(), [synchronized_magnetic(lambda _sim: None)], 1.25)
        )

        class CustomTime:
            def src_func(self, _time):
                return 0

        class Source:
            src = CustomTime()

        cases.append((FakeSimulation(sources=[Source()]), [], 1.25))

        for sim, funcs, condition in cases:
            sim._run_until(condition, funcs)
            self.assertTrue(sim.fields.calls)
            self.assertTrue(all(count == 1 for count in sim.fields.calls))

    @SERIAL_ONLY
    def test_exception_occurs_at_exact_event_before_advance(self):
        sim = FakeSimulation()

        def fail(_sim):
            raise RuntimeError("scheduled failure")

        with self.assertRaisesRegex(RuntimeError, "scheduled failure"):
            sim._run_until(2.0, [at_time(0.75, fail)])
        self.assertEqual(sim.fields.t, 3)
        self.assertEqual(sim.fields.calls, [3])

    @SERIAL_ONLY
    def test_time_boundary_recomputes_after_scheduled_dt_change(self):
        sim = FakeSimulation()

        def change_dt(simulation):
            simulation.fields.dt = 0.5

        sim._run_until(2.0, [at_time(0.5, change_dt)])
        self.assertEqual(sim.fields.calls, [2, 2])
        self.assertEqual(sim.fields.t, 4)
        self.assertEqual(sim.round_time(), 2.0)

    @SERIAL_ONLY
    def test_fractional_boundaries_match_step_oracle(self):
        for dt, stop, event, cadence in (
            (0.1, 1.03, 0.31, 0.27),
            (0.125, 2.01, 0.62, 0.41),
            (0.3, 3.2, 0.91, 0.77),
        ):
            traces = []
            calls = []
            for force_one_step in (False, True):
                sim = FakeSimulation(dt=dt)
                trace = []
                scheduled = combine_step_funcs(
                    at_time(event, recorder("time", trace)),
                    at_every(cadence, recorder("every", trace)),
                )
                funcs = (
                    [scheduled, lambda _sim: None] if force_one_step else [scheduled]
                )
                sim._run_until(stop, funcs)
                traces.append(trace)
                calls.append(list(sim.fields.calls))
            self.assertEqual(traces[0], traces[1])
            self.assertTrue(any(count > 1 for count in calls[0]))
            self.assertTrue(all(count == 1 for count in calls[1]))

    @SERIAL_ONLY
    def test_rounding_boundary_and_small_batches(self):
        sim = FakeSimulation(dt=0.1)
        for target in (0.0, 0.1, 0.2, 0.3, 100.0):
            step = _first_step_at_or_after_time(sim, target)
            self.assertGreaterEqual(float(np.float32(step * sim.fields.dt)), target)
            if step > sim.fields.t:
                self.assertLess(float(np.float32((step - 1) * sim.fields.dt)), target)
        self.assertIsNone(_first_step_at_or_after_time(sim, float("inf")))
        self.assertIsNone(_first_step_at_or_after_time(sim, float("nan")))
        self.assertIsNone(_first_step_at_or_after_time(sim, 1.0e300))

        zero = FakeSimulation()
        zero._run_until(0, [])
        self.assertEqual(zero.fields.calls, [])

        one = FakeSimulation()
        one._run_until(0.25, [])
        self.assertEqual(one.fields.calls, [1])

        two = FakeSimulation()
        two._run_until(0.5, [])
        self.assertEqual(two.fields.calls, [2])

    @SERIAL_ONLY
    def test_numeric_batches_are_bounded(self):
        sim = FakeSimulation()
        steps = 3 * simulation_module._FRONTEND_MAX_BATCH_STEPS + 17
        sim._run_until(steps * sim.fields.dt, [])
        self.assertEqual(sim.fields.t, steps)
        self.assertTrue(
            all(
                1 <= count <= simulation_module._FRONTEND_MAX_BATCH_STEPS
                for count in sim.fields.calls
            )
        )
        self.assertEqual(sim.fields.calls[-1], 17)

    @SERIAL_ONLY
    def test_native_sigint_latency_is_bounded(self):
        sim = Simulation(cell_size=mp.Vector3(0, 0, 2), dimensions=1, resolution=20)
        sim.init_sim()
        finished = threading.Event()

        def send_sigint():
            if not finished.wait(0.02):
                os.kill(os.getpid(), signal.SIGINT)

        old_handler = signal.signal(signal.SIGINT, signal.default_int_handler)
        interrupter = threading.Thread(target=send_sigint, daemon=True)
        started = time.monotonic()
        interrupter.start()
        try:
            with self.assertRaises(KeyboardInterrupt):
                sim._run_until(1.0e7, [])
            self.assertLess(time.monotonic() - started, 2.0)
        finally:
            finished.set()
            interrupter.join()
            signal.signal(signal.SIGINT, old_handler)
            sim.reset_meep()

    @SERIAL_ONLY
    def test_native_cpu_state_matches_forced_step_loop(self):

        def run(force_one_step):
            sim = Simulation(
                cell_size=mp.Vector3(0, 0, 2),
                dimensions=1,
                resolution=10,
                sources=[
                    mp.Source(
                        mp.GaussianSource(frequency=0.4, fwidth=0.2),
                        component=mp.Ex,
                        center=mp.Vector3(),
                    )
                ],
            )
            sim.init_sim()
            self.assertFalse(hasattr(sim.fields, "frontend_step_at_or_after"))
            live_fields = sim.fields
            recorded = RecordingFields(live_fields)
            sim.fields = recorded
            funcs = [lambda _sim: None] if force_one_step else []
            sim._run_until(1.0, funcs)
            result = (live_fields.t, sim.get_field_point(mp.Ex, mp.Vector3()))
            calls = list(recorded.calls)
            sim.fields = live_fields
            sim.reset_meep()
            return result, calls

        batched, batch_calls = run(False)
        stepped, step_calls = run(True)
        self.assertEqual(batched[0], stepped[0])
        self.assertEqual(batched[1], stepped[1])
        self.assertTrue(any(count > 1 for count in batch_calls))
        self.assertTrue(all(count == 1 for count in step_calls))

    @unittest.skipUnless(mp.count_processors() > 1, "MPI-specific batching test")
    def test_mpi_rank_divergent_frontend_facts_force_identical_steps(self):
        rank = mp.my_rank()
        sim = Simulation(
            cell_size=mp.Vector3(0, 0, 2),
            dimensions=1,
            resolution=10,
            sources=[
                mp.Source(
                    mp.GaussianSource(frequency=0.4, fwidth=0.2),
                    component=mp.Ex,
                    center=mp.Vector3(),
                )
            ],
        )
        sim.init_sim()
        self.assertFalse(hasattr(sim.fields, "frontend_step_at_or_after"))
        live_fields = sim.fields
        recorded = RecordingFields(
            live_fields, frontend_dt=live_fields.dt * (1.0 + 0.25 * rank)
        )
        sim.fields = recorded

        class RankLocalSource:
            src = object()

        if rank == 0:
            sim.sources.append(RankLocalSource())

        callback_hits = []
        divergent_callback = at_time(
            0.25 + 0.125 * rank,
            lambda current: callback_hits.append(current.fields.t),
        )
        try:
            sim._run_until(1.0, [divergent_callback])
            call_count = len(recorded.calls)
            self.assertTrue(recorded.calls)
            self.assertTrue(all(count == 1 for count in recorded.calls))
            self.assertEqual(mp.min_to_all(call_count), call_count)
            self.assertEqual(mp.max_to_all(call_count), call_count)
            self.assertTrue(callback_hits)
        finally:
            sim.fields = live_fields
            sim.reset_meep()


if __name__ == "__main__":
    unittest.main()
