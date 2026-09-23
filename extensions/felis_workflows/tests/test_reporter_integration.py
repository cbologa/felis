"""Optional CPU integration test against real OpenMMTools NetCDF storage."""
import pytest


def test_real_reporter_partial_resume_complete(cycle):
    omt = pytest.importorskip("openmmtools")
    import openmm as mm
    from openmm import unit as u
    from felis_workflows.planning import workdir
    from felis_workflows.validation import iteration_status
    root, science = cycle
    calc = science["calculations"][0]
    directory = workdir(root, calc) / "trj"
    directory.mkdir(parents=True)
    storage = directory / "b0.nc"
    omt.cache.global_context_cache.platform = mm.Platform.getPlatformByName("Reference")
    system = mm.System()
    system.addParticle(12)
    potential = mm.CustomExternalForce("10*(x*x+y*y+z*z)")
    potential.addParticle(0, [])
    system.addForce(potential)
    states = [omt.states.ThermodynamicState(system, temperature=t*u.kelvin) for t in [298.15, 300.0]]
    sampler = omt.multistate.ReplicaExchangeSampler(
        mcmc_moves=omt.mcmc.LangevinDynamicsMove(n_steps=1),
        number_of_iterations=2, online_analysis_interval=None)
    reporter = omt.multistate.MultiStateReporter(str(storage), checkpoint_interval=1)
    sampler.create(states, omt.states.SamplerState(positions=[[.01, .01, .01]]*u.nanometer), storage=reporter)
    storage.with_suffix(".create_done").write_text("created")
    sampler.run(n_iterations=1)
    reporter.close()
    task = {"stem": "b0", "iterations": 2, "checkpoint_interval": 1, "ilam": [0, 1]}
    assert not iteration_status(root, science, calc, "B", task)["complete"]
    resumed = omt.multistate.ReplicaExchangeSampler.from_storage(str(storage))
    resumed.run()
    resumed._reporter.close()
    assert iteration_status(root, science, calc, "B", task)["complete"]
    omt.cache.global_context_cache.empty()
