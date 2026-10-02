"""Group-only adapter: commit initialization before production on every rank."""
from pathlib import Path
import argparse
import os

from . import initialization
from .artifacts import group_dependencies, runtime_snapshot
from .common import WorkflowError
from .planning import calculation, load_run, units
from .runtime import activate_source


def guarded_factory(factory, restore, comm, root, science, calc, leg, unit, dependencies, runtime):
    def create_or_restore(thermo_states, pos, box, gk):
        _, _, _, nc = initialization.locations(root, science, calc, leg, unit)
        actual = Path(gk.dir.trj) / f"{gk.filename.stem}.nc"
        if actual.resolve() != nc.resolve():
            raise WorkflowError("Native sampler storage differs from its group initialization identity")
        mode = None
        if comm.mpi_rank == 0:
            current = initialization.state(root, science, calc, leg, unit, dependencies, runtime)
            if nc.with_suffix(".create_done").is_file():
                mode = "restore"
            elif nc.exists() or nc.with_name(nc.stem + "_checkpoint.nc").exists():
                raise WorkflowError("Uncommitted initialization files were not quarantined before launch")
            else:
                if current != "pending":
                    raise WorkflowError("New sampler initialization requires a recorded group intent")
                mode = "create"
        mode = comm.bcast(mode, root=0)
        # Bypass the native NetCDF-error delete/recreate fallback for a marked
        # trajectory. Restoration errors must preserve the checkpoint lineage.
        sampler = restore(str(nc)) if mode == "restore" else factory(thermo_states, pos, box, gk)
        if comm.mpi_rank == 0:
            initialization.commit_ready(root, science, calc, leg, unit, dependencies, runtime)
        comm.barrier()  # No rank may enter sampler.run() before ready is durable.
        return sampler
    return create_or_restore


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--calculation", required=True)
    parser.add_argument("--leg", choices=["A", "B"], required=True)
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args(argv)
    activate_source({"repo": os.environ["FELIS_REPO"]})
    root = args.run.resolve()
    science = load_run(root, prepared=True)
    calc = calculation(science, args.calculation)
    if args.leg == "A" and calc["solvent_owner"] != calc["key"]:
        raise WorkflowError("A shared solvent calculation must be run through its owner")
    unit = units(root, science, calc, args.leg)[args.index]
    dependencies = group_dependencies(root, science, calc, args.leg, args.index)
    runtime = runtime_snapshot(root)
    from felis.app.dyn.repex import mainfunc
    from felis.protocols.dyn import main_replica_exchange as protocol
    from felis.utils.mpi_tools import mpicomm
    from felis.utils.omm.omm_tools import ReplicaExchangeSampler
    original = protocol.get_replica_exchange_sampler
    protocol.get_replica_exchange_sampler = guarded_factory(
        original, ReplicaExchangeSampler.from_storage, mpicomm,
        root, science, calc, args.leg, unit, dependencies, runtime)
    try:
        mainfunc(unit["argv"][2:], protocol.mainfunc)
    finally:
        protocol.get_replica_exchange_sampler = original


if __name__ == "__main__":
    main()
