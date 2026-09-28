"""Parameterized receptor networks and evidence-only ABFE qualification."""
import csv
from pathlib import Path

import mdtraj as md
import numpy as np
import pytest

from felis_workflows.common import WorkflowError, read, write
from felis_workflows.config import site_config
from felis_workflows.planning import workdir
from felis_endpoint import qualification
from felis_endpoint.networks import analyze, frames_for
from felis_endpoint.qualification import qualify


def network_inputs(tmp_path):
    topology = md.Topology()
    chain = topology.add_chain()
    ligand = topology.add_residue("XYZ", chain, resSeq=10)
    topology.add_atom("O1", md.element.oxygen, ligand)
    topology.add_atom("H1", md.element.hydrogen, ligand)
    site = topology.add_residue("GLU", chain, resSeq=42)
    topology.add_atom("OE1", md.element.oxygen, site)
    water = topology.add_residue("HOH", chain, resSeq=99)
    topology.add_atom("O", md.element.oxygen, water)
    topology.add_atom("H2", md.element.hydrogen, water)
    topology.add_atom("H3", md.element.hydrogen, water)
    positions = np.array([[.5, .5, .5], [.6, .5, .5], [1., .5, .5],
                          [.75, .5, .5], [.85, .5, .5], [.65, .5, .5]], dtype=np.float32)
    xyz = np.stack((positions, positions.copy()))
    xyz[1, 3:, 0] += .7  # The second cluster loses the single-water bridge.
    trajectory = md.Trajectory(xyz, topology, time=[0, 50],
                               unitcell_lengths=np.ones((2, 3), dtype=np.float32) * 3,
                               unitcell_angles=np.ones((2, 3), dtype=np.float32) * 90)
    pdb, dcd = tmp_path / "network.pdb", tmp_path / "network.dcd"
    trajectory[0].save_pdb(str(pdb))
    trajectory.save_dcd(str(dcd))
    clusters = tmp_path / "clusters.tsv"
    clusters.write_text("frame\tcluster\n0\tnear\n1\tfar\n")
    config = {"schema_version": 1,
              "ligand": {"resname": "XYZ", "resid": 10, "acceptors": ["O1"], "donors": ["O1"]},
              "sites": [{"id": "acid-site", "resname": "GLU", "resid": 42, "acceptors": ["OE1"],
                         "donors": [], "water_bridge": True}],
              "water_resnames": ["HOH"],
              "loose": {"distance_A": 3.5, "angle_deg": 135},
              "strict": {"distance_A": 3.2, "angle_deg": 150},
              "trajectories": [{"replica": "replica-A", "trajectory": str(dcd), "topology": str(pdb),
                                "frames": {"indices": [0, 1]}, "cluster_tsv": str(clusters)}]}
    path = tmp_path / "network.yaml"
    write(path, config)
    return path, config, dcd


def tsv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_explicit_network_sites_hbond_thresholds_and_replica_cluster_outputs(tmp_path):
    config, _, _ = network_inputs(tmp_path)
    output = tmp_path / "analysis"
    result = analyze(config, output)
    assert result["frames"] == 2
    geometry = tsv(output / "endpoint_key_residue_geometry.tsv")
    bridge = tsv(output / "endpoint_water_bridge_frames.tsv")
    summary = tsv(output / "endpoint_network_by_replica_cluster.tsv")
    assert {row["site"] for row in geometry} == {"acid-site"}
    assert {(row["replica"], row["cluster"]) for row in summary} == {
        ("replica-A", "near"), ("replica-A", "far")}
    assert {row["cluster"]: int(row["bridges"]) for row in bridge} == {"near": 1, "far": 0}
    assert int(tsv(output / "endpoint_water_bridges.tsv")[0]["strict"]) == 1
    assert read(output / "analysis_provenance.json")["sites"][0]["resid"] == 42


def test_network_selection_is_explicit_and_no_fixed_sucrose_residue(tmp_path):
    path, value, _ = network_inputs(tmp_path)
    value["sites"][0]["resid"] = 120
    write(path, value)
    with pytest.raises(WorkflowError, match="exactly one residue"):
        analyze(path, tmp_path / "missing-site")
    assert frames_for({"start": 3, "stop": 10, "stride": 3}) == [3, 6, 9]
    with pytest.raises(WorkflowError, match="Duplicate"):
        frames_for({"indices": [1, 1]})


def test_network_time_window_selects_explicit_trajectory_frames(tmp_path):
    path, value, _ = network_inputs(tmp_path)
    value["trajectories"][0]["time_range_ns"] = {"start": 0.01, "stop": 0.1}
    value["trajectories"][0]["frame_interval_ps"] = 50
    write(path, value)
    result = analyze(path, tmp_path / "selected")
    assert result["frames"] == 1
    assert {row["cluster"] for row in tsv(tmp_path / "selected/endpoint_key_residue_geometry.tsv")} == {"far"}


def test_water_bridge_schema_requires_site_acceptor(tmp_path):
    path, config, _ = network_inputs(tmp_path)
    config["sites"][0]["acceptors"] = []
    config["sites"][0]["donors"] = ["OE1"]
    write(path, config)
    with pytest.raises(WorkflowError, match="requires site acceptor atoms; site-donor bridges are unsupported"):
        analyze(path, tmp_path / "donor-only")
    config["sites"][0]["water_bridge"] = False
    write(path, config)
    assert analyze(path, tmp_path / "direct-donor")["frames"] == 2


def test_qualification_reports_expected_coverage_and_native_tables(cycle, site, tmp_path):
    root, science = cycle
    first = science["calculations"][0]
    analysis = workdir(root, first) / "analysis"
    analysis.mkdir(parents=True)
    for name in ("A_converge_table.tsv", "B_converge_table.tsv"):
        (analysis / name).write_text("DeltaG\tBlock-1(+)\tBlock-1(-)\nTotal\t-1.0\t-0.8\n")
    destination = tmp_path / "qualification.json"
    report = qualify(root, site_config(site), destination)
    assert report["expected_calculations"] == len(science["calculations"])
    assert report["expected_replicas"] == 3
    assert set(report["missing_calculations"]) == {c["key"] for c in science["calculations"]}
    assert report["calculations"][first["key"]]["groups"]["A"]["expected"] == 25
    native = report["calculations"][first["key"]]["native_analysis"]
    assert native["A_converge_table.tsv"]["rows"][0]["Block-1(+)"] == "-1.0"
    assert native["B_converge_table.tsv"]["state"] == "present"
    assert native["sys_abfe.tsv"]["state"] == "missing"
    assert "not automatically classified" in report["scientific_convergence"]
    assert read(destination) == report


def test_replica_qualification_requires_every_calculation_and_table(cycle, site, monkeypatch):
    root, science = cycle
    calculations = science["calculations"]
    assert len([calc for calc in calculations if calc["replica"] == 1]) >= 2
    entries = {calc["key"]: {"calculation": calc["key"], "finalization": "complete", "state": "complete"}
               for calc in calculations}
    monkeypatch.setattr(qualification, "abfe_status",
                        lambda _: {"calculations": list(entries.values()), "global_inputs": "complete"})
    for calc in calculations:
        directory = workdir(root, calc) / "analysis"
        directory.mkdir(parents=True, exist_ok=True)
        for name in (*qualification.NATIVE, *qualification.CONVERGENCE):
            (directory / name).write_text("quantity\tvalue\nTotal\t-1.0\n")
    incomplete = next(calc for calc in calculations if calc["replica"] == 1)
    (workdir(root, incomplete) / "analysis" / qualification.CONVERGENCE[0]).unlink()
    report = qualify(root, site_config(site))
    assert report["completed_replicas"] == [2, 3]
    assert not report["calculations"][incomplete["key"]]["terminal_qualified"]
    assert all(report["calculations"][c["key"]]["terminal_qualified"]
               for c in calculations if c["replica"] == 1 and c != incomplete)
    unfinished = next(calc for calc in calculations if calc["replica"] == 2)
    entries[unfinished["key"]]["finalization"] = "partial"
    assert qualify(root, site_config(site))["completed_replicas"] == [3]
