"""Configurable direct H bonds and single-water bridges from endpoint trajectories.

The four detailed TSVs retain the exploratory analysis layout; the inputs and
receptor network are supplied by the investigator for each analysis.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import math
from pathlib import Path
import statistics

from felis_workflows.common import WorkflowError, identifier, keys, read, sha256, write

from .plan import finite_positive, path_from_config


def frames_for(value):
    keys(value, {"indices", "start", "stop", "stride"}, (), "frame selection")
    if "indices" in value:
        if len(value) != 1 or not isinstance(value["indices"], list) or not value["indices"] or \
                any(type(i) is not int or i < 0 for i in value["indices"]):
            raise WorkflowError("Frame indices must be explicit nonnegative integers")
        indices = value["indices"]
    else:
        if set(value) != {"start", "stop", "stride"} or any(type(value[k]) is not int for k in value) or \
                value["start"] < 0 or value["stop"] <= value["start"] or value["stride"] <= 0:
            raise WorkflowError("Frame range needs start, exclusive stop, and positive stride")
        indices = list(range(value["start"], value["stop"], value["stride"]))
    if len(indices) != len(set(indices)):
        raise WorkflowError("Duplicate frame indices")
    return indices


def threshold(value, label):
    keys(value, {"distance_A", "angle_deg"}, {"distance_A", "angle_deg"}, label)
    distance = finite_positive(value["distance_A"], f"{label}.distance_A")
    angle = finite_positive(value["angle_deg"], f"{label}.angle_deg")
    if angle >= 180:
        raise WorkflowError(f"{label}.angle_deg must be below 180")
    return distance, angle


def residue(topology, spec, label):
    keys(spec, {"resname", "resid", "chain_index", "donors", "acceptors", "id", "water_bridge"},
         {"resname"}, label)
    matches = [r for r in topology.residues if r.name == spec["resname"] and
               ("resid" not in spec or r.resSeq == spec["resid"]) and
               ("chain_index" not in spec or r.chain.index == spec["chain_index"])]
    if len(matches) != 1:
        raise WorkflowError(f"{label} must select exactly one residue; found {len(matches)}")
    return matches[0]


def selected_atoms(residue_value, names, label):
    if not isinstance(names, list) or any(not isinstance(n, str) for n in names):
        raise WorkflowError(f"{label} requires explicit atom names")
    by_name = {atom.name: atom.index for atom in residue_value.atoms}
    if len(by_name) != len(list(residue_value.atoms)) or any(name not in by_name for name in names):
        raise WorkflowError(f"{label} atom names are missing or ambiguous")
    return [by_name[name] for name in names]


def near_h(frame, donor, residue_value):
    import mdtraj as md
    hydrogens = [a.index for a in residue_value.atoms if a.element and a.element.symbol == "H"]
    if not hydrogens:
        return []
    distances = md.compute_distances(frame, [(donor, h) for h in hydrogens], periodic=True)[0] * 10
    return [h for h, distance in zip(hydrogens, distances) if distance <= 1.3]


def geometry(frame, donor, hydrogen, acceptor):
    import mdtraj as md
    distance = float(md.compute_distances(frame, [(donor, acceptor)], periodic=True)[0, 0]) * 10
    angle = math.degrees(float(md.compute_angles(frame, [(donor, hydrogen, acceptor)], periodic=True)[0, 0]))
    return distance, angle


def events(frame, donors, acceptors, loose, strict, direction):
    result = []
    for donor, hydrogens in donors.items():
        for hydrogen in hydrogens:
            for acceptor in acceptors:
                dist, angle = geometry(frame, donor, hydrogen, acceptor)
                if dist <= loose[0] and angle >= loose[1]:
                    result.append((direction, donor, hydrogen, acceptor, dist, angle,
                                   int(dist <= strict[0] and angle >= strict[1])))
    return result


def write_tsv(path, columns, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(columns)
        writer.writerows(rows)


def cluster_map(path):
    if path is None:
        return {}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if any("frame" not in r or "cluster" not in r for r in rows):
        raise WorkflowError("Cluster table needs frame and cluster columns")
    result = {int(r["frame"]): r["cluster"] for r in rows}
    if len(result) != len(rows):
        raise WorkflowError("Duplicate frame in cluster table")
    return result


def analyze(config_path, output, *, runtime=None):
    """Water bridges support water-donor to site-acceptor geometry only."""
    import mdtraj as md
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config = read(config_path)
    keys(config, {"schema_version", "ligand", "sites", "water_resnames", "loose", "strict", "trajectories"},
         {"schema_version", "ligand", "sites", "water_resnames", "loose", "strict", "trajectories"}, "network analysis")
    if config["schema_version"] != 1 or not isinstance(config["sites"], list) or not config["sites"] or \
            not isinstance(config["trajectories"], list) or not config["trajectories"]:
        raise WorkflowError("Network analysis needs sites and trajectories")
    loose, strict = threshold(config["loose"], "loose"), threshold(config["strict"], "strict")
    if strict[0] > loose[0] or strict[1] < loose[1]:
        raise WorkflowError("Strict H-bond geometry must be narrower than loose geometry")
    aliases = config["water_resnames"]
    if not isinstance(aliases, list) or not aliases or any(not isinstance(n, str) for n in aliases):
        raise WorkflowError("Supply explicit water residue aliases")
    if output.exists():
        raise WorkflowError(f"Use a new network analysis output directory: {output}")
    if output.is_relative_to(Path(__file__).resolve().parents[4]):
        raise WorkflowError("Generated network analysis must be outside the Git checkout")
    if any(not isinstance(spec, dict) or not isinstance(spec.get("id"), str) or
           type(spec.get("water_bridge", False)) is not bool for spec in config["sites"]) or \
            len({spec["id"] for spec in config["sites"]}) != len(config["sites"]):
        raise WorkflowError("Sites need distinct IDs and boolean water_bridge settings")
    output.mkdir(parents=True)
    frame_geometry, direct_events, bridge_frames, bridge_events = [], [], [], []
    inputs = []
    for source in config["trajectories"]:
        keys(source, {"replica", "trajectory", "topology", "frames", "frame_interval_ps", "time_range_ns", "cluster_tsv"},
             {"replica", "trajectory", "topology", "frames"}, "analysis trajectory")
        rep = identifier(source["replica"])
        trajectory = path_from_config(config_path.parent, source["trajectory"], "analysis trajectory")
        topology_path = path_from_config(config_path.parent, source["topology"], "analysis topology")
        clusters_path = path_from_config(config_path.parent, source["cluster_tsv"], "cluster TSV") if "cluster_tsv" in source else None
        selected = frames_for(source["frames"])
        frame_interval = finite_positive(source["frame_interval_ps"], "frame_interval_ps") if \
            "frame_interval_ps" in source else None
        time_range = source.get("time_range_ns")
        if time_range is not None:
            keys(time_range, {"start", "stop"}, {"start", "stop"}, "time_range_ns")
            if any(isinstance(time_range[k], bool) or not isinstance(time_range[k], (int, float)) or
                   not math.isfinite(time_range[k]) for k in ("start", "stop")) or \
                    time_range["start"] < 0 or time_range["stop"] <= time_range["start"]:
                raise WorkflowError("time_range_ns needs finite start >= 0 and stop > start")
        clusters = cluster_map(clusters_path)
        topology = md.load_topology(str(topology_path))
        lig = residue(topology, config["ligand"], "ligand")
        lig_acceptors = selected_atoms(lig, config["ligand"].get("acceptors"), "ligand acceptors")
        lig_donors = selected_atoms(lig, config["ligand"].get("donors"), "ligand donors")
        if not lig_acceptors and not lig_donors:
            raise WorkflowError("Ligand needs explicit polar donor or acceptor atoms")
        site_atoms = {}
        for spec in config["sites"]:
            identifier(spec["id"])
            keys(spec, {"id", "resname", "resid", "chain_index", "donors", "acceptors", "water_bridge"},
                 {"id", "resname", "donors", "acceptors"}, f'site {spec["id"]}')
            site_res = residue(topology, spec, f'site {spec["id"]}')
            site_atoms[spec["id"]] = (site_res,
                                     selected_atoms(site_res, spec["acceptors"], "site acceptors"),
                                     selected_atoms(site_res, spec["donors"], "site donors"),
                                     spec.get("water_bridge", False))
            if not site_atoms[spec["id"]][1] and not site_atoms[spec["id"]][2]:
                raise WorkflowError(f'Site {spec["id"]} has no polar atoms')
            if spec.get("water_bridge", False) and not site_atoms[spec["id"]][1]:
                raise WorkflowError(f'Water bridge site {spec["id"]} requires site acceptor atoms; site-donor bridges are unsupported')
        waters = []
        water_aliases = {alias.upper() for alias in aliases}
        for wat in topology.residues:
            if wat.name.upper() not in water_aliases:
                continue
            oxygen = [a.index for a in wat.atoms if a.element and a.element.symbol == "O"]
            if len(oxygen) == 1:
                waters.append((wat, oxygen[0]))
        if not waters:
            raise WorkflowError(f"No explicitly configured waters found in {topology_path}")
        inputs.append({"replica": rep, "trajectory": str(trajectory), "topology": str(topology_path),
                       "topology_sha256": sha256(topology_path), "frames": selected,
                       "time_range_ns": time_range, "frame_interval_ps": frame_interval,
                       "cluster_tsv": str(clusters_path) if clusters_path else None})
        for index in selected:
            try:
                frame = md.load_frame(str(trajectory), index, top=str(topology_path))
            except Exception as error:
                raise WorkflowError(f"Could not inspect {trajectory} frame {index}: {error}") from error
            time_ns = index * frame_interval / 1000 if frame_interval is not None else \
                      float(frame.time[0]) / 1000 if frame.time is not None else None
            if time_range is not None:
                if time_ns is None:
                    raise WorkflowError("Trajectory has no time data for time_range_ns selection")
                if not time_range["start"] <= time_ns < time_range["stop"]:
                    continue
            cluster = clusters.get(index, "unassigned")
            lig_h = {i: near_h(frame, i, lig) for i in lig_donors}
            lig_h = {i: h for i, h in lig_h.items() if h}
            water_o = [oxygen for _, oxygen in waters]
            ligand_polar = sorted(set(lig_acceptors + lig_donors))
            if water_o and ligand_polar:
                distances = md.compute_distances(frame, [(oxygen, atom) for oxygen in water_o
                                                         for atom in ligand_polar], periodic=True)[0] * 10
                near_ligand = distances.reshape(len(water_o), len(ligand_polar)).min(axis=1) <= loose[0]
            else:
                near_ligand = []
            for site, (site_res, acceptors, donors, bridge) in site_atoms.items():
                protein_h = {i: near_h(frame, i, site_res) for i in donors}
                protein_h = {i: h for i, h in protein_h.items() if h}
                polar = sorted(set(acceptors + donors))
                pairs = [(a, b) for a in polar for b in ligand_polar]
                if not pairs:
                    raise WorkflowError(f"Site {site} has no polar atoms")
                distances = md.compute_distances(frame, pairs, periodic=True)[0] * 10
                choice = int(distances.argmin())
                direct = events(frame, protein_h, lig_acceptors, loose, strict, "protein->ligand")
                direct += events(frame, lig_h, acceptors, loose, strict, "ligand->protein")
                frame_geometry.append([rep, index, time_ns, cluster, site, float(distances[choice]),
                                       pairs[choice][0], pairs[choice][1], len(direct), sum(x[6] for x in direct)])
                for direction, donor, hydrogen, acceptor, dist, angle, is_strict in direct:
                    direct_events.append([rep, index, time_ns, cluster, site, direction, donor, hydrogen,
                                          acceptor, dist, angle, is_strict])
                if not bridge:
                    continue
                found = []
                site_distances = md.compute_distances(frame, [(oxygen, atom) for oxygen in water_o
                                                               for atom in acceptors], periodic=True)[0] * 10
                near_site = site_distances.reshape(len(water_o), len(acceptors)).min(axis=1) <= loose[0]
                for wi, (water, oxygen) in enumerate(waters):
                    if not (near_ligand[wi] and near_site[wi]):
                        continue
                    water_h = near_h(frame, oxygen, water)
                    left = events(frame, {oxygen: water_h}, acceptors, loose, strict, "water->site")
                    right = events(frame, {oxygen: water_h}, lig_acceptors, loose, strict, "water->ligand")
                    right += events(frame, lig_h, [oxygen], loose, strict, "ligand->water")
                    pairs = [(a, b) for a in left for b in right
                             if not (b[0] == "water->ligand" and a[2] == b[2])]
                    if not pairs:
                        continue
                    a, b = max(pairs, key=lambda p: (p[0][6] + p[1][6], p[0][5] + p[1][5],
                                                     -p[0][4] - p[1][4]))
                    both = int(a[6] and b[6])
                    found.append(both)
                    bridge_events.append([rep, index, time_ns, cluster, site, water.resSeq,
                                          a[3], a[4], a[5], b[0], b[1], b[3], b[4], b[5], both])
                bridge_frames.append([rep, index, time_ns, cluster, site, len(found), sum(found)])
    write_tsv(output / "endpoint_key_residue_geometry.tsv",
              ["replica", "frame", "time_ns", "cluster", "site", "min_polar_A", "protein_atom", "ligand_atom",
               "hbonds", "strict_hbonds"], frame_geometry)
    write_tsv(output / "endpoint_direct_hbonds.tsv",
              ["replica", "frame", "time_ns", "cluster", "site", "direction", "donor", "hydrogen",
               "acceptor", "distance_A", "angle_deg", "strict"], direct_events)
    write_tsv(output / "endpoint_water_bridge_frames.tsv",
              ["replica", "frame", "time_ns", "cluster", "site", "bridges", "strict_bridges"], bridge_frames)
    write_tsv(output / "endpoint_water_bridges.tsv",
              ["replica", "frame", "time_ns", "cluster", "site", "water", "site_acceptor", "site_distance_A",
               "site_angle_deg", "ligand_link", "ligand_donor", "ligand_acceptor", "ligand_distance_A",
               "ligand_angle_deg", "strict"], bridge_events)
    grouped = defaultdict(list)
    by_site_bridge = defaultdict(list)
    for row in frame_geometry:
        grouped[(row[0], row[3], row[4])].append(row)
    for row in bridge_frames:
        by_site_bridge[(row[0], row[3], row[4])].append(row)
    summary = []
    for (rep, cluster, site), rows in sorted(grouped.items()):
        count = len(rows)
        bridges = by_site_bridge[(rep, cluster, site)]
        summary.append([rep, cluster, site, count, statistics.median(r[5] for r in rows),
                        sum(r[8] > 0 for r in rows) / count, sum(r[9] > 0 for r in rows) / count,
                        sum(r[5] > 0 for r in bridges) / len(bridges) if bridges else "",
                        sum(r[6] > 0 for r in bridges) / len(bridges) if bridges else ""])
    write_tsv(output / "endpoint_network_by_replica_cluster.tsv",
              ["replica", "cluster", "site", "frames", "median_min_A", "direct", "strict",
               "water", "water_strict"], summary)
    denominators = defaultdict(set)
    paths = defaultdict(set)
    for row in frame_geometry:
        denominators[(row[3], row[4])].add((row[0], row[1]))
    for row in direct_events:
        paths[(row[3], row[4], row[5], row[6], row[8])].add((row[0], row[1]))
    write_tsv(output / "endpoint_direct_paths_by_cluster.tsv",
              ["cluster", "site", "direction", "donor", "acceptor", "frames", "occupancy"],
              [[*key, len(frames), len(frames) / len(denominators[key[:2]])]
               for key, frames in sorted(paths.items())])
    bridge_denominators = defaultdict(set)
    bridge_paths = defaultdict(set)
    for row in bridge_frames:
        bridge_denominators[(row[3], row[4])].add((row[0], row[1]))
    for row in bridge_events:
        ligand_atom = row[11] if row[9] == "water->ligand" else row[10]
        bridge_paths[(row[3], row[4], row[6], row[9], ligand_atom)].add((row[0], row[1]))
    write_tsv(output / "endpoint_water_paths_by_cluster.tsv",
              ["cluster", "site", "site_acceptor", "ligand_link", "ligand_atom", "frames", "occupancy"],
              [[*key, len(frames), len(frames) / len(bridge_denominators[key[:2]])]
               for key, frames in sorted(bridge_paths.items())])
    write(output / "analysis_provenance.json", {"config_sha256": sha256(config_path), "inputs": inputs,
                                               "loose": config["loose"], "strict": config["strict"],
                                               "ligand": config["ligand"], "sites": config["sites"],
                                               "water_resnames": aliases, "producer_runtime": runtime})
    return {"output": str(output), "frames": len({(r[0], r[1]) for r in frame_geometry}),
            "direct_events": len(direct_events), "water_bridges": len(bridge_events)}
