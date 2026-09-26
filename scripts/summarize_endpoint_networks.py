import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])

def read(name):
    with (root / name).open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))

geom = read("endpoint_key_residue_geometry.tsv")
direct = read("endpoint_direct_hbonds.tsv")
bridge_frames = read("endpoint_water_bridge_frames.tsv")
bridges = read("endpoint_water_bridges.tsv")

focus = {"2:ASN121", "2:ASP120", "2:ASP256", "2:GLU280"}

g = defaultdict(list)
for row in geom:
    if row["residue"] in focus:
        g[(row["replica"], int(row["cluster"]), row["residue"])].append(row)

b = defaultdict(list)
for row in bridge_frames:
    b[(row["replica"], int(row["cluster"]), row["site"])].append(row)

print("===== Replica-stratified polar network =====")
print("replica\tcluster\tresidue\tframes\tmedian_A\tdirect\tstrict\twater\twater_strict")
for key, rows in sorted(g.items()):
    replica, cluster, residue = key
    n = len(rows)
    distances = [float(row["min_polar_A"]) for row in rows]
    direct_occ = sum(int(row["hbonds"]) > 0 for row in rows) / n
    strict_occ = sum(int(row["strict_hbonds"]) > 0 for row in rows) / n
    water = water_strict = "-"
    brow = b.get((replica, cluster, residue), [])
    if brow:
        water = f'{sum(int(row["bridges"]) > 0 for row in brow) / len(brow):.3f}'
        water_strict = f'{sum(int(row["strict_bridges"]) > 0 for row in brow) / len(brow):.3f}'
    print(
        replica, cluster, residue, n,
        f"{statistics.median(distances):.3f}",
        f"{direct_occ:.3f}", f"{strict_occ:.3f}",
        water, water_strict, sep="\t",
    )

cluster_frames = defaultdict(set)
for row in geom:
    cluster_frames[int(row["cluster"])].add((row["replica"], row["frame"]))

direct_paths = defaultdict(set)
for row in direct:
    key = (
        int(row["cluster"]), row["residue"], row["direction"],
        row["donor"], row["acceptor"],
    )
    direct_paths[key].add((row["replica"], row["frame"]))

print("\n===== Direct atom-pair occupancy by cluster =====")
print("cluster\tresidue\tdirection\tdonor\tacceptor\tframes\toccupancy")
ranked = []
for key, observed in direct_paths.items():
    cluster = key[0]
    occupancy = len(observed) / len(cluster_frames[cluster])
    ranked.append((occupancy, key, len(observed)))
for occupancy, key, n in sorted(ranked, reverse=True)[:40]:
    print(*key, n, f"{occupancy:.3f}", sep="\t")

bridge_denominator = defaultdict(set)
for row in bridge_frames:
    bridge_denominator[(int(row["cluster"]), row["site"])].add(
        (row["replica"], row["frame"])
    )

bridge_paths = defaultdict(set)
for row in bridges:
    if row["ligand_link"] == "water->ligand":
        ligand_atom = row["ligand_acceptor"]
    else:
        ligand_atom = row["ligand_donor"]
    key = (
        int(row["cluster"]), row["site"], row["asp_acceptor"],
        row["ligand_link"], ligand_atom,
    )
    bridge_paths[key].add((row["replica"], row["frame"]))

print("\n===== Single-water bridge-path occupancy by cluster =====")
print("cluster\tsite\tAsp_atom\tligand_link\tligand_atom\tframes\toccupancy")
ranked = []
for key, observed in bridge_paths.items():
    cluster, site = key[:2]
    denominator = len(bridge_denominator[(cluster, site)])
    occupancy = len(observed) / denominator
    ranked.append((occupancy, key, len(observed)))
for occupancy, key, n in sorted(ranked, reverse=True)[:40]:
    print(*key, n, f"{occupancy:.3f}", sep="\t")
