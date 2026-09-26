import csv, math, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path

import MDAnalysis as mda
import numpy as np
from MDAnalysis.lib.distances import calc_angles, distance_array

root, out = map(Path, sys.argv[1:3])
reps = ("r1", "r2", "r3")
keys = (("SER",18),("LYS",43),("TYR",81),("ASP",120),("ASN",121),
        ("SER",143),("TYR",193),("ASP",256),("GLU",280),("SER",281))
asp_sites = (("ASP",120),("ASP",256))
acc_names = {"ASP":{"OD1","OD2"}, "GLU":{"OE1","OE2"},
             "ASN":{"OD1"}, "SER":{"OG"}, "TYR":{"OH"}}
don_names = {"ASN":{"ND2"}, "SER":{"OG"}, "TYR":{"OH"}, "LYS":{"NZ"}}
water_names = {"SOL","HOH","WAT","TIP3","TIP3P","T3P","SPC","SPCE"}
loose = (3.5, 135.0)
strict = (3.2, 150.0)

with (out / "endpoint_pose_metrics.tsv").open() as fh:
    rows = list(csv.DictReader(fh, delimiter="\t"))
cluster = {(r["replica"], int(float(r["frame"]))): int(float(r["cluster"]))
           for r in rows}

def elem(a):
    e = str(getattr(a, "element", "")).strip().upper()
    return e or a.name.strip().lstrip("0123456789")[:1].upper()

def chain(a):
    return (str(getattr(a,"chainID","")).strip()
            or str(getattr(a,"segid","")).strip() or "?")

def rlabel(r): return f"{chain(r.atoms[0])}:{r.resname}{r.resid}"
def alabel(a): return f"{rlabel(a.residue)}:{a.name}"

def box(ts):
    b = ts.dimensions
    return b if b is not None and np.all(np.asarray(b[:3]) > 0) else None

def near_h(u, heavy, atom_indices, b):
    hs = [i for i in atom_indices if elem(u.atoms[i]) == "H"]
    if not hs: return []
    d = distance_array(u.atoms[[heavy]].positions, u.atoms[hs].positions, box=b)[0]
    return [i for i, x in zip(hs, d) if x <= 1.30]

def geom(u, d, h, a, b):
    dp, hp, ap = (u.atoms[i].position[None,:] for i in (d,h,a))
    da = float(distance_array(dp, ap, box=b)[0,0])
    ang = math.degrees(float(calc_angles(dp, hp, ap, box=b)[0]))
    return da, ang

def events(u, donors, acceptors, b, direction):
    ans = []
    for d, hs in donors.items():
        for h in hs:
            for a in acceptors:
                dist, ang = geom(u,d,h,a,b)
                if dist <= loose[0] and ang >= loose[1]:
                    ans.append((direction,d,h,a,dist,ang,
                                dist <= strict[0] and ang >= strict[1]))
    return ans

def choose_residue(u, rn, ri, lig, b):
    candidates = [r for r in u.residues if r.resname == rn and int(r.resid) == ri]
    if not candidates: raise RuntimeError(f"Missing {rn}{ri}")
    return min(candidates, key=lambda r: distance_array(
        r.atoms[[elem(a) != "H" for a in r.atoms]].positions,
        lig.positions, box=b).min())

frame_geometry, direct_events, bridge_frames, bridge_events = [], [], [], []

for rep in reps:
    work = root / "calculations/sucrose_in_R" / rep / "work/sucrose"
    u = mda.Universe(str(work/"prepare/sysB_em.pdb"),
                     str(out/f"{rep}_B79_endpoint.dcd"))
    u.trajectory[0]; b0 = box(u.trajectory.ts)
    lig_res = [r for r in u.residues if r.resname == "M00"]
    if len(lig_res) != 1: raise RuntimeError(f"{rep}: M00 ligands={len(lig_res)}")
    lig = lig_res[0]
    lig_heavy = lig.atoms[[elem(a) != "H" for a in lig.atoms]]
    lig_o = [a.index for a in lig.atoms if elem(a) == "O"]
    lig_don = {o: near_h(u,o,list(lig.atoms.indices),b0) for o in lig_o}
    lig_don = {d:hs for d,hs in lig_don.items() if hs}

    selected, prot_acc, prot_don, prot_polar = {}, {}, {}, {}
    for key in keys:
        rn, ri = key; r = choose_residue(u,rn,ri,lig_heavy,b0); selected[key] = r
        prot_acc[key] = [a.index for a in r.atoms if a.name in acc_names.get(rn,set())]
        prot_don[key] = {}
        for a in r.atoms:
            if a.name in don_names.get(rn,set()):
                hs = near_h(u,a.index,list(r.atoms.indices),b0)
                if hs: prot_don[key][a.index] = hs
        prot_polar[key] = sorted(set(prot_acc[key]) | set(prot_don[key]))

    water_o, water_h = [], {}
    water_resnames = Counter()
    for r in u.residues:
        oo = [a.index for a in r.atoms if elem(a) == "O"]
        hh = [a.index for a in r.atoms if elem(a) == "H"]
        is_water = r.resname.upper() in water_names or (len(r.atoms)<=4 and len(oo)==1 and len(hh)>=2)
        if is_water and len(oo)==1 and len(hh)>=2:
            water_o.append(oo[0]); water_h[oo[0]] = hh; water_resnames[r.resname] += 1
    if not water_o: raise RuntimeError(f"{rep}: no explicit waters found")
    print(f"{rep}: ligand O={len(lig_o)}, OH donors={len(lig_don)}, "
          f"waters={len(water_o)} {dict(water_resnames)}")

    for ts in u.trajectory[8:41]:
        f = int(ts.frame); c = cluster[(rep,f)]; b = box(ts); t = f*0.25
        for key in keys:
            r = selected[key]; polar = prot_polar[key]
            dm = distance_array(u.atoms[polar].positions, u.atoms[lig_o].positions, box=b)
            ip, il = np.unravel_index(int(dm.argmin()), dm.shape)
            ev = events(u,prot_don[key],lig_o,b,"protein->ligand")
            ev += events(u,lig_don,prot_acc[key],b,"ligand->protein")
            frame_geometry.append([rep,f,t,c,rlabel(r),float(dm[ip,il]),
                                   alabel(u.atoms[polar[ip]]),alabel(u.atoms[lig_o[il]]),
                                   len(ev),sum(x[6] for x in ev)])
            for direction,d,h,a,dist,ang,is_strict in ev:
                direct_events.append([rep,f,t,c,rlabel(r),direction,alabel(u.atoms[d]),
                                      alabel(u.atoms[h]),alabel(u.atoms[a]),dist,ang,int(is_strict)])

        wgroup = u.atoms[water_o]
        dwl = distance_array(wgroup.positions,u.atoms[lig_o].positions,box=b).min(axis=1)
        for key in asp_sites:
            r = selected[key]; aa = prot_acc[key]
            dwa = distance_array(wgroup.positions,u.atoms[aa].positions,box=b).min(axis=1)
            found = []
            for wi in np.where((dwl<=loose[0]) & (dwa<=loose[0]))[0]:
                wo = water_o[int(wi)]; wdon = {wo:water_h[wo]}
                left = events(u,wdon,aa,b,"water->aspartate")
                right = events(u,wdon,lig_o,b,"water->ligand")
                right += events(u,lig_don,[wo],b,"ligand->water")
                pairs = [(x,y) for x in left for y in right
                         if not (y[0]=="water->ligand" and x[2]==y[2])]
                if not pairs: continue
                x,y = max(pairs,key=lambda p:(p[0][6]+p[1][6],p[0][5]+p[1][5],-p[0][4]-p[1][4]))
                both_strict = int(x[6] and y[6]); found.append(both_strict)
                bridge_events.append([rep,f,t,c,rlabel(r),rlabel(u.atoms[wo].residue),
                                      alabel(u.atoms[x[3]]),x[4],x[5],y[0],
                                      alabel(u.atoms[y[1]]),alabel(u.atoms[y[3]]),y[4],y[5],both_strict])
            bridge_frames.append([rep,f,t,c,rlabel(r),len(found),sum(found)])

def save(name, header, data):
    with (out/name).open("w",newline="") as fh:
        w=csv.writer(fh,delimiter="\t"); w.writerow(header); w.writerows(data)

save("endpoint_key_residue_geometry.tsv",
     ["replica","frame","time_ns","cluster","residue","min_polar_A",
      "protein_atom","ligand_atom","hbonds","strict_hbonds"],frame_geometry)
save("endpoint_direct_hbonds.tsv",
     ["replica","frame","time_ns","cluster","residue","direction","donor",
      "hydrogen","acceptor","distance_A","angle_deg","strict"],direct_events)
save("endpoint_water_bridge_frames.tsv",
     ["replica","frame","time_ns","cluster","site","bridges","strict_bridges"],bridge_frames)
save("endpoint_water_bridges.tsv",
     ["replica","frame","time_ns","cluster","site","water","asp_acceptor",
      "asp_distance_A","asp_angle_deg","ligand_link","ligand_donor","ligand_acceptor",
      "ligand_distance_A","ligand_angle_deg","strict"],bridge_events)

print("\n===== Pooled direct H-bond geometry by cluster =====")
print("cluster\tresidue\tframes\tmedian_min_A\tcontact\tHbond>=135\tstrict")
g=defaultdict(list)
for r in frame_geometry: g[(r[3],r[4])].append(r)
for (c,site),rr in sorted(g.items()):
    d=[x[5] for x in rr]
    print(c,site,len(rr),f"{statistics.median(d):.3f}",
          f"{sum(x<=loose[0] for x in d)/len(d):.3f}",
          f"{sum(x[8]>0 for x in rr)/len(rr):.3f}",
          f"{sum(x[9]>0 for x in rr)/len(rr):.3f}",sep="\t")

print("\n===== Pooled Asp water bridges by cluster =====")
print("cluster\tsite\tframes\tbridge\tstrict\tmean_count\tmax")
g=defaultdict(list)
for r in bridge_frames: g[(r[3],r[4])].append(r)
for (c,site),rr in sorted(g.items()):
    n=[x[5] for x in rr]; sn=[x[6] for x in rr]
    print(c,site,len(rr),f"{sum(x>0 for x in n)/len(n):.3f}",
          f"{sum(x>0 for x in sn)/len(sn):.3f}",f"{sum(n)/len(n):.3f}",max(n),sep="\t")

print("\n===== Most frequent direct atom pairs =====")
count=Counter((r[3],r[4],r[5],r[6],r[8]) for r in direct_events)
print("cluster\tresidue\tdirection\tdonor\tacceptor\tevents")
for k,n in count.most_common(30): print(*k,n,sep="\t")

print("\nSaved:")
for name in ("endpoint_key_residue_geometry.tsv","endpoint_direct_hbonds.tsv",
             "endpoint_water_bridge_frames.tsv","endpoint_water_bridges.tsv"):
    print(out/name)
