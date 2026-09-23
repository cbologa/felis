#!/usr/bin/env python3
"""
Add deterministic ACE and NME caps to every protein chain.

Input must be heavy-atom-only and contain TER records between chains.
Output is renumbered within each chain:
    ACE = 1, protein = 2..N+1, NME = N+2
A mapping TSV preserves the pre-cap residue identifiers.

Only heavy atoms for ACE (CH3,C,O) and NME (N,C) are written.  LEaP adds the
cap hydrogens.  Geometry is constructed from terminal backbone atoms and is
deterministic; no random orientation is used.
"""
from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np

def atom_name(x): return x[12:16].strip()
def res_name(x): return x[17:20].strip()
def chain_id(x): return x[21:22]
def res_id(x): return x[22:26].strip()
def xyz(x): return np.array([float(x[30:38]),float(x[38:46]),float(x[46:54])],dtype=float)

def unit(v):
    n=float(np.linalg.norm(v))
    if n < 1e-8:
        raise ValueError("degenerate terminal geometry")
    return v/n

def normal_from(a,b):
    c=np.cross(a,b)
    if np.linalg.norm(c) < 1e-8:
        # deterministic fallback not parallel to a
        ref=np.array([1.,0.,0.])
        if abs(np.dot(unit(a),ref)) > 0.9:
            ref=np.array([0.,1.,0.])
        c=np.cross(a,ref)
    return unit(c)

def pdb_new(serial,name,resname,chain,resid,pos,element):
    return (f"ATOM  {serial:5d} {name:>4s} {resname:>3s} {chain:1s}{resid:4d}    "
            f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}"
            f"  1.00  0.00          {element:>2s}\n")

def rewrite_original(line,serial,chain,resid):
    s=line.rstrip("\n")
    if len(s)<80: s=s.ljust(80)
    chars=list(s)
    chars[6:11]=list(f"{serial:5d}")
    chars[21]=chain
    chars[22:26]=list(f"{resid:4d}")
    chars[26]=" "
    return "".join(chars).rstrip()+"\n"

def parse_segments(lines):
    segs=[]; current=[]; last_chain=None
    for x in lines:
        if x.startswith("ATOM  "):
            ch=chain_id(x)
            if current and last_chain is not None and ch != last_chain:
                segs.append(current); current=[]
            current.append(x); last_chain=ch
        elif x.startswith("TER"):
            if current:
                segs.append(current); current=[]
                last_chain=None
    if current: segs.append(current)
    return segs

def residue_groups(seg):
    out=[]; key=None; cur=[]
    for x in seg:
        k=(chain_id(x),x[22:26],x[26:27],res_name(x))
        if key is None or k==key:
            cur.append(x); key=k
        else:
            out.append((key,cur)); cur=[x]; key=k
    if cur: out.append((key,cur))
    return out

def byname(lines):
    return {atom_name(x):xyz(x) for x in lines}

def min_dist(point, protein_xyz):
    return float(np.linalg.norm(protein_xyz-point,axis=1).min())

def ace_geometry(first_atoms,protein_xyz):
    q=byname(first_atoms)
    for n in ("N","CA","C"):
        if n not in q: raise SystemExit(f"ERROR: N-terminal residue lacks {n}")
    N,CA,C=q["N"],q["CA"],q["C"]
    Ccap=N + 1.335*unit(N-CA)
    u=unit(N-Ccap)                              # Ccap -> protein N
    plane_n=normal_from(CA-N,C-N)
    v=unit(np.cross(plane_n,u))
    choices=[]
    for sign in (+1.,-1.):
        vv=sign*v
        dO=-0.5*u + (math.sqrt(3)/2.0)*vv
        dM=-0.5*u - (math.sqrt(3)/2.0)*vv
        O=Ccap + 1.229*dO
        CH3=Ccap + 1.522*dM
        score=min(min_dist(O,protein_xyz),min_dist(CH3,protein_xyz))
        choices.append((score,O,CH3))
    score,O,CH3=max(choices,key=lambda z:z[0])
    return CH3,Ccap,O,score

def nme_geometry(last_atoms,protein_xyz):
    q=byname(last_atoms)
    for n in ("C","O","CA"):
        if n not in q: raise SystemExit(f"ERROR: C-terminal residue lacks {n}")
    C,O,CA=q["C"],q["O"],q["CA"]
    Ncap=C + 1.335*unit(C-(O+CA)/2.0)
    u=unit(C-Ncap)                              # Ncap -> protein carbonyl C
    plane_n=normal_from(O-C,CA-C)
    v=unit(np.cross(plane_n,u))
    choices=[]
    for sign in (+1.,-1.):
        dM=-0.5*u + sign*(math.sqrt(3)/2.0)*v
        CH3=Ncap + 1.450*dM
        score=min_dist(CH3,protein_xyz)
        choices.append((score,CH3))
    score,CH3=max(choices,key=lambda z:z[0])
    return Ncap,CH3,score
