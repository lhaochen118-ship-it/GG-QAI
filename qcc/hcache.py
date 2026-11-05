"""History-cache search: Toffoli-like gates whose product is a GF(2) combination of earlier values.

For a gate at time T acting with product v (CCX/CCZ: a&b(&c), CSWAP: c&(p^q), times its condition
mask), look for an idle wire z (untouched and zero on [t0, T]) and test whether v lies in the span of

    {all wire values at t0} U {wire deltas of every nonlinear wire update in (t0, T)} U {1}.

Every such value can be XORed into z with CNOTs only (a wire value: CX w z; the delta of a CCX,
CSWAP or conditional update on wire w: CX w z before and after it), so z ends holding v; the gate
becomes CX z target (CSWAP: into both targets; CCZ: Z z), and z is erased with HMR plus the
Clifford repair CZ(a,b) (CSWAP: CZ(c,p) CZ(c,q)) conditioned on the outcome (CCZ needs a
conditional CCZ: half a Toffoli on average). Saves 1 Toffoli (CCZ: 1/2) per hit, times its
execution probability. Usage: python hcache.py circuit.kmx [--shots 4096] [--out hits.json]
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from ksim import Kmx, TOFFOLI_LIKE, pack

NONLIN = {'CCX', 'CSWAP'}


def gf2_rank_add(basis: dict, v: np.ndarray) -> np.ndarray | None:
    """Reduce packed bit vector v by basis {pivot_bit: vec}; return remainder (None if zero)."""
    v = v.copy()
    while True:
        nz = np.flatnonzero(v)
        if not len(nz):
            return None
        w = nz[0]
        bit = int(v[w]) & -int(v[w])
        piv = w * 64 + bit.bit_length() - 1
        if piv not in basis:
            return v
        v ^= basis[piv]


def insert(basis: dict, v: np.ndarray) -> bool:
    r = gf2_rank_add(basis, v)
    if r is None:
        return False
    nz = np.flatnonzero(r)
    w = nz[0]
    bit = int(r[w]) & -int(r[w])
    basis[w * 64 + bit.bit_length() - 1] = r
    return True


def record(circ: Kmx, xs: np.ndarray, x_qubits, seed: int, snap_times: set[int]):
    """One pass: product vectors of Toffoli-like ops, deltas of nonlinear updates, snapshots."""
    shots = len(xs)
    W = (shots + 63) // 64
    rng = np.random.default_rng(seed)
    q = np.zeros((circ.num_qubits, W), dtype=np.uint64)
    b = np.zeros((max(circ.num_bits, 1), W), dtype=np.uint64)
    valid = pack(np.ones(shots, dtype=bool))
    for k, qi in enumerate(x_qubits):
        q[qi] = pack(((xs >> k) & 1).astype(bool))
    stack = [valid]
    prods, deltas, snaps = {}, {}, {}
    for idx, op in enumerate(circ.ops):
        if idx in snap_times:
            snaps[idx] = q.copy()
        n = op.name
        if n == 'PUSH_CONDITION':
            stack.append(stack[-1] & b[op.cond]); continue
        if n == 'POP_CONDITION':
            stack.pop(); continue
        m = stack[-1] if op.cond is None else stack[-1] & b[op.cond]
        cond = m is not valid and not np.array_equal(m, valid)
        qs = op.qs
        if n == 'CX':
            if cond:
                deltas[idx] = (qs[1], q[qs[0]] & m)
            q[qs[1]] ^= q[qs[0]] & m
        elif n == 'X':
            if cond:
                deltas[idx] = (qs[0], m.copy())
            q[qs[0]] ^= m
        elif n == 'CCX':
            p = q[qs[0]] & q[qs[1]] & m
            prods[idx] = p; deltas[idx] = (qs[2], p)
            q[qs[2]] ^= p
        elif n == 'CCZ':
            prods[idx] = q[qs[0]] & q[qs[1]] & q[qs[2]] & m
        elif n == 'CSWAP':
            t = q[qs[0]] & (q[qs[1]] ^ q[qs[2]]) & m
            prods[idx] = t; deltas[idx] = (qs[1], t)
            q[qs[1]] ^= t; q[qs[2]] ^= t
        elif n == 'SWAP':
            t = (q[qs[0]] ^ q[qs[1]]) & m
            if cond:
                deltas[idx] = (qs[0], t)
            q[qs[0]] ^= t; q[qs[1]] ^= t
        elif n == 'HMR':
            r = rng.integers(0, 2**64, size=W, dtype=np.uint64) & valid
            b[op.bs[0]] = (b[op.bs[0]] & ~m) | (r & m)
            q[qs[0]] &= ~m
        elif n == 'R':
            q[qs[0]] &= ~m
        elif n == 'BIT_STORE0':
            b[op.bs[0]] &= ~m
        elif n == 'BIT_STORE1':
            b[op.bs[0]] |= m
        elif n == 'BIT_INVERT':
            b[op.bs[0]] ^= m
    return prods, deltas, snaps


def idle_intervals(circ: Kmx, min_len: int):
    """Per qubit, maximal [s, e) op ranges in which the qubit is not an operand."""
    last = {}
    out = []
    for idx, op in enumerate(circ.ops):
        for qi in op.qs:
            s = last.get(qi, -1) + 1
            if idx - s >= min_len:
                out.append((qi, s, idx))
            last[qi] = idx
    for qi in range(circ.num_qubits):
        s = last.get(qi, -1) + 1
        if len(circ.ops) - s >= min_len:
            out.append((qi, s, len(circ.ops)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('circuit')
    ap.add_argument('--shots', type=int, default=4096)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--min-len', type=int, default=20)
    ap.add_argument('--max-window', type=int, default=6000)
    ap.add_argument('--part', default='0/1', help='k/n: only test targets with index %% n == k')
    ap.add_argument('--out', default='hcache_hits.json')
    a = ap.parse_args()
    circ = Kmx.load(a.circuit)
    rng = np.random.default_rng(a.seed)
    max_x = 3 * (2**10 - 2)**2
    xs = np.r_[np.arange(1, 1025), rng.integers(1, max_x + 1, a.shots - 1024)].astype(np.int64)
    assert len(xs) % 64 == 0
    ivs = idle_intervals(circ, a.min_len)
    snap_times = {s for _, s, _ in ivs}
    prods, deltas, snaps = record(circ, xs, circ.registers['x'], a.seed, snap_times)
    # zero at interval start on the sample
    ivs = [(qi, s, e) for qi, s, e in ivs if not snaps[s][qi].any()]
    k, n = map(int, a.part.split('/'))
    targets = [i for i in sorted(prods) if i % n == k and circ.ops[i].name in TOFFOLI_LIKE]
    delta_idx = np.array(sorted(deltas))
    hits = []
    for T in targets:
        v = prods[T]
        if not v.any():
            hits.append({'T': T, 'kind': 'zero'}); continue
        cands = [(s, qi, e) for qi, s, e in ivs if s < T and e > T + 1 and T - s <= a.max_window]
        if not cands:
            continue
        s, qi, e = min(cands)
        basis = {}
        insert(basis, np.full_like(v, ~np.uint64(0)))
        for w in range(circ.num_qubits):
            if snaps[s][w].any():
                insert(basis, snaps[s][w])
        lo, hi = np.searchsorted(delta_idx, [s, T])
        for d in delta_idx[lo:hi]:
            insert(basis, deltas[int(d)][1])
            if gf2_rank_add(basis, v) is None:
                break
        if gf2_rank_add(basis, v) is None:
            hits.append({'T': T, 'op': circ.ops[T].name, 'cache': qi, 't0': s, 'idle_end': e,
                         'rank': len(basis)})
            print(json.dumps(hits[-1]), flush=True)
    json.dump(hits, open(a.out, 'w'))
    print('targets', len(targets), 'hits', sum(1 for h in hits if 'cache' in h), 'zero', sum(1 for h in hits if h.get('kind') == 'zero'))


if __name__ == '__main__':
    main()
