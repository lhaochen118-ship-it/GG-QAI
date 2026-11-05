"""Bit-sliced simulator for kickmix text circuits (.kmx), built for analysis.

Every shot is one bit lane: a qubit or classical bit is a uint64 array of `W = ceil(shots / 64)`
words, so one gate updates all shots with one numpy call (the full invsqrt domain, 3.1M inputs,
runs in a few seconds). Besides the final state it records what the official verifier hides:

* `rot_bits[r]`  - bool per shot, whether rotation `r` (a non-Clifford+T `Z_POW`) fired;
* `phase8`       - the remaining phase per shot in units of 1/8 turn (Z, CZ, CCZ, NEG, HMR kickback,
                   Clifford and T `Z_POW` angles), exact;
* `toffolis`     - executed Toffoli-like gates summed over shots (conditional gates count only when
                   their condition holds, like `km.Simulator`);
* `lifetimes`    - per qubit, the [start, end) instruction intervals during which it is nonzero on
                   some shot (only with `track_life=True`).

Usage: `Kmx.parse(text)` or `Kmx.load(path)` (.kmb is converted through kickmix), then
`simulate(circ, xs, x_qubits, seed=...)`.
"""

from __future__ import annotations

import dataclasses
import re
from fractions import Fraction

import numpy as np

ONES = np.uint64(0xFFFFFFFFFFFFFFFF)
TOFFOLI_LIKE = {'CCX', 'CCZ', 'CSWAP'}
_TOKEN = re.compile(r'([qb])(\d+)')


@dataclasses.dataclass
class Op:
    name: str
    qs: tuple[int, ...]          # qubit operands
    bs: tuple[int, ...]          # classical operands (HMR target, BIT_* target)
    cond: int | None             # `if bN` suffix
    angle: Fraction | None = None  # Z_POW exponent in half turns


@dataclasses.dataclass
class Kmx:
    ops: list[Op]
    num_qubits: int
    num_bits: int
    registers: dict[str, list[int]]

    @classmethod
    def parse(cls, text: str) -> 'Kmx':
        ops, regs, pending = [], {}, {}
        nq = nb = 0
        for line in text.splitlines():
            t = line.split()
            if not t:
                continue
            if t[0] == 'APPEND_TO_REGISTER':
                pending.setdefault(t[2], []).append(int(t[1][1:]))
                continue
            if t[0] in ('INFO_TAG', 'MARK'):
                continue
            if t[0] == 'REGISTER':
                regs[t[2].strip('"')] = pending.pop(t[1])
                continue
            cond = None
            if 'if' in t:
                i = t.index('if')
                cond = int(t[i + 1][1:])
                t = t[:i]
            angle = None
            if t[0] == 'Z_POW':
                angle = Fraction(t[2])
                t = t[:2]
            qs = tuple(int(a[1:]) for a in t[1:] if a[0] == 'q')
            bs = tuple(int(a[1:]) for a in t[1:] if a[0] == 'b')
            nq = max([nq, *(q + 1 for q in qs)])
            nb = max([nb, *(b + 1 for b in bs), (cond + 1) if cond is not None else 0])
            ops.append(Op(t[0], qs, bs, cond, angle))
        return cls(ops, nq, nb, regs)

    @classmethod
    def load(cls, path: str) -> 'Kmx':
        if path.endswith('.kmx'):
            return cls.parse(open(path).read())
        import io
        import kickmix as km
        buf = io.StringIO()
        km.Circuit.from_file(path).to_file(buf, format='kmx')
        return cls.parse(buf.getvalue())

    def dump(self) -> str:
        """Back to .kmx text (registers first, as kickmix writes them)."""
        out = []
        for i, (name, qs) in enumerate(self.registers.items()):
            out += [f'APPEND_TO_REGISTER q{q} r{i}' for q in qs] + [f'REGISTER r{i} "{name}"']
        for op in self.ops:
            args = [f'q{q}' for q in op.qs] + [f'b{b}' for b in op.bs]
            if op.angle is not None:
                args.append(_fmt_angle(op.angle))
            if op.cond is not None:
                args += ['if', f'b{op.cond}']
            out.append(' '.join([op.name, *args]))
        return '\n'.join(out) + '\n'


def _fmt_angle(a: Fraction) -> str:
    """Exact decimal for dyadic angles (as kickmix prints them), else a long decimal."""
    n, d = a.numerator, a.denominator
    if d & (d - 1) == 0:
        k = d.bit_length() - 1
        s = str(abs(n) * 5**k)
        s = s.rjust(k + 1, '0')
        s = (s[:-k] + '.' + s[-k:]).rstrip('0').rstrip('.') if k else s
        return ('-' if n < 0 else '') + (s if '.' in s or k == 0 else s)
    return f'{float(a):.60f}'


def angle_class(a: Fraction) -> str:
    if a % Fraction(1, 2) == 0:
        return 'clifford'
    if a % Fraction(1, 4) == 0:
        return 't'
    return 'rotation'


def pack(bits: np.ndarray) -> np.ndarray:
    """bool[shots] -> uint64[W] (lane i = shot i)."""
    padded = np.pad(np.asarray(bits, dtype=bool), (0, -len(bits) % 64))
    return np.packbits(padded, bitorder='little').view(np.uint64).copy()


def unpack(words: np.ndarray, shots: int) -> np.ndarray:
    return np.unpackbits(words.view(np.uint8), bitorder='little')[:shots].astype(bool)


def popcount(words: np.ndarray) -> int:
    return int(np.bitwise_count(words).sum())


@dataclasses.dataclass
class SimResult:
    q: np.ndarray            # uint64[num_qubits, W] final qubit state
    phase8: np.ndarray       # uint8[shots] phase in 1/8 turns (mod 8), rotations excluded
    rot_ops: list[int]       # instruction index of each rotation
    rot_angles: list[Fraction]
    rot_bits: np.ndarray     # bool[n_rot, shots]
    toffolis: int            # executed Toffoli-like gates summed over shots
    t_gates: int             # executed T-class Z_POW summed over shots
    rotations: int           # executed rotations summed over shots
    lifetimes: dict[int, list[tuple[int, int]]] | None
    reset_dirty: int         # shots where an R hit a |1> qubit


def simulate(circ: Kmx, xs: np.ndarray, x_qubits: list[int], *, seed: int = 0,
             track_life: bool = False, hmr_override: dict[int, np.ndarray] | None = None,
             probe=None, snap_at=None) -> SimResult:
    """Runs `circ` on basis inputs `xs` (one shot each) with uniformly random HMR outcomes.

    `hmr_override` maps an instruction index of an HMR to a bool[shots] outcome vector.
    `probe(idx, op, q, mask)` is called before every Toffoli-like gate.
    `snap_at`: dict whose keys are instruction indices; just before each, (qubits, half-turn
    parity, eighth-turn counts, number of rotations so far) is copied in.
    """
    shots = len(xs)
    W = (shots + 63) // 64
    rng = np.random.default_rng(seed)
    q = np.zeros((circ.num_qubits, W), dtype=np.uint64)
    b = np.zeros((max(circ.num_bits, 1), W), dtype=np.uint64)
    valid = pack(np.ones(shots, dtype=bool))
    for k, qi in enumerate(x_qubits):
        q[qi] = pack(((xs >> k) & 1).astype(bool))
    half = np.zeros(W, dtype=np.uint64)                 # half-turn parity
    eighth = np.zeros(shots, dtype=np.int64)            # T/S contributions, in 1/8 turns
    rot_ops, rot_angles, rot_bits = [], [], []
    toffolis = t_gates = rotations = reset_dirty = 0
    stack = [valid]
    life_start = {} if track_life else None
    lifetimes = {i: [] for i in range(circ.num_qubits)} if track_life else None
    if track_life:
        for i in range(circ.num_qubits):
            if q[i].any():
                life_start[i] = 0

    def rand_words():
        return rng.integers(0, 2**64, size=W, dtype=np.uint64, endpoint=False) & valid

    for idx, op in enumerate(circ.ops):
        if snap_at is not None and idx in snap_at:
            snap_at[idx] = (q.copy(), half.copy(), eighth.copy(), len(rot_ops))
        name = op.name
        if name in ('INFO_TAG', 'MARK'):
            continue
        if name == 'PUSH_CONDITION':
            stack.append(stack[-1] & b[op.cond])
            continue
        if name == 'POP_CONDITION':
            stack.pop()
            continue
        m = stack[-1] if op.cond is None else stack[-1] & b[op.cond]
        qs = op.qs
        if probe is not None and name in TOFFOLI_LIKE:
            probe(idx, op, q, m)
        if name == 'CX':
            q[qs[1]] ^= q[qs[0]] & m
        elif name == 'CZ':
            half ^= q[qs[0]] & q[qs[1]] & m
        elif name == 'Z':
            half ^= q[qs[0]] & m
        elif name == 'X':
            q[qs[0]] ^= m
        elif name == 'CCX':
            q[qs[2]] ^= q[qs[0]] & q[qs[1]] & m
            toffolis += popcount(m)
        elif name == 'CCZ':
            half ^= q[qs[0]] & q[qs[1]] & q[qs[2]] & m
            toffolis += popcount(m)
        elif name == 'CSWAP':
            t = q[qs[0]] & (q[qs[1]] ^ q[qs[2]]) & m
            q[qs[1]] ^= t
            q[qs[2]] ^= t
            toffolis += popcount(m)
        elif name == 'SWAP':
            t = (q[qs[0]] ^ q[qs[1]]) & m
            q[qs[0]] ^= t
            q[qs[1]] ^= t
        elif name == 'HMR':
            r = pack(hmr_override[idx][:shots]) if hmr_override and idx in hmr_override else rand_words()
            bt = op.bs[0]
            b[bt] = (b[bt] & ~m) | (r & m)
            half ^= q[qs[0]] & r & m
            q[qs[0]] &= ~m
        elif name == 'R':
            dirty = q[qs[0]] & m
            reset_dirty += popcount(dirty)
            half ^= dirty & rand_words()
            q[qs[0]] &= ~m
        elif name == 'NEG':
            half ^= m
        elif name == 'BIT_STORE0':
            b[op.bs[0]] &= ~m
        elif name == 'BIT_STORE1':
            b[op.bs[0]] |= m
        elif name == 'BIT_INVERT':
            b[op.bs[0]] ^= m
        elif name == 'Z_POW':
            fired = q[qs[0]] & m
            cls = angle_class(op.angle)
            if cls == 'rotation':
                rot_ops.append(idx)
                rot_angles.append(op.angle)
                rot_bits.append(unpack(fired, shots))
                rotations += popcount(m)
            else:
                eighth += unpack(fired, shots) * int(op.angle * 4)
                t_gates += popcount(m) if cls == 't' else 0
        else:
            raise ValueError(f'unknown op {name} at {idx}')
        if track_life:
            for qi in qs:
                nz = bool(q[qi].any())
                if nz and qi not in life_start:
                    life_start[qi] = idx
                elif not nz and qi in life_start:
                    lifetimes[qi].append((life_start.pop(qi), idx + 1))
    if track_life:
        for qi, s in life_start.items():
            lifetimes[qi].append((s, len(circ.ops)))
    phase8 = ((eighth + 4 * unpack(half, shots)) % 8).astype(np.uint8)
    return SimResult(q, phase8, rot_ops, rot_angles,
                     np.array(rot_bits) if rot_bits else np.zeros((0, shots), bool),
                     toffolis, t_gates, rotations, lifetimes, reset_dirty)
