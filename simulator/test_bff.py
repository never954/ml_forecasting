"""
Correctness tests for the BFF interpreter.

Two kinds of evidence:
  1. Targeted semantic tests — hand-built tapes with a known outcome, respecting
     that code and data share one tape (head0 starts on the code, so operations
     that touch head0 modify the program itself; the tests account for this).
  2. Fuzz equivalence — a slow, deliberately transparent reference interpreter is
     run against the fast numba interpreter on thousands of random tapes; every
     final tape must match byte-for-byte. This certifies the fast engine against
     a readable implementation of the paper's instruction set.

Run: python simulator/test_bff.py
"""
import numpy as np
from bff import run_pair, LT, GT, LB, RB, DEC, INC, COPY0TO1, COPY1TO0, JZ, JNZ


# ---------------------------------------------------------------- reference
def run_pair_reference(tape, max_steps):
    """Slow, obvious reimplementation of BFF semantics for cross-checking."""
    L = len(tape)
    ip = h0 = h1 = steps = 0
    while steps < max_steps:
        c = tape[ip]
        if c == LT:   h0 = (h0 - 1) % L
        elif c == GT: h0 = (h0 + 1) % L
        elif c == LB: h1 = (h1 - 1) % L
        elif c == RB: h1 = (h1 + 1) % L
        elif c == DEC: tape[h0] = (int(tape[h0]) - 1) & 255
        elif c == INC: tape[h0] = (int(tape[h0]) + 1) & 255
        elif c == COPY0TO1: tape[h1] = tape[h0]
        elif c == COPY1TO0: tape[h0] = tape[h1]
        elif c == JZ:
            if tape[h0] == 0:
                depth, j, found = 1, ip + 1, False
                while j < L:
                    if tape[j] == JZ: depth += 1
                    elif tape[j] == JNZ:
                        depth -= 1
                        if depth == 0: found = True; break
                    j += 1
                if not found: break
                ip = j
        elif c == JNZ:
            if tape[h0] != 0:
                depth, j, found = 1, ip - 1, False
                while j >= 0:
                    if tape[j] == JNZ: depth += 1
                    elif tape[j] == JZ:
                        depth -= 1
                        if depth == 0: found = True; break
                    j -= 1
                if not found: break
                ip = j
        ip = (ip + 1) % L
        steps += 1
    return tape


# ---------------------------------------------------------------- targeted
def test_increment_moves_off_code_then_increments():
    # '>' moves head0 to cell 1 (a noop data byte = 1); '+' then increments cell 1.
    t = np.array([GT, 1, INC, 0, 0, 0, 0, 0], dtype=np.uint8)
    run_pair(t, 3)
    assert t[1] == 2, t[1]            # 1 -> 2


def test_copy_from_read_head_to_write_head():
    # cell 0 holds data 7 (noop); '}' moves write head to 1; '.' copies cell0 -> cell1.
    t = np.array([7, RB, COPY0TO1, 0, 0, 0, 0, 0], dtype=np.uint8)
    run_pair(t, 3)
    assert t[1] == 7, t[1]


def test_clear_loop_bracket_matching():
    # cell 0 holds count 2 (noop); "[ - ]" decrements it to zero. Tests nesting-aware jumps.
    t = np.array([2, JZ, DEC, JNZ, 0, 0, 0, 0], dtype=np.uint8)
    run_pair(t, 20)
    assert t[0] == 0, t[0]


def test_copy_a_data_block_to_a_new_location():
    # The self-replication primitive, made explicit (no loop, so no self-overwrite):
    # a 2-byte data block [7, 9] in cells 0-1 is copied into cells 4-5 by advancing
    # the write head and issuing '.' copies. Duplicating a block into fresh cells is
    # exactly what an emergent replicator must do.
    t = np.array([7, 9, RB, RB, RB, RB, COPY0TO1, GT, RB, COPY0TO1, 0, 0, 0, 0, 0, 0],
                 dtype=np.uint8)
    run_pair(t, 10)
    assert t[4] == 7 and t[5] == 9, (t[4], t[5])


# ---------------------------------------------------------------- fuzz
def test_fuzz_equivalence_with_reference():
    rng = np.random.default_rng(12345)
    trials = 3000
    for _ in range(trials):
        L = int(rng.integers(8, 40))
        base = rng.integers(0, 256, size=L, dtype=np.uint8)
        steps = int(rng.integers(0, 500))
        a = base.copy(); b = base.copy()
        run_pair(a, steps)
        run_pair_reference(b, steps)
        if not np.array_equal(a, b):
            raise AssertionError(f"mismatch on L={L} steps={steps}")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
