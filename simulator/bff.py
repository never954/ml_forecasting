"""
BFF (extended Brainfuck) substrate — from-scratch reproduction of the
"Computational Life" primordial soup (Aguera y Arcas et al., 2024, arXiv:2406.19108).

Programs are `plen`-byte tapes. Two programs are concatenated into one shared
tape and executed as BFF; both halves can be modified (self-modifying code).
No fitness function, no selection, no seeded replicator — self-replicators emerge
purely from interaction. This is Stage 1 of our project: reproduce the substrate.

Instruction set (byte value -> action); every other byte is a no-op:
    <  (60)  head0 -= 1
    >  (62)  head0 += 1
    {  (123) head1 -= 1
    }  (125) head1 += 1
    -  (45)  tape[head0] -= 1
    +  (43)  tape[head0] += 1
    .  (46)  tape[head1] = tape[head0]     (copy from read head to write head)
    ,  (44)  tape[head0] = tape[head1]     (copy from write head to read head)
    [  (91)  if tape[head0] == 0: jump forward past matching ]
    ]  (93)  if tape[head0] != 0: jump back  to  matching [
Pointers wrap modulo tape length; byte arithmetic wraps modulo 256.
Execution stops after `max_steps` instructions or on an unmatched bracket.
"""
import numpy as np
from numba import njit

# Instruction byte codes
LT, GT, LB, RB = 60, 62, 123, 125        # < > { }
DEC, INC = 45, 43                         # - +
COPY0TO1, COPY1TO0 = 46, 44               # . ,
JZ, JNZ = 91, 93                          # [ ]


@njit(cache=True)
def run_pair(tape, max_steps):
    """Execute a single concatenated tape in place. Returns steps executed."""
    L = tape.shape[0]
    ip = 0
    h0 = 0
    h1 = 0
    steps = 0
    while steps < max_steps:
        instr = tape[ip]
        if instr == LT:
            h0 = (h0 - 1) % L
        elif instr == GT:
            h0 = (h0 + 1) % L
        elif instr == LB:
            h1 = (h1 - 1) % L
        elif instr == RB:
            h1 = (h1 + 1) % L
        elif instr == DEC:
            tape[h0] = (tape[h0] - 1) & 255
        elif instr == INC:
            tape[h0] = (tape[h0] + 1) & 255
        elif instr == COPY0TO1:
            tape[h1] = tape[h0]
        elif instr == COPY1TO0:
            tape[h0] = tape[h1]
        elif instr == JZ:
            if tape[h0] == 0:
                depth = 1
                j = ip + 1
                found = False
                while j < L:
                    if tape[j] == JZ:
                        depth += 1
                    elif tape[j] == JNZ:
                        depth -= 1
                        if depth == 0:
                            found = True
                            break
                    j += 1
                if not found:
                    break
                ip = j
        elif instr == JNZ:
            if tape[h0] != 0:
                depth = 1
                j = ip - 1
                found = False
                while j >= 0:
                    if tape[j] == JNZ:
                        depth += 1
                    elif tape[j] == JZ:
                        depth -= 1
                        if depth == 0:
                            found = True
                            break
                    j -= 1
                if not found:
                    break
                ip = j
        ip += 1
        if ip >= L:
            ip = 0
        steps += 1
    return steps


@njit(cache=True)
def run_epoch(soup, plen, order, max_steps):
    """One epoch: pair programs per `order` and let each pair interact."""
    n_pairs = order.shape[0] // 2
    tape = np.empty(2 * plen, dtype=np.uint8)
    for k in range(n_pairs):
        a = order[2 * k]
        b = order[2 * k + 1]
        for i in range(plen):
            tape[i] = soup[a * plen + i]
            tape[plen + i] = soup[b * plen + i]
        run_pair(tape, max_steps)
        for i in range(plen):
            soup[a * plen + i] = tape[i]
            soup[b * plen + i] = tape[plen + i]


def make_soup(n_programs, plen, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=n_programs * plen, dtype=np.uint8), rng


def mutate(soup, rng, rate):
    """Random background mutation: flip a Poisson number of bytes to random values."""
    if rate <= 0:
        return
    k = rng.poisson(rate * soup.shape[0])
    if k > 0:
        idx = rng.integers(0, soup.shape[0], size=k)
        soup[idx] = rng.integers(0, 256, size=k, dtype=np.uint8)
