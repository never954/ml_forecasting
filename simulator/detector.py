"""
High-order-entropy detector (Aguera y Arcas et al., 2024).

high_order_entropy = Shannon entropy per byte  -  normalized Kolmogorov complexity

Kolmogorov complexity is approximated by a real compressor (zlib), which is a
valid upper bound on K. For a random soup, bytes are ~uniform (Shannon ~8 bits)
AND incompressible (K/n ~8 bits), so HOE ~ 0. When self-replicators take over,
the soup fills with repeated structure: it compresses far below its byte entropy,
so K/n drops and HOE rises sharply. That rise is the transition signature used
as the ground-truth label in this project.
"""
import zlib
import numpy as np


def shannon_bits_per_byte(soup):
    counts = np.bincount(soup, minlength=256).astype(np.float64)
    total = counts.sum()
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def kolmogorov_bits_per_byte(soup, level=6):
    comp = len(zlib.compress(soup.tobytes(), level))
    return comp * 8.0 / soup.shape[0]


def high_order_entropy(soup):
    """Return (HOE, shannon, kolmogorov) in bits per byte."""
    h = shannon_bits_per_byte(soup)
    k = kolmogorov_bits_per_byte(soup)
    return h - k, h, k
