"""Bit-plane encoder / decoder (BPE) following the structure of CCSDS 122.0-B, Section 4.

What matches the standard
-------------------------
* Segments of S blocks, coded independently, each starting with a header.
* DC coefficients quantized by the factor q (Section 4.3.1.3) and coded
  differentially with Rice codes per gaggle of 16 blocks; the remaining DC bits
  between q-1 and BitDepthAC are sent next, the lower ones in stage 0.
* Per-block AC bit depths (BitDepthAC_Blk), coded differentially per gaggle.
* Bit-planes from BitDepthAC-1 down to 0, each with stage 0 (DC bit), stages
  1-3 (significance of parents, children, grandchildren, using the tranB,
  tranD, tranG and tranH "any descendant became significant" words) and
  stage 4 (refinement bits of all previously significant coefficients,
  parents first, then children, then grandchildren).
* The bitstream is embedded: any prefix of a segment can be decoded.

Simplifications (not bit-exact with the Blue Book)
--------------------------------------------------
* Header layout is custom (32-bit segment index, 24-bit block count).
* Significance words are sent uncoded instead of with the variable-length
  word codes of Section 4.5.3, so the compression ratio is somewhat lower.
* DC/depth differences use a zigzag mapping instead of the standard's mapping.

The decoder keeps, for every coefficient, the lowest bit-plane p whose bit has
been received. All bits at planes >= p are known, so the true value lies in an
interval of width 2**p. This is what the infill methods use as constraints.
"""

from __future__ import annotations

import numpy as np
from numba import njit

HEADER_BITS = 72
UNCODED = 31          # Rice option meaning "values sent raw"
GAGGLE = 16
P_UNKNOWN = 99        # p value for coefficients with no information at all

# --------------------------------------------------------------------------
# Bit I/O (MSB first)
# --------------------------------------------------------------------------


@njit(cache=True)
def _wbits(buf, pos, val, n):
    for i in range(n - 1, -1, -1):
        if (val >> i) & 1:
            buf[pos >> 3] |= np.uint8(0x80 >> (pos & 7))
        pos += 1
    return pos


@njit(cache=True)
def _rbits(buf, pos, n, total):
    """Read n bits. Returns (value, new_pos); new_pos = -1 if not enough data."""
    if pos < 0 or pos + n > total:
        return 0, -1
    v = 0
    for _ in range(n):
        v = (v << 1) | ((buf[pos >> 3] >> (7 - (pos & 7))) & 1)
        pos += 1
    return v, pos


@njit(cache=True)
def _bitlen(v):
    n = 0
    while v > 0:
        v >>= 1
        n += 1
    return n


@njit(cache=True)
def _zigzag(d):
    return 2 * d if d >= 0 else -2 * d - 1


@njit(cache=True)
def _unzigzag(z):
    return z >> 1 if (z & 1) == 0 else -((z + 1) >> 1)


@njit(cache=True)
def _sext(v, n):
    """Sign-extend an n-bit two's complement value."""
    if n > 0 and (v >> (n - 1)) & 1:
        return v - (1 << n)
    return v


# --------------------------------------------------------------------------
# Rice coding of a list of non-negative values, gaggle by gaggle
# --------------------------------------------------------------------------


@njit(cache=True)
def _rice_encode(buf, pos, vals, nraw):
    n = vals.shape[0]
    for g0 in range(0, n, GAGGLE):
        g1 = min(g0 + GAGGLE, n)
        best_k = UNCODED
        best_cost = nraw * (g1 - g0)
        for k in range(0, min(nraw, UNCODED)):
            cost = 0
            for i in range(g0, g1):
                cost += (vals[i] >> k) + 1 + k
            if cost < best_cost:
                best_cost = cost
                best_k = k
        pos = _wbits(buf, pos, best_k, 5)
        for i in range(g0, g1):
            if best_k == UNCODED:
                pos = _wbits(buf, pos, vals[i], nraw)
            else:
                pos += vals[i] >> best_k            # zeros
                pos = _wbits(buf, pos, 1, 1)
                pos = _wbits(buf, pos, vals[i] & ((1 << best_k) - 1), best_k)
    return pos


@njit(cache=True)
def _rice_decode_one(buf, pos, k, nraw, total):
    if k == UNCODED:
        return _rbits(buf, pos, nraw, total)
    q = 0
    while True:
        b, pos = _rbits(buf, pos, 1, total)
        if pos < 0:
            return 0, -1
        if b == 1:
            break
        q += 1
    r, pos = _rbits(buf, pos, k, total)
    if pos < 0:
        return 0, -1
    return (q << k) | r, pos


# --------------------------------------------------------------------------
# Helpers shared by encoder and decoder
# --------------------------------------------------------------------------


@njit(cache=True)
def dc_quant_factor(bdc, bac):
    """Quantization factor q of CCSDS 122.0-B, Section 4.3.1.3."""
    half = 1 + bac // 2
    if bdc <= 3:
        qp = 0
    elif bdc - half <= 1:
        qp = bdc - 3
    elif bdc - half > 10:
        qp = bdc - 10
    else:
        qp = half
    return max(qp, bac)


# --------------------------------------------------------------------------
# Encoder
# --------------------------------------------------------------------------


@njit(cache=True)
def encode_segment(blk, seg_index, start_flag, end_flag):
    """Encode one segment. blk: (S, 64) int64 weighted coefficients.

    Returns (buffer, n_bytes).
    """
    S = blk.shape[0]
    a = np.abs(blk)

    # ---- bit depths ---------------------------------------------------------
    bdc = 1
    for m in range(S):
        dc = blk[m, 0]
        while not (-(1 << (bdc - 1)) <= dc <= (1 << (bdc - 1)) - 1):
            bdc += 1
    bd = np.zeros(S, np.int64)
    for m in range(S):
        mx = 0
        for k in range(1, 64):
            if a[m, k] > mx:
                mx = a[m, k]
        bd[m] = _bitlen(mx)
    bac = 0
    for m in range(S):
        bac = max(bac, bd[m])
    q = dc_quant_factor(bdc, bac)

    nbits_bound = HEADER_BITS + S * (209 * bac + 160) + 64 * (S // GAGGLE + 2) * 2
    buf = np.zeros(nbits_bound // 8 + 16, np.uint8)
    pos = 0

    # ---- header ---------------------------------------------------------------
    pos = _wbits(buf, pos, seg_index, 32)
    pos = _wbits(buf, pos, start_flag, 1)
    pos = _wbits(buf, pos, end_flag, 1)
    pos = _wbits(buf, pos, bdc, 6)
    pos = _wbits(buf, pos, bac, 6)
    pos = _wbits(buf, pos, S, 24)
    pos = HEADER_BITS

    # ---- quantized DC -----------------------------------------------------
    nq = max(bdc - q, 1)
    qdc = np.empty(S, np.int64)
    for m in range(S):
        qdc[m] = blk[m, 0] >> q
    pos = _wbits(buf, pos, qdc[0] & ((1 << nq) - 1), nq)
    if S > 1:
        diffs = np.empty(S - 1, np.int64)
        for m in range(1, S):
            diffs[m - 1] = _zigzag(qdc[m] - qdc[m - 1])
        pos = _rice_encode(buf, pos, diffs, nq + 2)

    # ---- additional DC bit-planes (q-1 .. bac) -------------------------------
    for b in range(q - 1, bac - 1, -1):
        for m in range(S):
            pos = _wbits(buf, pos, (blk[m, 0] >> b) & 1, 1)

    if bac == 0:
        return buf, (pos + 7) // 8

    # ---- AC bit depths per block --------------------------------------------
    nd = _bitlen(bac)
    pos = _wbits(buf, pos, bd[0], nd)
    if S > 1:
        diffs = np.empty(S - 1, np.int64)
        for m in range(1, S):
            diffs[m - 1] = _zigzag(bd[m] - bd[m - 1])
        pos = _rice_encode(buf, pos, diffs, nd + 2)

    # ---- group maxima for the transition words ------------------------------
    maxB = np.zeros(S, np.int64)
    maxD = np.zeros((S, 3), np.int64)
    maxG = np.zeros((S, 3), np.int64)
    maxH = np.zeros((S, 3, 4), np.int64)
    for m in range(S):
        for i in range(3):
            for j in range(4):
                c = a[m, 4 + 4 * i + j]
                maxD[m, i] = max(maxD[m, i], c)
                for k in range(4):
                    g = a[m, 16 + 16 * i + 4 * j + k]
                    maxH[m, i, j] = max(maxH[m, i, j], g)
                maxG[m, i] = max(maxG[m, i], maxH[m, i, j])
            maxD[m, i] = max(maxD[m, i], maxG[m, i])
            maxB[m] = max(maxB[m], maxD[m, i])

    sigp = np.full((S, 64), -1, np.int64)   # plane at which coefficient became significant
    Bd = np.zeros(S, np.bool_)
    Dd = np.zeros((S, 3), np.bool_)
    Gd = np.zeros((S, 3), np.bool_)
    Hd = np.zeros((S, 3, 4), np.bool_)

    for b in range(bac - 1, -1, -1):
        for m in range(S):
            # stage 0: DC bit
            pos = _wbits(buf, pos, (blk[m, 0] >> b) & 1, 1)
            if bd[m] <= b:
                continue
            # stage 1: parents
            for i in range(3):
                k = 1 + i
                if sigp[m, k] < 0:
                    s = (a[m, k] >> b) & 1
                    pos = _wbits(buf, pos, s, 1)
                    if s:
                        pos = _wbits(buf, pos, 1 if blk[m, k] < 0 else 0, 1)
                        sigp[m, k] = b
            # stage 2: children
            if not Bd[m]:
                s = 1 if (maxB[m] >> b) > 0 else 0
                pos = _wbits(buf, pos, s, 1)
                if s:
                    Bd[m] = True
            if not Bd[m]:
                continue
            for i in range(3):
                if not Dd[m, i]:
                    s = 1 if (maxD[m, i] >> b) > 0 else 0
                    pos = _wbits(buf, pos, s, 1)
                    if s:
                        Dd[m, i] = True
            for i in range(3):
                if Dd[m, i]:
                    for j in range(4):
                        k = 4 + 4 * i + j
                        if sigp[m, k] < 0:
                            s = (a[m, k] >> b) & 1
                            pos = _wbits(buf, pos, s, 1)
                            if s:
                                pos = _wbits(buf, pos, 1 if blk[m, k] < 0 else 0, 1)
                                sigp[m, k] = b
            # stage 3: grandchildren
            for i in range(3):
                if Dd[m, i] and not Gd[m, i]:
                    s = 1 if (maxG[m, i] >> b) > 0 else 0
                    pos = _wbits(buf, pos, s, 1)
                    if s:
                        Gd[m, i] = True
            for i in range(3):
                if not Gd[m, i]:
                    continue
                for j in range(4):
                    if not Hd[m, i, j]:
                        s = 1 if (maxH[m, i, j] >> b) > 0 else 0
                        pos = _wbits(buf, pos, s, 1)
                        if s:
                            Hd[m, i, j] = True
                for j in range(4):
                    if not Hd[m, i, j]:
                        continue
                    for kk in range(4):
                        k = 16 + 16 * i + 4 * j + kk
                        if sigp[m, k] < 0:
                            s = (a[m, k] >> b) & 1
                            pos = _wbits(buf, pos, s, 1)
                            if s:
                                pos = _wbits(buf, pos, 1 if blk[m, k] < 0 else 0, 1)
                                sigp[m, k] = b
        # stage 4: refinement (parents, then children, then grandchildren)
        for t in range(3):
            k0 = 1 if t == 0 else (4 if t == 1 else 16)
            k1 = 4 if t == 0 else (16 if t == 1 else 64)
            for m in range(S):
                for k in range(k0, k1):
                    if sigp[m, k] > b:
                        pos = _wbits(buf, pos, (a[m, k] >> b) & 1, 1)

    return buf, (pos + 7) // 8


# --------------------------------------------------------------------------
# Decoder
# --------------------------------------------------------------------------


@njit(cache=True)
def decode_segment(buf, n_blocks):
    """Decode a (possibly truncated) segment.

    Returns (ok_header, seg_index, mag, sgn, p) where, for each of the n_blocks
    blocks and 64 coefficients:

    * DC (k = 0): mag holds the signed value with all bits below p zero;
      p = P_UNKNOWN if the DC itself was not received.
    * AC: |value| = mag + r with 0 <= r < 2**p, sign sgn (0 if still unknown,
      which happens only when mag == 0).
    """
    total = buf.shape[0] * 8
    S = n_blocks
    mag = np.zeros((S, 64), np.int64)
    sgn = np.zeros((S, 64), np.int8)
    p = np.full((S, 64), P_UNKNOWN, np.int64)

    seg_index, pos = _rbits(buf, 0, 32, total)
    if pos < 0 or total < HEADER_BITS:
        return False, -1, mag, sgn, p
    pos = 32 + 2
    bdc, pos = _rbits(buf, pos, 6, total)
    bac, pos = _rbits(buf, pos, 6, total)
    hdr_S, pos = _rbits(buf, pos, 24, total)
    pos = HEADER_BITS
    if hdr_S != S:
        S = min(S, hdr_S)
    q = dc_quant_factor(bdc, bac)

    # From the header alone: |AC| < 2**bac.
    for m in range(S):
        for k in range(1, 64):
            p[m, k] = bac

    # ---- quantized DC -----------------------------------------------------
    nq = max(bdc - q, 1)
    v, pos = _rbits(buf, pos, nq, total)
    if pos < 0:
        return True, seg_index, mag, sgn, p
    prev = _sext(v, nq)
    mag[0, 0] = prev << q
    p[0, 0] = q
    m = 1
    while m < S:
        k, pos = _rbits(buf, pos, 5, total)
        if pos < 0:
            return True, seg_index, mag, sgn, p
        g1 = min(m + GAGGLE, S)
        while m < g1:
            z, pos = _rice_decode_one(buf, pos, k, nq + 2, total)
            if pos < 0:
                return True, seg_index, mag, sgn, p
            prev = prev + _unzigzag(z)
            mag[m, 0] = prev << q
            p[m, 0] = q
            m += 1

    for b in range(q - 1, bac - 1, -1):
        for m in range(S):
            v, pos = _rbits(buf, pos, 1, total)
            if pos < 0:
                return True, seg_index, mag, sgn, p
            mag[m, 0] += v << b
            p[m, 0] = b

    if bac == 0:
        return True, seg_index, mag, sgn, p

    # ---- AC bit depths ------------------------------------------------------
    nd = _bitlen(bac)
    bd = np.zeros(S, np.int64)
    v, pos = _rbits(buf, pos, nd, total)
    if pos < 0:
        return True, seg_index, mag, sgn, p
    bd[0] = v
    m = 1
    while m < S:
        k, pos = _rbits(buf, pos, 5, total)
        if pos < 0:
            return True, seg_index, mag, sgn, p
        g1 = min(m + GAGGLE, S)
        while m < g1:
            z, pos = _rice_decode_one(buf, pos, k, nd + 2, total)
            if pos < 0:
                return True, seg_index, mag, sgn, p
            bd[m] = bd[m - 1] + _unzigzag(z)
            m += 1
    for m in range(S):
        for k in range(1, 64):
            p[m, k] = bd[m]

    sig = np.zeros((S, 64), np.bool_)
    sigp = np.full((S, 64), -1, np.int64)
    Bd = np.zeros(S, np.bool_)
    Dd = np.zeros((S, 3), np.bool_)
    Gd = np.zeros((S, 3), np.bool_)
    Hd = np.zeros((S, 3, 4), np.bool_)

    for b in range(bac - 1, -1, -1):
        for m in range(S):
            v, pos = _rbits(buf, pos, 1, total)
            if pos < 0:
                return True, seg_index, mag, sgn, p
            mag[m, 0] += v << b
            p[m, 0] = b
            if bd[m] <= b:
                continue
            # stage 1
            for i in range(3):
                k = 1 + i
                if not sig[m, k]:
                    s, pos = _rbits(buf, pos, 1, total)
                    if pos < 0:
                        return True, seg_index, mag, sgn, p
                    if s:
                        sb, pos = _rbits(buf, pos, 1, total)
                        if pos < 0:
                            return True, seg_index, mag, sgn, p
                        sig[m, k] = True
                        sigp[m, k] = b
                        mag[m, k] = 1 << b
                        sgn[m, k] = -1 if sb else 1
                    p[m, k] = b
            # stage 2
            if not Bd[m]:
                s, pos = _rbits(buf, pos, 1, total)
                if pos < 0:
                    return True, seg_index, mag, sgn, p
                if s:
                    Bd[m] = True
                else:
                    for k in range(4, 64):
                        p[m, k] = b
                    continue
            for i in range(3):
                if not Dd[m, i]:
                    s, pos = _rbits(buf, pos, 1, total)
                    if pos < 0:
                        return True, seg_index, mag, sgn, p
                    if s:
                        Dd[m, i] = True
                    else:
                        for j in range(4):
                            p[m, 4 + 4 * i + j] = b
                        for k in range(16 + 16 * i, 32 + 16 * i):
                            p[m, k] = b
            for i in range(3):
                if Dd[m, i]:
                    for j in range(4):
                        k = 4 + 4 * i + j
                        if not sig[m, k]:
                            s, pos = _rbits(buf, pos, 1, total)
                            if pos < 0:
                                return True, seg_index, mag, sgn, p
                            if s:
                                sb, pos = _rbits(buf, pos, 1, total)
                                if pos < 0:
                                    return True, seg_index, mag, sgn, p
                                sig[m, k] = True
                                sigp[m, k] = b
                                mag[m, k] = 1 << b
                                sgn[m, k] = -1 if sb else 1
                            p[m, k] = b
            # stage 3
            for i in range(3):
                if Dd[m, i] and not Gd[m, i]:
                    s, pos = _rbits(buf, pos, 1, total)
                    if pos < 0:
                        return True, seg_index, mag, sgn, p
                    if s:
                        Gd[m, i] = True
                    else:
                        for k in range(16 + 16 * i, 32 + 16 * i):
                            if not sig[m, k]:
                                p[m, k] = b
            for i in range(3):
                if not Gd[m, i]:
                    continue
                for j in range(4):
                    if not Hd[m, i, j]:
                        s, pos = _rbits(buf, pos, 1, total)
                        if pos < 0:
                            return True, seg_index, mag, sgn, p
                        if s:
                            Hd[m, i, j] = True
                        else:
                            for kk in range(4):
                                k = 16 + 16 * i + 4 * j + kk
                                if not sig[m, k]:
                                    p[m, k] = b
                for j in range(4):
                    if not Hd[m, i, j]:
                        continue
                    for kk in range(4):
                        k = 16 + 16 * i + 4 * j + kk
                        if not sig[m, k]:
                            s, pos = _rbits(buf, pos, 1, total)
                            if pos < 0:
                                return True, seg_index, mag, sgn, p
                            if s:
                                sb, pos = _rbits(buf, pos, 1, total)
                                if pos < 0:
                                    return True, seg_index, mag, sgn, p
                                sig[m, k] = True
                                sigp[m, k] = b
                                mag[m, k] = 1 << b
                                sgn[m, k] = -1 if sb else 1
                            p[m, k] = b
        # stage 4
        for t in range(3):
            k0 = 1 if t == 0 else (4 if t == 1 else 16)
            k1 = 4 if t == 0 else (16 if t == 1 else 64)
            for m in range(S):
                for k in range(k0, k1):
                    if sigp[m, k] > b:
                        v, pos = _rbits(buf, pos, 1, total)
                        if pos < 0:
                            return True, seg_index, mag, sgn, p
                        mag[m, k] |= v << b
                        p[m, k] = b

    return True, seg_index, mag, sgn, p
