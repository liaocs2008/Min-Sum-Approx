import numpy as np
from scipy.linalg import circulant


def offset_min_sum(v2c, offset=0.625):
    assert len(v2c.shape) == 2
    circulant_index = circulant(np.arange(v2c.shape[-1])).T[1:]
    tmp1 = v2c[:, circulant_index]
    tmp2 = np.prod(np.sign(tmp1), axis=1)
    new_c2v = np.min(np.abs(tmp1), axis=1) - offset
    new_c2v[new_c2v < 0] = 0
    ans = new_c2v * tmp2
    return ans


def scaled_min_sum(v2c, coefficient=0.9375):
    assert len(v2c.shape) == 2
    circulant_index = circulant(np.arange(v2c.shape[-1])).T[1:]
    tmp1 = v2c[:, circulant_index]
    tmp2 = np.prod(np.sign(tmp1), axis=1)
    ans = np.min(np.abs(tmp1), axis=1) * tmp2 * coefficient
    return ans


def generate_flooding(h_matrix):
    m, n = h_matrix.shape
    n_c = {}  # given a node c, tell all neighbors (v) of c
    for c in range(m):
        res = np.where(h_matrix[c] > 0)[0]
        if res.shape[0]:  # ensure non-empty
            n_c[c] = res

    n_v = {}
    for v in range(n):
        res = np.where(h_matrix[:, v] > 0)[0]
        if res.shape[0]:
            n_v[v] = res

    edges = np.count_nonzero(h_matrix)
    return edges, n_v, n_c


def flooding(lv, n_v, n_c, m, n, iterations, f):
    assert lv.shape[1] == n
    batch = lv.shape[0]
    sv = np.copy(lv)
    msg = np.zeros([batch, m, n])

    for t in range(iterations):
        # msg storing c2v is now getting v2c
        for v in n_v:
            # compute sv
            ind_c = n_v[v]
            tmp_c2v = msg[:, ind_c, v]  # c2v
            sv[:, v] = lv[:, v] + tmp_c2v.sum(axis=1)

            # compute v2c
            msg[:, ind_c, v] = sv[:, v].reshape([batch, -1]) - tmp_c2v

        # msg storing v2c is now getting c2v
        for c in n_c:
            ind_v = n_c[c]
            if len(ind_v) == 1:
                msg[:, c, ind_v] = 0
            else:
                msg[:, c, ind_v] = f(msg[:, c, ind_v])

    return sv


def awgn_channel(x, code_rate, snr_in_db, noise):
    assert 1 > code_rate > 0
    s = 1 - 2 * x  # 0->1, 1->-1, x~B(1,0.5), E(s)=0, D(s)=4D(x)=1
    snr = 10 ** (snr_in_db / 10.0)
    sigma = np.sqrt(1 / (2.0 * snr * code_rate))
    y = s + noise * sigma  # E(y)=0, D(y)=1+sigma**2
    lv = 2 * y / (sigma ** 2)
    return lv


def encode(g_matrix, v, snr_in_db, noise):
    x = v.dot(g_matrix) % 2
    r = v.shape[1] / x.shape[1]  # code rate
    lv = awgn_channel(x, r, snr_in_db, noise)
    return lv, x


def decode(lv, info, m, n, t, f):
    edges, n_v, n_c = info
    d = flooding(lv, n_v, n_c, m, n, t, f)
    binary_d = (d < 0).astype(np.int)
    return binary_d
