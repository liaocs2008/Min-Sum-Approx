import numpy as np
from scipy.linalg import circulant 

def offset_min_sum(v2c, offset=0.625):
    assert len(v2c.shape) == 2
    circ_ind = circulant(np.arange(v2c.shape[-1])).T[1:]
    tmp1 = v2c[:, circ_ind]
    tmp2 = np.prod(np.sign(tmp1), axis=1)
    new_c2v = np.min(np.abs(tmp1), axis=1) - offset
    new_c2v[new_c2v < 0] = 0
    ans = new_c2v * tmp2
    return ans


def generate_flooding(H):
    M, N = H.shape
    n_c = {} # given a node c, tell all neighbors (v) of c
    for c in range(M):
        res = np.where(H[c] > 0)[0]
        if res.shape[0]: # ensure non-empty
            n_c[c] = res
    
    n_v = {}
    for v in range(N):
        res = np.where(H[:,v] > 0)[0]
        if res.shape[0]:
            n_v[v] = res
    
    edges = np.count_nonzero(H)
    return edges, n_v, n_c


def flooding(lv, edges, n_v, n_c, M, N, T, f):
    assert lv.shape[1] == N
    batch = lv.shape[0]
    sv = np.copy(lv)
    msg = np.zeros([batch, M, N])
    
    for t in range(T):
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


def AWGN_channel(x, code_rate, SNR_in_db, noise):
    assert code_rate < 1 and code_rate > 0
    s = 1 - 2 * x   # 0->1, 1->-1, x~B(1,0.5), E(s)=0, D(s)=4D(x)=1
    SNR = 10 ** (SNR_in_db / 10.0)
    sigma = np.sqrt(1 / (2.0 * SNR * code_rate))
    y = s + noise * sigma  # E(y)=0, D(y)=1+sigma**2
    Lch = 2 * y / (sigma**2)
    return Lch


def encode(G, v, SNR_in_db, noise):
    x = v.dot(G) % 2
    R = v.shape[1] / x.shape[1]  # code rate
    Lch = AWGN_channel(x, R, SNR_in_db, noise)
    return Lch, x


def decode(Lch, info, M, N, T, f):
    edges, n_v, n_c = info
    d = flooding(Lch, edges, n_v, n_c, M, N, T, f)
    binary_d = (d < 0).astype(np.int)
    return binary_d
