import argparse
from functools import partial
import numpy as np
from utils import *

parser = argparse.ArgumentParser("Belief Propagation Decoder")
parser.add_argument('--B', type=int, default=64, help='Batch size')
parser.add_argument('--C', type=str, default='bp', help='bp or lch')
parser.add_argument('--F', type=str, default='offset', help='Functions to select')
parser.add_argument('--G', type=str, default='./hmatrix/BCH_63_51_2_strip_G.npy', help='Generator matrix')
parser.add_argument('--H', type=str, default='./hmatrix/BCH_63_51_2_strip_H.npy', help='Parity Check matrix')
parser.add_argument('--K', type=int, default=51, help='Message length')
parser.add_argument('--N', type=int, default=63, help='Code length')
parser.add_argument('--T', type=int, default=5, help='Iterations')
parser.add_argument('--X', type=int, default=1000000, help='Maximum number of test')
parser.add_argument('--Y', type=int, default=100, help='Least number of error frame')
parser.add_argument('--s_snr', default=1.0, type=float, help='SNR start value in dB')
parser.add_argument('--e_snr', default=6.0, type=float, help='SNR end value in dB')
parser.add_argument('--step', default=1.0, type=float, help='SNR step in dB')
parser.add_argument('--offset', default=0.3, type=float, help='Offset min sum value')
parser.add_argument('--coefficient', default=0.9375, type=float, help='Scaled min sum value')
parser.add_argument('--seed', type=int, default=2020, help='Random seed')


if __name__ == "__main__":
    args = parser.parse_args()
    
    # load given matrix
    G = np.load(args.G)
    H = np.load(args.H)
    assert np.all(0 == (G.dot(H.T) % 2)) and np.all((args.K, args.N) == G.shape)
    
    # prepare data
    np.random.seed(args.seed)
    
    # generate info
    info = generate_flooding(H)
    
    # select functions
    if args.F == "offset":
        f = partial(offset_min_sum, offset=args.offset)
    elif args.F == "scaled":
        f = partial(scaled_min_sum, coef=args.coef)
    else:
        f = None
    
    # start running
    for SNR_in_db in np.arange(args.s_snr, args.e_snr+args.step, args.step):
        num_batches = 0
        diff_ber = 0
        diff_fer = 0
        while diff_fer < args.Y and num_batches * args.B < args.X:
            me = np.random.randint(2, size=[args.B, args.K])
            no = np.random.randn(args.B, args.N)
            lv, x = encode(G, me, SNR_in_db, no)
            #print("llr", Lch.sum())
            if args.C == 'bp':
                binary_d = decode(lv, info, H.shape[1], args.N, args.T, f)
            else:
                binary_d = (lv < 0).astype(int)
            res = binary_d != x  # (batch, N)
            
            num_batches += 1
            diff_ber += res.astype(int).sum()
            diff_fer += np.sum((np.sum(res, axis=1) > 0).astype(int))
        
        # final output
        BER = diff_ber / (num_batches * args.B * args.N)
        FER = diff_fer / (num_batches * args.B)
        out = "N= {} K= {} SNR_in_db= {:.2f} BER= {:.6f} FER= {:.6f}".format(
              args.N, args.K, SNR_in_db, BER, FER)
        if args.F == 'offset':
            print(args.F, args.offset, out)
        else:
            print(args.F, args.coefficient, out)
