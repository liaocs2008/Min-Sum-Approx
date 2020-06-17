import argparse
from functools import partial
import numpy as np
from utils import *

parser = argparse.ArgumentParser("Belief Propagation Decoder")
parser.add_argument('--B', type=int, default=64, help='Batch size')
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
            Lch, x = encode(G, me, SNR_in_db, no)
            binary_d = decode(Lch, info, args.K, args.N, args.T, f)
            res = binary_d != x  # (batch, N)
            
            num_batches += 1
            diff_ber += res.sum()
            diff_fer += np.sum( np.sum(res, axis=1) > 0 )
        
        # final output
        BER = diff_ber / (num_batches * args.B * args.N)
        FER = diff_fer / (num_batches * args.B)
        print(args.F, args.offset, "N={} K={} SNR_in_db={:.2f} BER={:.6f} FER={:.6f}".format(
              args.N, args.K, SNR_in_db, BER, FER))
