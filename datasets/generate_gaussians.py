'''Generate Gaussian-distributed vectors (X, Y).
'''

import os
import sys
import numpy as np
import argparse

parser = argparse.ArgumentParser('Generate Gaussian input-label pairs.')
parser.add_argument('--output-dir', type=str, default='./gaussian')
parser.add_argument('--x-size', type=int, default=5)
parser.add_argument('--y-size', type=int, default=3)
parser.add_argument('--num-groups', type=int, default=10000)
parser.add_argument('--num-samples', help="Number of samples per group.", type=int, default=100)
parser.add_argument('--plot', action="store_true", help="Plot data?")
parser.add_argument('--add-bias', action="store_true", help="Add bias?")
parser.add_argument('--add-noise', type=int, default=3, help="Add noise?")
args = parser.parse_args()

if args.plot:
    import matplotlib.pyplot as plt

if __name__ == '__main__':
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for i in range(args.num_groups):
        W = np.random.uniform(low=-20, high=20, size=(args.x_size, args.y_size))

        x = np.random.uniform(low=-10, high=10, size=(args.num_samples, args.x_size))
        y = np.matmul(x, W)

        if args.add_bias:
            b = np.random.uniform(low=-20, high=20, size=args.y_size)
            y += b

        if args.add_noise:
            y += np.random.normal(loc=0, scale=args.add_noise, size=y.shape) # Add noise

        if args.plot:
            plt.plot(y[:,0], y[:,1], 'o')
        else:
            output_path = os.path.join(args.output_dir, 'group_'+str(i)+'.npz')
            if args.add_bias:
                np.savez(output_path, x_size=args.x_size, y_size=args.y_size,
                    W=W, b=b, x=x, y=y)
            else:
                np.savez(output_path, x_size=args.x_size, y_size=args.y_size,
                    W=W, x=x, y=y)

    if args.plot:
        plt.show()
