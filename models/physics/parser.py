import argparse
from os.path import join

DATAROOT = '/data/vision/oliva/scenedataset/urops/scenelayout/'
ROOT = '../..'
parser = argparse.ArgumentParser(description='Physics mass inference model.')

""""" DATA LOADING """""

parser.add_argument('--data-file', type=str, default=join(DATAROOT, '.physics_b_n3_e50_p21_f120_collide.h5'),
                    help='path to training data')
parser.add_argument('--num-points', type=int, default=-1,
                    help='max points to use. (-1 for no max)')
parser.add_argument('--test-points', type=int, default=1000,
                    help='num files to test.')
parser.add_argument('--max-samples', type=int,
                    help='max samples per group.')
parser.add_argument('--num-workers', type=int, default=16,
                    help='Num workers to load data')
parser.add_argument('--num-encode', type=int, default=50,
                    help='Number of timesteps to use to encode.')
parser.add_argument('--n-enc-frames', type=int, default=4,
                    help='Number of frames to combine during prediction at each time step.')
parser.add_argument('--n-frames', type=int, default=2,
                    help='Number of frames to combine during prediction at each time step.')
parser.add_argument('--rolling', action='store_true', help='Are encodings rolling?')
parser.add_argument('--max-pred-frames', type=int,
                    help='Max pred frames.')

""""" CHECKPOINTING """""

parser.add_argument('--base-log-folder', type=str, default=join(DATAROOT, 'runs/'),
                    help='Base log folder.')
parser.add_argument('--log-dir', type=str, default='',
                    help='Run-specific log directory.')
parser.add_argument('--continue-train', action="store_true",
                    help='Continue previous train.')
parser.add_argument('--render', action='store_true', help='Whether to render file.')

""""" HYPERPARAMETERS """""

parser.add_argument('--enc', action='store_true',
                    help='Whether to use EncNet.')
parser.add_argument('--pred', action='store_true',
                    help='Whether to use PredNet.')
parser.add_argument('--all', action='store_true',
                    help='Whether to use both enc net and pred net.')
parser.add_argument('--no-enc', action='store_true',
                    help='Whether to replace encoding with zeros.')

parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs')
parser.add_argument('--batch-size', type=int, default=1024,
                    help='batch size')
parser.add_argument('--loss-fn', type=str, default='mse',
                    help='Loss function')

parser.add_argument('--lr-enc', type=float, default=5e-4,
                    help='enc model learning rate')
parser.add_argument('--enc-model', type=str, default='lstm',
                    help='EncNet to use')
parser.add_argument('--enc-widths', type=int, default=[24],
                    nargs='+', help='EncNet widths')
parser.add_argument('--depth', type=int, default=4,
                    help='Depth of EncNet')
parser.add_argument('--trans-widths', type=int, default=[20, 20, 24],
                    nargs='+', help='TransNet widths')
parser.add_argument('--use-prior', action='store_true', help='Use a trainable prior for encoding?')

parser.add_argument('--lr-pred', type=float, default=5e-4,
                    help='pred model learning rate')
parser.add_argument('--code-size', type=int, default=64,
                    help='Size of code.')
parser.add_argument('--ro-discount', help="")
parser.add_argument('--noise', type=float, help="specify standard deviation of noise.")
parser.add_argument('--alpha', type=float, default=6e5,
                    help='pred model learning rate alpha decay')
parser.add_argument('--offsets', type=int, default=[1, 2, 4],
                    nargs='+', help='The timestep offset values.')
parser.add_argument('--non-rollout', action='store_true', default=False,
                    help='Set whether to only use non-rollout supervision signal.')
parser.add_argument('--non-ro-weight', type=float, default=0.,
                    help='Set whether to use non-rollout supervision signal.')
parser.add_argument('--predict-delta', action='store_true',
                    help='Whether to predict the delta of position or velocity.')
parser.add_argument('--beta', type=float,
                    help='Higher beta indicates longer periods of timestep discounting.')

""""" MISCELLANEOUS """""

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--progbar', action='store_true',
                    help='Progress bar')

