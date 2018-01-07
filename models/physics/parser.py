import argparse
from os.path import join

# DATAROOT = '/om/user/dzd123/'
DATAROOT = '/data/vision/oliva/scenedataset/urops/scenelayout/'
ROOT = '../..'
parser = argparse.ArgumentParser(description='Physics mass inference model.')

""""" DATA LOADING """""

parser.add_argument('--data-file', type=str, default=join(DATAROOT, 'physics_b_n3_e50_p25_f120_collide_three.h5'),
                    help='path to training data')
parser.add_argument('--num-points', type=int, default=-1,
                    help='max points to use. (-1 for no max)')
parser.add_argument('--test-points', type=int, default=1000,
                    help='num files to test.')
parser.add_argument('--max-samples', type=int,
                    help='max samples per group.')
parser.add_argument('--frames-per-samp', type=int, default=2,
                    help='Number of frames to combine during prediction at each time step.')
# parser.add_argument('--max-pred-frames', type=int,
#                     help='Max pred frames.')

""""" CHECKPOINTING """""

# parser.add_argument('--base-log-folder', type=str, default=join(DATAROOT, 'runs/'),
#                     help='Base log folder.')
parser.add_argument('--log-dir', type=str, default=join(DATAROOT, 'logs/'),
                    help='Run-specific log directory.')
parser.add_argument('--ckpt-dir', type=str, default=join(DATAROOT, 'ckpt/'),
                    help='Checkpoint directory.')
parser.add_argument('--restore', type=str,
                    help='Restore a specific checkpoint.')
parser.add_argument('--new-dir', action='store_true',
                    help='Used with --restore: places results into a new directory.')
parser.add_argument('--save-all', action='store_true',
                    help='Save all epochs, rather than just the best ones.')
# parser.add_argument('--continue-train', action="store_true",
#                     help='Continue previous train.')
# parser.add_argument('--render', action='store_true', help='Whether to render file.')

""""" HYPERPARAMETERS """""

parser.add_argument('--enc-only', action='store_true',
                    help='Whether to use EncNet only.')
# parser.add_argument('--pred', action='store_true',
#                     help='Whether to use PredNet.')
# parser.add_argument('--all', action='store_true',
#                     help='Whether to use both enc net and pred net.')
# parser.add_argument('--no-enc', action='store_true',
#                     help='Whether to replace encoding with zeros.')
parser.add_argument('--freeze-encs', action='store_true',
                    help='Whether to freeze training of encodings.')
parser.add_argument('--calc-encs', action='store_true',
                    help='Whether to save encodings.')
parser.add_argument('--logy', action='store_true',
                    help='Whether to regress on log(mass) or mass.')
parser.add_argument('--runtime', action='store_true',
                    help='Whether to record runtimes.')
parser.add_argument('--baseline', action='store_true',
                    help='Whether to calculate baseline loss.')
parser.add_argument('--long', action='store_true',
                    help='Whether to save a long rollout.')

parser.add_argument('--epochs', type=int, default=500,
                    help='number of epochs')
parser.add_argument('--batch-size', type=int, default=1024,
                    help='batch size')
parser.add_argument('--loss-fn', type=str, default='mse',
                    help='Loss function')
parser.add_argument('--lr-enc', type=float, default=5e-4,
                    help='enc model learning rate')
parser.add_argument('--lr-pred', type=float, default=5e-4,
                    help='pred model learning rate')
parser.add_argument('--decay-cutoff', type=int, default=-1,
                    help='if loss does not improve for this number of epochs, both'
                         'learning rates are decreased by factor --decay-factor')
parser.add_argument('--decay-factor', type=float, default=0.5,
                    help='ratio to cut learning rate by after --decay-cutoff reached')

parser.add_argument('--enc-lstm-widths', type=int, default=[36, 36, 36, 36],
                    nargs='+', help='EncNet widths')
parser.add_argument('--enc-dense-widths', type=int, default=[36, 36, 45],
                    nargs='+', help='TransNet widths')

parser.add_argument('--code-size', type=int, default=64,
                    help='Size of code.')
parser.add_argument('--ro-discount', help="")
parser.add_argument('--noise', type=float, help="specify standard deviation of noise.")
# parser.add_argument('--alpha', type=float, default=6e5,
#                     help='pred model learning rate alpha decay')
parser.add_argument('--offsets', type=int, default=[1, 2, 4],
                    nargs='+', help='The timestep offset values.')
parser.add_argument('--beta', type=float,
                    help='Higher beta indicates longer periods of timestep discounting.')

""""" MISCELLANEOUS """""

# parser.add_argument('--progbar', action='store_true',
#                     help='Progress bar')

