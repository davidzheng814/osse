import argparse
from os.path import join

# DATAROOT = '/om/user/dzd123/'
DATAROOT = '/data/vision/oliva/scenedataset/urops/scenelayout/'
ROOT = '../..'
parser = argparse.ArgumentParser(description='Physics mass inference model.')

""""" DATA LOADING """""

parser.add_argument('--data-file', type=str, default=join(DATAROOT, 'physics_b.h5'),
                    help='path to training data')
parser.add_argument('--num-points', type=int, default=0,
                    help='max points to use. (0 for no max)')
parser.add_argument('--test-points', type=int, default=5000,
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
parser.add_argument('--pred-only', action='store_true',
                    help='Whether to use PredNet only.')
parser.add_argument('--skip-train', action='store_true',
                    help='Whether to skip training')
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
parser.add_argument('--ref-enc-zero', action='store_true',
                    help='Zero out reference object encoding')
parser.add_argument('--no-ref-enc-sub', action='store_true',
                    help='Do not subtract reference object encoding')

parser.add_argument('--epochs', type=int, default=500,
                    help='number of epochs')
parser.add_argument('--batch-size', type=int, default=1024,
                    help='batch size')
parser.add_argument('--lr', type=float, default=2e-3,
                    help='learning rate')
parser.add_argument('--enc-reg-factor', type=float, default=0.,
                    help='L2 Regularization on effects if using inet_enc_model')
parser.add_argument('--reg-factor', type=float, default=0.,
                    help='L2 Regularization on effects')
parser.add_argument('--decay-cutoff', type=int, default=-1,
                    help='if loss does not improve for this number of epochs, both'
                         'learning rates are decreased by factor --decay-factor')
parser.add_argument('--decay-factor', type=float, default=0.8,
                    help='ratio to cut learning rate by after --decay-cutoff reached')
parser.add_argument('--noise', type=float, default=0, help="specify standard deviation of noise (as ratio of standard deviation of state elements).")
parser.add_argument('--grad-clip', type=float, default=5.,
                    help='maximum norm of gradients for clipping. set to 0 for no clip')

parser.add_argument('--enc-lstm-widths', type=int, default=[36, 36, 36, 36],
                    nargs='+', help='EncNet widths')
parser.add_argument('--enc-dense-widths', type=int, default=[12, 12, 15],
                    nargs='+', help='TransNet widths')
parser.add_argument('--inet-pred-frames', type=int, default=2,
                    help='Number of pred frames to combine for inet_enc_model')

parser.add_argument('--re-widths', type=int, default=[150, 150, 150, 150],
                    nargs='+', help='RelationNet widths')
parser.add_argument('--sd-widths', type=int, default=[100, 100],
                    nargs='+', help='SelfDynamicsNet widths')
parser.add_argument('--effect-width', type=int, default=50,
                    help='Width of effect (output of relation net and sd net)')
parser.add_argument('--agg-widths', type=int, default=[100, 50],
                    nargs='+', help='AggregationNet widths')

