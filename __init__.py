# CycleGAN modified from https://github.com/dragen1860/TensorFlow-2.x-Tutorials/tree/master/15-CycleGAN
# Tensorflow 2.0 multi-GPU: https://www.tensorflow.org/guide/distributed_training#using_tfdistributestrategy_with_custom_training_loops
# Tensorflow virtual GPUs: https://www.tensorflow.org/guide/gpu

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import tensorflow as tf
import numpy as np
import argparse
import os
from tfw import Training_framework

assert tf.__version__.startswith('2.2'), 'Please use TensorFlow version 2.2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.debugging.set_log_device_placement(True)

tf.random.set_seed(27)
np.random.seed(27)

#Parse input arguments
parser = argparse.ArgumentParser(description='CycleGAN Multi-GPU synchronous training Example')
parser.add_argument('--a-dir', default='./images/trainA',
                    help='style A training dir')
parser.add_argument('--b-dir', default='./images/trainB',
                    help='style B training dir')
parser.add_argument('--beta-1', type=float, default=0.5,
                    help='beta 1 parameter for CycleGAN')
parser.add_argument('--lr', type=float, default=0.0002,
                    help='starting learning rate for CycleGAN')
parser.add_argument('--batch', type=int, default=1,
                    help='batch size per device')
parser.add_argument('--epochs', type=int, default=100,
                    help='total epochs')
parser.add_argument('--lsgan', type=bool, default=True,
                    help='lsgan')
parser.add_argument('--cyc-lambda', type=int, default=10,
                    help='cyc_lambda')
parser.add_argument('--vcpu', type=bool, default=True,
                    help='use virtual CPU if no GPU found')
parser.add_argument('--n-vcpu', type=int, default=2,
                    help='number of virtual CPUs')
parser.add_argument('--load', type=bool, default=True,
                    help='load model if exist')
parser.add_argument('--load-dir', default='./model',
                    help='model dir')

args = parser.parse_args()

def main(args):
    fw = Training_framework(args.a_dir,
                            args.b_dir,
                            args.beta_1,
                            args.lr,
                            args.batch,
                            args.epochs,
                            args.lsgan,
                            args.cyc_lambda,
                            args.vcpu,
                            args.n_vcpu,
                            args.load,
                            args.load_dir)
    fw.train()

if __name__ == '__main__':
    main(args)