# encoding: utf-8
import os
import sys
import argparse
import tensorflow as tf
from Detect.DetectModel import DetectModel
from Detect.DetectModelTrain import DetectModelTrain
from Detect.DetectModelTest import DetectModelTest

def get_session(gpu_fraction):

    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = int(os.environ.get('OMP_NUM_THREADS'))
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def main(args):

    if not (args.run_train or args.run_test):
        print 'Set atleast one of the options --train | --test '
        parser.print_help()
        return

    if args.run_test or args.run_train:
        session = get_session(args.gpu_limit)

    if args.run_train:

        trainer = DetectModelTrain()
        trainer.setup()
        trainer.run(session)

    if args.run_test:

        tester = DetectModelTest()
        tester.setup(session)
        tester.run(session)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Utility for Training/Testing DL models(Concepts/Captions) using tensorflow')
    parser.add_argument('--train', dest='run_train', action='store_true', default=False, help='Launch training')
    parser.add_argument('--test', dest='run_test', action='store_true', default=False, help='Launch testing on a list of images')

    args = parser.parse_args()

    main(args)