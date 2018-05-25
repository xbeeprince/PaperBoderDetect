# encoding: utf-8

import os
import wget
from pyunpack import Archive
import tensorflow as  tf
from time import strftime, localtime
from termcolor import colored


def get_local_time():
    return strftime("%Y-%m-%d %H:%M:%S", localtime())

def print_info(info_string):
    info = '[{0}][INFO] {1}'.format(get_local_time(), info_string)
    print colored(info, 'green')


def print_warning(warning_string):
    warning = '[{0}][WARNING] {1}'.format(get_local_time(), warning_string)

    print colored(warning, 'blue')


def print_error(error_string):
    error = '[{0}][ERROR] {1}'.format(get_local_time(), error_string)

    print colored(error, 'red')

def projectRootDir():
    help_dir = os.path.abspath(os.path.dirname(__file__))
    root_dir = os.path.abspath(os.path.dirname(help_dir))
    return root_dir

def getDirWithPath(dirPath):
    root_dir = projectRootDir()
    tmpDirPath = os.path.join(root_dir, dirPath)
    return tmpDirPath

def getConfigDir():
    tmpConfigDir = getDirWithPath("Config")
    return  tmpConfigDir

def getDataDir():
    tmpDataDir = getDirWithPath("hed-data")
    return tmpDataDir

def getModelDir():
    tmpModelDir = getDirWithPath("Detect")
    return tmpModelDir

def read_file_list(filelist):
    pfile = open(filelist)
    filenames = pfile.readlines()
    pfile.close()

    filenames = [f.strip() for f in filenames]

    return filenames

def get_session(gpu_fraction):

    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = 5 #int(os.environ.get('OMP_NUM_THREADS'))
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def download_data(filepath, outputdir):

    _, rar_file = os.path.split(filepath)
    rar_file = os.path.join(outputdir, rar_file)

    if not os.path.exists(rar_file):
        print_info('Downloading {} to {}'.format(filepath, rar_file))
        _ = wget.download(filepath, out=outputdir)

    print_info('Decompressing {} to {}'.format(rar_file, outputdir))
    Archive(rar_file).extractall(outputdir)

def getFileBaseName(file):
    return os.path.basename(file)

def getFileBaseNameWithFormat(file):
    baseName = getFileBaseName(file)
    return os.path.splitext(baseName)[0]