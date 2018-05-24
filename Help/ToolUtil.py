# encoding: utf-8

import os
from time import strftime, localtime
from termcolor import colored


def get_local_time():
    return strftime("%d %b %Y %Hh%Mm%Ss", localtime())

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