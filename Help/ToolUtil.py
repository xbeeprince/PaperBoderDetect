# encoding: utf-8

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
