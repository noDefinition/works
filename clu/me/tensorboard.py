from subprocess import Popen
from argparse import ArgumentParser
from utils import *

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-p', type=int, default=23456, help='port')
    args = parser.parse_args()

    new_paths = iu.list_children('./', ctype=iu.DIR, pattern=r'(log)?\d+', full_path=True)
    old_paths = iu.list_children('./logs', ctype=iu.DIR, pattern=r'(log)?\d+', full_path=True)
    log_files = (new_paths + old_paths)
    if True:
        logdir = iu.choose_from(log_files)
    else:
        logdir = iu.most_recent(log_files)
    command = 'tensorboard --logdir {}/ --port {}'.format(logdir, args.p)
    print('logdir: ', logdir)
    Popen(command, shell=True).communicate()
