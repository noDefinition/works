from argparse import ArgumentParser
from subprocess import Popen

from utils import *

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-l', action='store_true', default=False, help='if find in logs')
    args = parser.parse_args()

    path = './logs' if args.l else './'
    log_files = iu.list_children(path, ctype=iu.DIR, pattern='^log\d', full_path=True)
    # log_files = iu.list_children('./', ctype=iu.DIR, pattern='^log\d', full_path=True)
    logdir = iu.most_recent(log_files)
    print('logdir: ', logdir)
    command = 'tensorboard --logdir {} --port {}'.format(logdir, 23456)
    Popen(command, shell=True).communicate()
