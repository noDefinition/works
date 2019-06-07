from subprocess import Popen

from utils import *

if __name__ == '__main__':
    new_log_files = iu.list_children('./', ctype=iu.DIR, pattern=r'log\d', full_path=True)
    old_log_files = iu.list_children('./logs', ctype=iu.DIR, pattern=r'log\d', full_path=True)
    log_files = new_log_files + old_log_files
    if True:
        logdir = iu.choose_from(log_files)
    else:
        logdir = iu.most_recent(log_files)
    command = 'tensorboard --logdir {}/ --port {}'.format(logdir, 23456)
    print('logdir: ', logdir)
    Popen(command, shell=True).communicate()
