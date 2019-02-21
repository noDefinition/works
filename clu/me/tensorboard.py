from subprocess import Popen

from utils import *


# if False:
#     logdir = iu.choose_from('./', children_type=iu.TYPE_DIR, pattern='log_')
# else:
#     logdir = iu.most_recent_modified('./', children_type=iu.TYPE_DIR, pattern='log_')
#
# print('select log directory', logdir)
# command = 'tensorboard --logdir {}/ --port {}'.format(logdir, 54321)
# Popen(command, shell=True)


# def event():
#     import time
#     from queue import Queue
#     from threading import Thread
#     from subprocess import Popen
#
#     quit_ = 'q'
#
#     def read_key_input():
#         while True:
#             value = input()
#             q_exec.put(value)
#
#     q_exec = Queue()
#     Thread(target=read_key_input, daemon=True).start()
#
#     while True:
#         if False:
#             logdir = iu.choose_from('./', children_type=iu.DIR, pattern='log_')
#         else:
#             logdir = iu.most_recent('./', children_type=iu.DIR, pattern='log_')
#         command = 'tensorboard --logdir {}/ --port {}'.format(logdir, 54321)
#         print('logdir: ', logdir)
#         p = Popen(command, shell=True)
#         msg = str(q_exec.get()).strip()
#         if msg == quit_:
#             exit()
#         p.terminate()
#         time.sleep(3)
#         print('sub process terminated')


if __name__ == '__main__':
    log_files = iu.list_children('./', ctype=iu.DIR, pattern='log_', full_path=True)
    if False:
        logdir = iu.choose_from(log_files, )
    else:
        logdir = iu.most_recent(log_files)
    command = 'tensorboard --logdir {}/ --port {}'.format(logdir, 6006)
    print('logdir: ', logdir)
    Popen(command, shell=True).communicate()
