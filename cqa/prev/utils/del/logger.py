#########################################################################
# File Name: log.py
# Author: wwang
# mail: 750636248@qq.com
# Created Time: 2017年07月27日 星期四 18时23分31秒
#########################################################################

import numpy as np, sys, math, os
import time, threading
import atexit, shutil
import util
import functools

indent = 2
func_deep = 0

#wraps
def run_time(func):
    functools.wraps(func)
    def __wrap(*args, **kw):
        global func_deep
        print('{}> call {}()...'.format(' ' * func_deep * indent, func.__name__))
        bt = time.time()
        func_deep += 1
        ret = func(*args, **kw)
        func_deep -= 1
        ct = time.time() - bt
        print('{}< call {}() over, time: {:.2f}'.format(' ' * func_deep * indent, func.__name__, ct))
        return ret
    return __wrap

def run_for(n = None, step = 0):
    def __d(func):
        def __w(*args):
            d = args[-1]
            if step == 0:
                return [func(*args[:-1], _d) for _d in d]
            bt = time.time()
            _n = len(d) if n is None else n
            def _f(i, _d):
                if i % step == 0:
                    nt = time.time() - bt
                    rt = (_n - i) * nt / i if i else 0
                    print('\r{} #{}/{}, cost {:.2f}s, rest: {:.2f}s({:.1f}min),  '.format(func.__name__, i, _n, nt, rt, rt / 60), end = '', flush = True)
                return func(*args[:-1], _d)
            ret = [_f(i, _d) for i, _d in enumerate(d)]
            print('over ~')
            return ret
        return __w
    return __d

def red_str(s):
    return '\033[1;31;40m{}\033[0m'.format(s)

def date(t = None):
    if t is None:
        t = time.time()
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))

pre_t = 0
bad_log = True
test_running = False
log_home = util.config['Dirs']['loghome']
test_log_dir = util.config['Dirs']['testlog']
bad_log_dir = util.config['Dirs']['badlog']
fn = ''

def log_start(_fn):
    global fn
    fn = '{}/{}_{}.log'.format(log_home, date(), _fn)
    print('log file: {}'.format(fn))
    atexit.register(log_over)

def _log(s, to_file = True, red = False, **kw):
    if to_file:
        try:
            open(fn, 'a').write(s + '\n')
        except Exception as e:
            print(e)
    if red:
        s = red_str(s)
    print(s, **kw)

def log(s, i = 30, **kw):
    now = time.time()
    s = str(s)
    global pre_t
    if i >= 0 and now - pre_t > i:
        _log(date(now))
        pre_t = now
    _log(s, **kw)

class Timer:
    def __init__(self):
        self.end = True

    def loop(self):
        while not self.end:
            time.sleep(0.1)
            print('\r{:.1f}s'.format(time.time() - self.start_time), end = '')

    def start(self):
        if not self.end: return
        self.start_time = time.time()
        self.end = False
        self.thread = threading.Thread(target = self.loop)
        self.thread.setDaemon(True)
        self.thread.start()

    def stop(self):
        if self.end: return 0.0
        self.end = True
        self.thread.join()
        print('\r                \r', end = '')
        return time.time() - self.start_time
_timer = Timer()

def log_over():
    global fn
    log('log file: {}\nnow: {}'.format(fn, date()))
    print('bad log: {}'.format(str(bad_log)))
    print('test running: {}'.format(str(test_running)))
    if bad_log:
        print(red_str('Bad log!'))
        if os.path.isfile(fn):
            shutil.move(fn, bad_log_dir)
            print(red_str('moved to bad_log/'))
    if test_running:
        print(red_str('test running log!'))
        if os.path.isfile(fn):
            shutil.move(fn, test_log_dir)
            print(red_str('moved to test_log/'))
    print(red_str('over~******************************************************************************************************\n\n'))
    fn = ''

def main():
    print('hello world, log')
    log(10)
    _timer.start()
    time.sleep(2)
    _timer.stop()
    log(10)

if __name__ == '__main__':
    main()

