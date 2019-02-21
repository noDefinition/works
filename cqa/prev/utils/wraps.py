import numpy as np, sys, math, os
import time, threading
import atexit, shutil
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

