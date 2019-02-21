import threading
import time


class Object:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def update(self, **kwargs):
        self.__dict__.update(kwargs)


class UniqueName:
    def __init__(self):
        self.nc = {}

    def qeury(self, name):
        self.nc.setdefault(name, 0)
        return self.nc[name]

    def __call__(self, name):
        self.nc.setdefault(name, 0)
        self.nc[name] += 1
        return '{}_{}'.format(name, self.nc[name])


class Timer:
    def __init__(self):
        self.end = True

    def loop(self):
        while not self.end:
            time.sleep(0.1)
            print('\r{:.1f}s'.format(time.time() - self.start_time), end='')

    def start(self):
        if not self.end: return
        self.start_time = time.time()
        self.end = False
        self.thread = threading.Thread(target=self.loop)
        self.thread.setDaemon(True)
        self.thread.start()

    def stop(self):
        if self.end: return 0.0
        self.end = True
        self.thread.join()
        print('\r                \r', end='')
        return time.time() - self.start_time


class ProgressBar:
    def __init__(self, N, msg='ProgressBar', style='None'):
        self.max_step = N
        self.max_show = 60
        self.now_step = 0
        self.now_show = 0
        self.cache_step = 0

        self.msg = '{}: ['.format(msg)
        self.cache = 0
        self.add_len = 20
        print(self.msg, end='')
        s = ' ' * (self.max_show + self.add_len)
        print('{}'.format(s), end='')
        self.start_time = time.time()
        self.pre_flush_time = self.start_time

    def get_show(self, step):
        rate = step / self.max_step
        return int(rate * self.max_show)

    def make_a_step(self):
        self.cache_step += 1
        now = time.time()
        if now - self.pre_flush_time > 0.1:
            self.flush(self.cache_step)
            self.cache_step = 0
            self.pre_flush_time = now

    def flush(self, step):
        self.now_step += step
        nxt_show = self.get_show(self.now_step)
        n = nxt_show - self.now_show
        print('\b' * self.add_len, end='')
        if n:
            if self.now_show == 0:
                print('\b' * (self.max_show - self.now_show), end='')
                print('=' * (n - 1), end='>')
            else:
                print('\b' * (self.max_show - self.now_show + 1), end='')
                print('=' * n, end='>')
            print(' ' * (self.max_show - nxt_show), end='')
        ct = time.time() - self.start_time
        t = ct / self.now_step * self.max_step if self.now_step else 0.0
        s = '] {}/{}, {:.1f}s/{:.1f}s'.format(self.now_step, self.max_step, ct, t)
        self.add_len = max(self.add_len, len(s))
        print(s, end='')
        print(' ' * (self.add_len - len(s)), end='', flush=True)
        self.now_show = nxt_show

    def stop(self):
        s = ' ' * (self.max_show + self.add_len + len(self.msg))
        print('\r{}\r'.format(s), end='')
        return time.time() - self.start_time


class Logger:
    def __init__(self):
        self.pre_t = 0
        self.fn = ''

    def set_fn(self, home, fn):
        self.unique_fn = '{} {}'.format(self.date(), fn)
        self.fn = '{}/{}.log'.format(home, self.unique_fn)
        print('log file: {}'.format(self.fn))

    @staticmethod
    def red_str(s):
        return '\033[1;31;40m{}\033[0m'.format(s)

    @staticmethod
    def date(t=None):
        if t is None:
            t = time.time()
        return time.strftime("%Y-%m-%d %H.%M.%S", time.localtime(t))

    def _log(self, s, to_file=True, red=False):
        if to_file and self.fn:
            try:
                open(self.fn, 'a').write(s + '\n')
            except Exception as e:
                print(e)
        if red:
            s = self.red_str(s)
        print(s)

    def log(self, s, i=60, **kw):
        now = time.time()
        s = str(s)
        if i >= 0 and now - self.pre_t > i:
            self._log(self.date(now))
            self.pre_t = now
        self._log(s, **kw)


def main():
    print('hello world, log')
    log = Logger()
    log.log(10)


if __name__ == '__main__':
    main()
