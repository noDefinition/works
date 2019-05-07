import multiprocessing as mp
from argparse import ArgumentParser
from subprocess import DEVNULL as V, Popen

from utils import au, iu, tmu
from utils.tune.arg_keys import X


class LY:
    def __init__(self, *pairs_list):
        import types
        if len(pairs_list) == 1 and isinstance(pairs_list[0], types.GeneratorType):
            target = pairs_list[0]
        else:
            target = pairs_list
        self.pairs_list = list(list(pairs) for pairs in target)

    def __add__(self, other):
        """ 同级扩展 """
        return LY(*(self.pairs_list + other.pairs_list))

    def __mul__(self, other):
        """ 前后级扩展 """
        if len(self.pairs_list) == 0 or len(other.pairs_list) == 0:
            raise ValueError('empty layer is not allowed')
        return LY(a + b for a in self.pairs_list for b in other.pairs_list)

    def eval(self):
        return au.merge(au.grid_params(pairs) for pairs in self.pairs_list)


def auto_gpu(func, args_list, device2max, callback=None):
    def on_process_end(d, p, q):
        p.terminate()
        device2remain[d] += 1
        if callable(callback):
            callback(q.get())

    def recall_devices():
        while len(pool) >= max_pool_size:
            for j in range(len(pool) - 1, -1, -1):
                d, p, q = pool[j]
                if not p.is_alive():
                    on_process_end(d, p, q)
                    pool.pop(j)
                    pbar.update(1)
            time.sleep(1)

    def allocate_device():
        for d in device2remain.keys():
            if device2remain[d] > 0:
                device2remain[d] -= 1
                return d
        raise ValueError('no device can be allocated')

    def allocate_run(a):
        d = allocate_device()
        a = [d] + list(a)
        q = mp.Queue()
        p = mp.Process(target=_sub, args=(func, a, q), daemon=True)
        p.start()
        pool.append((d, p, q))

    import time
    from tqdm import tqdm
    device2remain = dict(device2max)
    max_pool_size = sum(device2remain.values())
    pool = list()
    # timer = tmu.Timer(len(args_list))
    # timer.progress(can_print=(idx >= max_pool_size - 1))
    pbar = tqdm(total=len(args_list), ncols=50, leave=True, desc='args')
    for idx, args in enumerate(args_list):
        recall_devices()
        allocate_run(args)
    for _d, _p, _q in pool:
        _p.join()
        on_process_end(_d, _p, _q)
        pbar.update(1)
    pbar.close()


def _sub(f, a, q):
    q.put(f(*a))


def run_on_gpu(device_id, od, device2max, max2frac, cmd_pre):
    od.update({X.gi: device_id, X.gp: max2frac[device2max[device_id]]})
    entries = [(k if k.startswith('-') else '-' + k, v) for k, v in od.items()]
    command = cmd_pre + au.entries2name(entries, inter=' ', inner=' ')
    v = None if (sum(device2max.values()) == 1) else V
    Popen(command, cwd='./', shell=True, stdin=v, stdout=v, stderr=None).communicate()
    return device_id, od[X.gid]


def run_on_end(args):
    device_id, gid = args
    print('tu: run_on_end, gid', gid)


def run_od_list(cmd_pre, od_list, dev_ids, dev_max, func=run_on_gpu, callback=run_on_end):
    if isinstance(dev_max, int):
        device2max = {dev_id: dev_max for dev_id in dev_ids}
    elif isinstance(dev_max, list):
        assert len(dev_max) == len(dev_ids)
        device2max = dict(zip(dev_ids, dev_max))
    else:
        raise ValueError('dev_max invalid: {}'.format(dev_max))
    max2frac = {i: round(1 / (1.15 * i), 2) for i in range(1, 5)}
    args_list = [(od, device2max, max2frac, cmd_pre) for od in od_list]
    auto_gpu(func, args_list, device2max, callback)


def get_log_path(str_list, make_new: bool):
    if make_new:
        log_path = './log_{}_{}'.format(tmu.format_date()[2:], '+'.join(str_list))
        iu.mkdir(log_path, rm_prev=True)
    else:
        log_path = iu.choose_from(iu.list_children('./', iu.DIR, 'log', full_path=True))
    print('log path:', log_path)
    return log_path


def update_od_list(od_list, log_path, shuffle):
    for i, od in enumerate(od_list):
        od[X.gid] = i
        od[X.lg] = log_path
    if shuffle:
        od_list = au.shuffle(od_list)
    for i, od in enumerate(od_list):
        print(au.entries2name(od, inner='=', inter=' ')) if i <= 10 else None
    return od_list


def get_common_parser() -> ArgumentParser:
    def _(name, **kwargs):
        prev('-' + name, **kwargs)

    parser = ArgumentParser()
    prev = parser.add_argument
    parser.add_argument = _
    # return parser
    # parser = get_augmented_parser()
    # _ = parser.add_argument
    _(X.gid, type=int, help='global id', required=True)
    _(X.lid, type=int, help='local id')
    _(X.gi, type=int, help='gpu id number')
    _(X.gp, type=float, help='gpu fraction')

    _(X.lg, type=str, help='logging path')
    _(X.dn, type=str, help='data name')
    _(X.vs, type=str, help='model version')
    _(X.ep, type=int, help='epoch num', required=True)
    return parser


def parse_args(parser: ArgumentParser) -> dict:
    return dict((k, v) for k, v in parser.parse_args().__dict__.items())


if __name__ == '__main__':
    print(parse_args(get_common_parser()))
