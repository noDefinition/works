import multiprocessing as mp
from subprocess import DEVNULL as V, Popen
from typing import List, Tuple, Dict, Callable, Generator
from utils import au
from utils.tune.arg_keys import X
# import types


class LY:
    def __init__(self, *pairs_list):
        if len(pairs_list) == 1 and isinstance(pairs_list[0], Generator):
            target = pairs_list[0]
        else:
            target = pairs_list
        self.pairs_list: List[List[Tuple]] = list(list(pairs) for pairs in target)

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


def auto_gpu(func: Callable, args_list: List[Tuple], device2max: Dict, callback=None):
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
        k = dict(device_id=d)
        q = mp.Queue()
        p = mp.Process(target=subp, args=(func, a, k, q), daemon=True)
        pool.append((d, p, q))
        p.start()

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


def subp(f, a, k, q):
    q.put(f(*a, **k))


def run_on_end(args):
    device_id, gid = args
    print('tu: run_on_end, gid', gid)


def run_on_gpu(cmd_pre, od, max2frac, device_max, device_id):
    od.update({X.gi: device_id, X.gp: max2frac[device_max[device_id]]})
    entries = [(k if k.startswith('-') else '-' + k, v) for k, v in od.items()]
    cmd_full = cmd_pre + au.entries2name(entries, inter=' ', inner=' ')
    v = None if (sum(device_max.values()) == 1) else V
    Popen(cmd_full, cwd='./', shell=True, stdin=v, stdout=v, stderr=None).communicate()
    return device_id, od[X.gid]


def run_od_list(cmd_pre, od_list, dev_ids, dev_max, subfunc=run_on_gpu, callback=run_on_end):
    dev2max = get_dev2max(dev_ids, dev_max)
    max2frac = get_max2frac()
    args_list = [(cmd_pre, od, max2frac, dev2max) for od in od_list]
    auto_gpu(subfunc, args_list, dev2max, callback)


def get_dev2max(dev_ids, dev_max):
    if isinstance(dev_max, int):
        dev2max = {dev_id: dev_max for dev_id in dev_ids}
    elif isinstance(dev_max, list):
        assert len(dev_max) == len(dev_ids)
        dev2max = dict(zip(dev_ids, dev_max))
    else:
        raise ValueError('dev_max invalid: {}'.format(dev_max))
    return dev2max


def get_max2frac():
    max2frac = {1: 0.87, 2: 0.43, 3: 0.29, 4: 0.22}
    max2frac[1] = 0.5
    max2frac[2] = 0.3
    return max2frac


def update_od_list(od_list, log_path, shuffle):
    for i, od in enumerate(od_list):
        od[X.gid] = i
        od[X.lg] = log_path
    if shuffle:
        od_list = au.shuffle(od_list)
    for i, od in enumerate(od_list):
        print(au.entries2name(od, inner='=', inter=' ')) if i <= 10 else None
    return od_list
