import multiprocessing as mp
import utils.array_utils as au


def split_multi(array, process_num):
    from math import ceil
    batch_size = int(ceil(len(array) / process_num))
    return list(au.split_slices(array, batch_size))


def multi_process(func, args_list=None, kwargs_list=None, while_wait=None):
    res_list = list()
    process_num = len(args_list) if args_list is not None else len(kwargs_list)
    pool = mp.Pool(processes=process_num)
    for i in range(process_num):
        args = args_list[i] if args_list else ()
        kwargs = kwargs_list[i] if kwargs_list else {}
        res_list.append(pool.apply_async(func=func, args=args, kwds=kwargs))
    pool.close()
    if while_wait is not None:
        while_wait()
    pool.join()
    results = [res.get() for res in res_list]
    return results


def multi_process_batch(func, batch_size, args_list, kwargs_list=None):
    n_args = len(args_list)
    if kwargs_list is None:
        kwargs_list = [{}] * n_args
    results = list()
    for s, u in list(au.split_since_until(n_args, batch_size)):
        print('going to process {} / {}'.format(u, n_args))
        results.extend(multi_process(func, args_list[s: u], kwargs_list[s: u]))
    return results
