from argparse import ArgumentParser
from subprocess import Popen, DEVNULL as V

import utils.multiprocess_utils as mu
import utils.logger_utils as lu

from clu.data.datasets import *
from utils.node_utils import Nodes, entries2name

d_classes, gpus, batch_size = Nodes.select(
    n1702=([Data20ng], [0, 1], 2),
    ngpu=([Data20ng], [0, 3], 2),
)
gpu_frac = {1: 0.8, 2: 0.4, 3: 0.27, 4: 0.18}[int(batch_size / len(gpus))]
print('Use data: {}, GPU: {}, batch size: {}'.format([d.name for d in d_classes], gpus, batch_size))


def get_all_i_need():
    parser = ArgumentParser()
    parser.add_argument('-do', type=float, help='model: dropout')
    parser.add_argument('-ed', type=int, help='model: embed dim')
    parser.add_argument('-bs', type=int, help='iter: pos sample batch size')
    parser.add_argument('-ws', type=int, help='model: window size')
    parser.add_argument('-id', type=int, default=0, help='identification')
    parser.add_argument('-cn', help='cluster num')
    parser.add_argument('-dn', help='data class name')
    parser.add_argument('-gp', type=float, help='gpu fraction')
    args = parser.parse_args()
    print('Using class', args.dn)
    d_class = name2d_object[args.dn]
    args.cn = d_class.clu_num
    # params_as_whole = ','.join(['{}={}'.format(k, v) for k, v in args.__dict__.items() if v is not None])
    params_as_whole = entries2name(args.__dict__)
    logger = lu.get_logger('./logging/{}.txt'.format(params_as_whole))
    return args, logger, d_class


def run_short_attention(command, cwd='./'):
    Popen(command, cwd=cwd, shell=True, bufsize=1, stdin=V, stdout=V, stderr=V).communicate()
    # Popen(command, cwd=cwd, shell=True, bufsize=1).communicate()


def run():
    cmd = 'CUDA_VISIBLE_DEVICES={} python3.6 short_cnn.py '
    # name_value_list = (
    #     ('-do', [0.7]),
    #     ('-ed', [256]),
    #     ('-bs', [100]),
    #     ('-ws', [5]),
    #     ('-id', [i for i in range(10)]),
    # )   # for Kaggle
    # name_value_list = (
    #     ('-do', [0.7]),
    #     ('-ed', [256]),
    #     ('-bs', [100]),
    #     ('-ws', [3]),
    #     ('-id', [i for i in range(12)]),
    # )  # for Event
    # name_value_list = (
    #     ('-do', [0.5]),
    #     ('-ed', [256]),
    #     ('-bs', [100]),
    #     ('-ws', [5]),
    #     ('-id', [i for i in range(12)]),
    # )  # for Google
    nv_list = (
        ('-do', [0.3, 0.5, 0.7]),
        ('-ed', [200, 300]),
        ('-bs', [50, 100]),
        ('-ws', [3, 5, 7]),
        ('-id', [i for i in range(2)]),
        ('-dn', [d.name for d in d_classes]),
        ('-gp', [gpu_frac]),
    )  # for DataSnippets
    grid = au.grid_params(nv_list)
    args_list = [[cmd.format(gpus[idx % len(gpus)]) + entries2name(g, inter=' ', intra=' ', postfix='')]
                 for idx, g in enumerate(grid)][6:]
    print(args_list[:5])
    mu.multi_process_batch(run_short_attention, batch_size=batch_size, args_list=args_list)


if __name__ == '__main__':
    # fi.mkdir('./logging/', remove_previous=True)
    run()
