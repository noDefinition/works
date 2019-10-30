import uclu.bert.main as m
from uclu.bert import UB
from uclu.bert.models import *
from uclu.data.datasets import *
from utils import Nodes
from utils.tune.base_grid import BaseGrid
from utils.tune.tune_utils import LY


class GridUcluBert(BaseGrid):
    def __init__(self):
        super(GridUcluBert, self).__init__()
        vers = [B1]
        self.datas, self.vers, self.gpu_ids, self.gpu_max = Nodes.select(
            nnew=(class_list, vers, [1], 2),
            n1702=(class_list, vers, [1], 2),
        )
        if self.is_debug:
            self.datas, self.gpu_ids, self.gpu_max = [DataSo], [1], 1
        self.is_pretrain = 0
        self.main_file = ['./main.py', './main_pre.py'][self.is_pretrain]
        self.comment = self.get_comment(default='try_b1')

    def grid_od_list(self):
        ret_ly = LY({
            UB.vs: [v.__name__ for v in self.vers],
            UB.lid: [3],
            UB.dn: [d.name],
            UB.cn: [d.topic_num],
        } for d in self.datas)
        if self.is_pretrain:
            ret_ly *= LY({
                UB.lr: [1e-5],
                UB.ep: [1], UB.bs: [2], UB.mxl: [0],
                UB.ly: [2], UB.dh: [64], UB.nh: [8],
                UB.do: [0.7, 0.3, 0.5, 0.1],
            })
        else:
            ret_ly *= LY({
                UB.lr: [1e-3],
                UB.ep: [20],
                UB.bs: [64],
                UB.tpad: [20],
                UB.bpad: [80],
            })
        return ret_ly.eval()


if __name__ == '__main__':
    GridUcluBert().main()
