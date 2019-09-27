from uclu.data.datasets import *
from uclu.bert import UB
import uclu.bert.main as m

from utils import Nodes
from utils.tune.base_grid import BaseGrid
from utils.tune.tune_utils import LY


class GridUcluBert(BaseGrid):
    def __init__(self):
        super(GridUcluBert, self).__init__()
        self.datas, self.gpu_ids, self.gpu_max = Nodes.select(
            nnew=(class_list, [0], 1),
        )
        if self.is_debug:
            self.datas, self.gpu_ids, self.gpu_max = [DataSo], [1], 1
        self.comment = self.get_comment(default='try_bert_1')
        self.copy_modules = [m] + self.vers

    def grid_od_list(self):
        ret_ly = LY({
            UB.lid: [3],
            UB.dn: [d.name],
            UB.cn: [d.topic_num],
            UB.ep: [10],
            UB.bs: [3],
            UB.mxl: [129],
            UB.lr: [1e-5],

            UB.ly: [2],
            UB.dh: [32],
            UB.nh: [4],
        } for d in self.datas)
        return ret_ly.eval()


if __name__ == '__main__':
    GridUcluBert().main()
