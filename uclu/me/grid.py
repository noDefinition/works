import uclu.me.main as m
from uclu.me import U
from uclu.me.models import *
from uclu.data.datasets import *
from utils import Nodes
from utils.tune.base_grid import BaseGrid
from utils.tune.tune_utils import LY


class GridUclu(BaseGrid):
    def __init__(self):
        super(GridUclu, self).__init__()
        self.datas, self.vers, self.gpu_ids, self.gpu_max = Nodes.select(
            nnew=(class_list, [V1], [1, 0], 2),
        )
        if self.is_debug:
            self.datas, self.gpu_ids, self.gpu_max = [DataSo], [1], 1
        self.comment = self.get_comment(default='try_try_v1')
        self.copy_modules = [m] + self.vers

    def grid_od_list(self):
        data_ly = LY({
                         U.bs: [512], U.dn: [d.name], U.cn: [d.topic_num],
                     } for d in self.datas)
        model_ly = LY()
        for v in self.vers:
            tmp_ly = LY({
                U.lid: [77], U.ep: [80], U.lr: [1e-5, 1e-6], U.vs: [v.__name__],
            })
            if v in {V1}:
                tmp_ly *= LY({
                    U.ed: [64, 300, 128, 256], U.tpad: [20], U.bpad: [100],
                })
            model_ly += tmp_ly
        ret_ly = data_ly * model_ly
        # iu.json.dump([dict(p) for p in ret_ly.pairs_list], open('fuckk.json', mode='w'), indent=4)
        return ret_ly.eval()


if __name__ == '__main__':
    GridUclu().main()
