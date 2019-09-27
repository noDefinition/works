import uclu.me.main as m
from uclu.data.datasets import DataSo, class_list
from uclu.me import U
from uclu.me.models import *
from utils import Nodes
from utils.tune.base_grid import BaseGrid
from utils.tune.tune_utils import LY


class Grid(BaseGrid):
    def __init__(self):
        super(Grid, self).__init__()
        self.datas, self.vers, self.gpu_ids, self.gpu_max = Nodes.select(
            nnew=(class_list, [V3], [1, 0], 1),
            # n1702=(class_list, [UAtt], [0, ], 2),
        )
        if self.is_debug:
            self.datas, self.gpu_ids, self.gpu_max = [DataSo], [1], 1
        self.comment = self.get_comment(default='get_losses')
        self.copy_modules = [m] + self.vers

    def grid_od_list(self):
        data_ly = LY({
            U.bs: [256],
            U.dn: [d.name],
            U.cn: [d.topic_num],
        } for d in self.datas)
        model_ly = LY()
        for v in self.vers:
            tmp_ly = LY({
                U.lid: [5],
                U.ep: [80],
                U.lr: [1e-3],
                U.vs: [v.__name__],
            })
            if v in {V2}:
                tmp_ly *= LY({
                    U.ed: [64, 128],
                    U.tpad: [30],
                    U.bpad: [200],
                    U.addb: [0, 1],
                    U.addu: [0, 1],
                    U.pair: [0],
                })
            if v in {UAtt}:
                tmp_ly *= LY({
                    U.ed: [64, 128],
                    U.tpad: [30],
                    U.bpad: [200],
                    U.addb: [0.1, 0.5, 0.7, 1],
                    U.addu: [0.1, 0.5, 0.7, 1],
                    U.addpnt: [0, 0.1, 0.5, 0.7, 1],
                })
            if v in {V3}:
                tmp_ly *= LY({
                    U.ed: [64],
                    U.tpad: [30],
                    U.bpad: [200],
                }) * LY({
                    U.addb: [1],
                    U.addu: [1],
                    U.addpnt: [1],
                    # U.addpnt: [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5,
                    #            10, 20, 50, 100, 200, 500, 0, ],
                    # }, {
                    #     U.addb: [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 0, ],
                    #     U.addu: [1],
                    #     U.addpnt: [1],
                    # }, {
                    #     U.addb: [1],
                    #     U.addu: [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 0, ],
                    #     U.addpnt: [1],
                })
            model_ly += tmp_ly
        ret_ly = data_ly * model_ly
        # from utils import iu
        # iu.dump_array('./fuck.json', ret_ly.eval())
        # exit()
        return ret_ly.eval()


if __name__ == '__main__':
    Grid().main()
