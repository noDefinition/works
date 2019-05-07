from cqa.data.datasets import DataSo, DataZh
from utils import Nodes, tu, iu, tmu
from cqa.mee.models import *
from cqa.mee import K


class GridMee:
    def __init__(self):
        self.datas, self.vers, self.gpu_ids, self.gpu_max = Nodes.select(
            n1702=([DataZh, DataSo], [AAAI17], [0, ], 2),
            ngpu=([DataSo, DataZh], [IJCAI15], [0, 2], 1),
        )
        # self.use_fda = 0 if len(self.gpu_ids) * self.gpu_max <= 1 else 1
        self.use_fda = 1
        self.comment = 'have_a_try'

    def grid_od_list(self):
        LY = tu.LY
        common_ly = LY((
            (K.ep, [20]), (K.es, [6]), (K.lr, [1e-4]),
            (K.dn, [d.name for d in self.datas]), (K.fda, [self.use_fda]),
            (K.lid, list(range(5))),
        ))
        vers_ly = LY()
        for v in self.vers:
            tmp_ly = LY(((K.vs, [v.__name__]),))
            if v == V1 or v == V5 or v == P1 or v == P2:
                tmp_ly *= LY((
                    (K.reg, [0, 1, 1e-2]),
                ))
            if v == U1:
                tmp_ly *= LY((
                    # (K.reg, [0, 1]),
                    # )) * LY((
                    (K.woru, [0]),
                ), (
                    (K.woru, [1, 2]),
                    (K.eps, [100, 1000, 10000]),
                ))
            if v == W1:
                tmp_ly *= LY((
                    # (K.reg, [1e-2]),
                    (K.reg, [0, 1, 1e-2]),
                    (K.eps, [0, 1e-1, 1e-2, 1e-3]),
                ))
            if v == S1:
                tmp_ly *= LY((
                    (K.mix, [0, 0.5, 1, 2]),
                ))
            if v == S2:
                tmp_ly *= LY((
                    (K.mix, [0, 0.5, 2]),
                    (K.dpt, [1, 3]),
                    (K.act, [0]),
                    # (K.act, [0, 1, 2]),
                ))
            if v == P3:
                tmp_ly *= LY((
                    (K.dpt, [4]),
                    (K.act, [0, 1, 2]),
                ))
            # if v == AAAI15:
            #     tmp_ly *= LY((
            #         (K.act, [0, 1, 2]),
            #     ))
            vers_ly += tmp_ly
        return (common_ly * vers_ly).eval()

    @staticmethod
    def get_args():
        parser = tu.get_common_parser()
        _ = parser.add_argument
        _(K.fda, type=int, help='if use full data, 0=simple, 1=full', required=True)
        _(K.es, type=int, help='runtime: step before early stop')
        _(K.lr, type=float, help='train: learning rate')

        _(K.reg, type=float, default=0, help='coeff of regularization')
        _(K.drp, type=float, default=1, help='probability of dropout')
        _(K.temp, type=float, default=0, help='temperature in kl/softmax')

        _(K.woru, type=int, help='adv noise for word/user, 0=no adv')
        _(K.eps, type=float, help='epsilon for adv gradient')
        _(K.atp, type=int, help='adv type')

        _(K.mix, type=float, help='mixture weight of extra modules')
        _(K.dpt, type=int, help='depth of dense')
        _(K.act, type=int, help='activate function of dense')
        return tu.parse_args(parser)

    def make_log_path(self, make_new):
        dnames, vnames = [d.name for d in self.datas], [v.__name__ for v in self.vers]
        tstr, dstr, vstr = tmu.format_date()[4:], '+'.join(dnames), '+'.join(vnames)
        log_path = './log' + '_'.join([tstr, vstr, dstr, self.comment])
        if make_new:
            import inspect
            import cqa.mee.main as m
            iu.mkdir(log_path, rm_prev=True)
            iu.copy(__file__, log_path)
            iu.copy(inspect.getfile(m), log_path)
            for v in self.vers:
                iu.copy(inspect.getfile(v), log_path)
        return log_path

    def main(self):
        log_path = self.make_log_path(make_new=True)
        od_list = self.grid_od_list()
        od_list = tu.update_od_list(od_list, log_path, shuffle=True)
        tu.run_od_list('python3.6 ./main.py ', od_list, self.gpu_ids, self.gpu_max)
        new_name = log_path + '_OVER'
        iu.rename(log_path, new_name)
        iu.move(new_name, './logs')

    @staticmethod
    def test_od():
        log_path = './jupyter_wtffff'
        iu.mkdir(log_path, rm_prev=True)
        return dict([
            (K.gid, 0), (K.gi, 0), (K.gp, 0.2), (K.ep, 20), (K.es, 5), (K.lr, 5e-3), (K.fda, True),
            (K.vs, P1.__name__), (K.reg, 0), (K.dn, DataZh.name), (K.lg, log_path),
        ])


if __name__ == '__main__':
    GridMee().main()
