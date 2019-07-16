from collections import OrderedDict as Od
from utils import mu, iu, au
from typing import List, Tuple
import numpy as np
from clu.me import C
import pandas as pd

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)


class Analyze(object):
    def __init__(self):
        self.args = self.get_args()
        self.use_max: bool = self.args.m
        self.sort_group: bool = self.args.s
        self.choose_log: bool = self.args.c
        self.choose_log: bool = True
        self.inner_top: int = 5
        # self.group_top: int = 5
        self.exclude: set = None
        self.group_by: List = None
        self.log_path: str = None

    @staticmethod
    def read_file(file: str):
        e_flag = ''
        score_od = Od()
        for line in iu.read_lines(file):
            if line.startswith('{') and 'nmi' in line:
                for k, v in iu.loads(line).items():
                    score_od.setdefault(k, list()).append(v)
            elif line.startswith('e'):
                e_flag = line[line.find('-') + 1:]
        if len(score_od) == 0:
            print('{} is empty'.format(iu.get_name(file).split(',', maxsplit=1)[0]))
        epoch = len(list(score_od.values())[0]) if len(score_od) > 0 else 0
        return score_od, epoch, e_flag

    @staticmethod
    def read_board(files: List[str], use_max: bool, inner_top: int) -> pd.DataFrame:
        i = 0
        board = pd.DataFrame()
        for file in files:
            score_od, epoch, e_flag = Analyze.read_file(file)
            if len(score_od) == 0:
                continue
            # entries = au.name2entries(fname, exclude=Analyze.exclude, postfix='.txt')
            for p, v in au.name2entries(iu.get_name(file), postfix='.txt'):
                board.loc[i, p] = v
            for k, v in score_od.items():
                values = sorted(v) if use_max else v
                top_values = values[-inner_top:]
                board.loc[i, k] = np.round(np.mean(top_values), 4)
            board.loc[i, 'ep'] = '{} {}'.format(epoch, e_flag)
            i += 1
        return board

    def read_board_multi(self, files: List[str], use_max: bool, inner_top: int) -> pd.DataFrame:
        files_parts: List[List[str]] = mu.split_multi(files, process_num=10)
        args_list: List[Tuple] = [(fs, use_max, inner_top) for fs in files_parts]
        board_list: List[pd.DataFrame] = mu.multi_process(self.read_board, args_list)
        return pd.concat(board_list)

    def print_groups(self, files: List[str], out_name: str = None):
        read_func = self.read_board_multi if len(files) >= 20 else self.read_board
        board = read_func(files, self.use_max, self.inner_top)
        drop_cols = list(set(board.columns) & self.exclude)
        board = board.drop(drop_cols, axis=1).fillna('_')
        temp = 'score_sum'
        board[temp] = board[au.s_acc] + board[au.s_ari] + board[au.s_nmi]
        board.sort_values(by=[C.dn, temp], ascending=False, inplace=True)
        board.pop(temp)
        print(board)
        if out_name:
            board.to_csv('{}.csv'.format(out_name))

        # ret_dfs = list()
        # for data_name, dn_df in board.groupby(C.dn):
        #     print(data_name)
        #     for values, df in board.groupby(self.group_by):
        #         # for dn_bv, dn_df in au.group_data_frame_columns(board, columns=[C.dn]):
        #         # data_name = dn_bv[C.dn]
        #         # print(data_name)
        #         # bv_df_list = au.group_data_frame_columns(dn_df, columns=group_by)
        #         bv_df_sc_list = list()
        #         for bv, df in bv_df_list:
        #             evals = ['nmi', 'ari', 'acc']
        #             # evals += [s + '_nis' for s in evals]
        #             sc = Od((s, group_score(df, s)) for s in evals)
        #             bv_df_sc_list.append((bv, df, sc))
        #         if self.sort_group:
        #             # bv_df_sc_list = sorted(bv_df_sc_list, key=lambda item: (
        #             #     item[-1]['acc'], item[-1]['ari'], item[-1]['nmi']))
        #         ret_df = pd.DataFrame()
        #         for bv, df, sc in bv_df_sc_list:
        #             print('<{}>'.format(len(df)), end='    ')
        #             print(' '.join(['{}={:7}'.format(*x) for x in bv.items()]), end='|  ')
        #             print('    '.join(['{} {:.4f}'.format(s, g) for s, g in sc.items()]))
        #             print(df.iloc[:4, :], '\n')
        #             # print(','.join(df[gid_].tolist()))
        #             i = len(ret_df)
        #             for b, v in bv.items():
        #                 ret_df.loc[i, b] = v
        #             for s, c in sc.items():
        #                 ret_df.loc[i, s] = round(float(c), 4)
        #             ret_df.loc[i, 'LEN'] = len(df)
        #             ret_df.loc[i, C.dn] = data_name
        #         print(data_name, '\n' * 3)
        #         ret_dfs.append(ret_df)

    @staticmethod
    def get_args():
        from argparse import ArgumentParser
        parser = ArgumentParser()
        parser.add_argument('-m', action='store_false', default=True, help='using max scores')
        parser.add_argument('-s', action='store_false', default=True, help='if sort tables')
        parser.add_argument('-c', action='store_false', default=True, help='if choose file')
        return parser.parse_args()

    def get_log_path(self):
        new_paths = iu.list_children('./', iu.DIR, r'^log\d', True)
        old_paths = iu.list_children('./logs', iu.DIR, r'^log\d', True)
        log_paths = new_paths + old_paths
        log_path = iu.choose_from(log_paths) if self.choose_log else iu.most_recent(log_paths)
        print('logging path:', log_path)
        return log_path

    def main(self):
        # group_by = [l1_, l2_, l3_, l4_, vs_]
        #     comment = 'tune_l1_l3_mean'
        # if 'mean_pool_tune_l2' in log_path:
        #     group_by = [l1_, l2_, l3_, eps_, worc_]
        #     comment = 'tune_l2_mean'
        # if 'mean_pool_tune_eps' in log_path:
        #     group_by = [l1_, l2_, l3_, eps_, worc_]
        #     comment = 'tune_eps_mean'
        # if 'l3_eps' in log_path:
        #     group_by = [l1_, l2_, l3_, eps_, worc_]
        #     comment = 'l3_eps'
        # if 'no_trn_w_or_c' in log_path:
        #     group_by = [l1_, l2_, l3_, eps_, wtrn_, ctrn_, worc_]
        #     comment = 'no_trn_w_or_c'
        # if 'l1_l2_eps=50' in log_path:
        #     group_by = [l1_, l2_, l3_, eps_, worc_]
        #     # comment = 'l1_l2_eps=50'
        # if 'l2_eps_over' in log_path:
        #     group_by = [l1_, l2_, l3_, eps_, worc_]
        # # if 'ablation_eps=50' in log_path:
        # #     group_by = [l1_, l2_, l3_, eps_, worc_, vs_]
        # #     comment = 'ablation_eps=50'
        # if 'c_num' in log_path:
        #     exclude = {ep_, bs_, ns_, gi_, gp_}
        #     group_by = [l1_, l2_, l3_, eps_, cn_]
        #     comment = 'c_num'
        # # if 'abal_wo_L1L2' in log_path:
        # #     group_by = [l1_, l2_, l3_, worc_]
        # #     comment = 'abal_wo_L1L2'
        self.exclude = {C.ep, C.cn, C.ns, C.gi, C.gp, C.mgn}
        # self.group_by = None
        self.log_path = self.get_log_path()
        log_name = iu.get_name(self.log_path)
        print('Using {} scores'.format(['last', 'max'][int(self.use_max)]))
        log_files = iu.list_children(self.log_path, pattern=r'gid.+\.txt$', full_path=True)
        print(len(log_files))
        self.print_groups(log_files, out_name=log_name)


if __name__ == '__main__':
    Analyze().main()
