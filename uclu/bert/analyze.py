from collections import OrderedDict as Od
from utils import mu, iu, au
from typing import List, Tuple, Dict
import numpy as np
from uclu.bert import UB, UcluBertArgs
import pandas as pd

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)


class Analyze(object):
    def __init__(self):
        self.args = self.get_args()
        self.use_max: bool = self.args.m
        self.sort_group: bool = self.args.s
        self.choose_log: bool = self.args.c
        self.inner_top: int = 5
        self.exclude: set = {UB.ep, UB.cn, UB.gi, UB.gp}
        self.log_path: str = './'

    @staticmethod
    def read_line(line: str) -> Dict:
        if 'acc' not in line:
            return None
        return iu.loads(line)

    def read_file(self, file: str) -> Dict:
        score_od = Od()
        for line in iu.read_lines(file):
            obj = self.read_line(line)
            if obj is not None:
                for k, v in iu.loads(line).items():
                    score_od.setdefault(k, list()).append(v)
        return score_od

    def read_board(self, files: List[str], use_max: bool, inner_top: int) -> pd.DataFrame:
        s_list = list()
        for file in files:
            fname = iu.get_name(file)
            params = au.name2entries(fname, exclude=self.exclude, postfix='.txt')
            params = dict(params)
            score_od = self.read_file(file)
            if len(score_od) == 0:
                print('{} is empty'.format(fname))
                continue
            for k, v in score_od.items():
                values = sorted(v) if use_max else v
                top_values = values[-inner_top:]
                params[k] = np.round(np.mean(top_values), 4)
            s_list.append(pd.Series(params))
        return pd.concat(s_list, axis=1).T

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
        temp = 'acc'
        board.sort_values(by=[UB.dn, temp], ascending=False, inplace=True)
        print(board)
        if out_name:
            board.to_csv('{}.csv'.format(out_name))

    @staticmethod
    def get_args():
        from argparse import ArgumentParser
        parser = ArgumentParser()
        parser.add_argument('-m', action='store_false', default=True, help='using max scores')
        parser.add_argument('-s', action='store_false', default=True, help='if sort tables')
        parser.add_argument('-c', action='store_false', default=True, help='if choose file')
        return parser.parse_args()

    def get_log_path(self):
        new_paths = iu.list_children('./', iu.DIR, r'^(log)?\d+', True)
        if iu.exists('./logs'):
            old_paths = iu.list_children('./logs', iu.DIR, r'^(log)?\d+', True)
        else:
            old_paths =[]
        log_paths = (new_paths + old_paths)
        log_path = iu.choose_from(log_paths) if self.choose_log else iu.most_recent(log_paths)
        print('logging path:', log_path)
        return log_path

    def main(self):
        self.log_path = self.get_log_path()
        log_name = iu.get_name(self.log_path)
        print('Using {} scores'.format(['last', 'max'][int(self.use_max)]))
        log_files = iu.list_children(self.log_path, pattern=r'gid.+\.txt$', full_path=True)
        print(len(log_files))
        self.print_groups(log_files, out_name=log_name)


if __name__ == '__main__':
    Analyze().main()
