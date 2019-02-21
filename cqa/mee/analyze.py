import pandas as pd
from utils import iu, au
from cqa.mee import *

from argparse import ArgumentParser


def main(log_path):
    log_files = iu.list_children(log_path, pattern='^gid.+\.txt$', full_path=True)
    print('log path:', log_path)
    best_list = list()
    for file in log_files:
        entries = au.name2entries(
            name=iu.get_name(file), postfix='.txt',
            exclude={dn_, gi_, gp_, es_, ep_, sc_, lid_, fda_, bs_}
        )
        scores = [iu.loads(l) for l in iu.read_lines(file) if (l.startswith('{') and 'valid' in l)]
        if len(scores) == 0:
            print(au.entries2name(entries), 'lacks test info')

        best_score = [score for score in scores if 'test' in score][-1]
        best_score.pop('brk_cnt')
        name2score = dict()
        for name in ['valid', 'test']:
            for k , v in best_score[name].items():
                name2score['{}_{}'.format(name, k)] = round(v, 4)
        best_list.append((dict(entries), name2score))

    df = pd.DataFrame()
    # best_list = sorted(best_list, key=lambda item: int(item[0][gid_]))
    for i, (name2param, name2score) in enumerate(best_list):
        for k, v in list(name2param.items()) + list(name2score.items()):
            df.loc[i, k] = v
    df.fillna('-', inplace=True)
    print(df.sort_values(by='valid_NDCG'))
    # print(df)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', action='store_true', default=False, help='if choose log path')
    args = parser.parse_args()

    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.width', 1000)

    cand_paths = iu.list_children('./', iu.DIR, '^log')
    if args.c:
        _log_path = iu.choose_from(cand_paths)
    else:
        _log_path = iu.most_recent(cand_paths)

    main(_log_path)
