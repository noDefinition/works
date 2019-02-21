from collections import OrderedDict as Od

import pandas as pd

from clu.data.datasets import *

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)


def read_scores_from_file(file):
    e_flag = ''
    score_od = Od()
    for line in iu.read_lines(file):
        if line.startswith('b'):
            continue
        elif line.startswith('e'):
            e_flag = line[line.find('-') + 1:]
        elif line.startswith('{') and 'nmi' in line:
            for k, v in iu.loads(line).items():
                score_od.setdefault(k, list()).append(v)
    if len(score_od) == 0:
        fname = iu.base_name(file)
        print('{} - empty'.format(fname[:fname.find(',')]))
    epoch = len(list(score_od.values())[0]) if len(score_od) > 0 else 0
    return score_od, epoch, e_flag


def read_board_files(use_max, files):
    i = 0
    board = pd.DataFrame()
    for file in files:
        score_od, epoch, e_flag = read_scores_from_file(file)
        if len(score_od) == 0:
            continue
        fname = file[file.rfind('/') + 1:]
        for p, v in au.name2entries(fname, exclude=exclude, postfix='.txt'):
            board.loc[i, p] = v
        for k, v in score_od.items():
            inner_top = 5
            values = sorted(v)[-inner_top:] if use_max else v[-inner_top:]
            board.loc[i, k] = round(float(np.mean(values)), 4)
        board.loc[i, 'ep'] = '{} {}'.format(epoch, e_flag)
        i += 1
    return board


def read_board_multi(use_max, files):
    files_list = mu.split_multi(files, process_num=10)
    board_list = mu.multi_process(read_board_files, [(use_max, fs) for fs in files_list])
    return pd.concat(board_list)


def print_groups(use_max, sort_group, files):
    def group_score(frame, col):
        group_top = 5
        mean = np.mean(frame[col].values[1:group_top])
        return mean

    board = (read_board_multi if len(files) >= 60 else read_board_files)(use_max, files)
    board.fillna('_', inplace=True)
    for s in set(board.columns).intersection(set(group_by)):
        print(s, sorted(Counter(board[s])))
    sort_by = 'acc'
    board.sort_values(by=sort_by, ascending=False, inplace=True)

    ret_dfs = list()
    for dn_bv, dn_df in au.group_data_frame_columns(board, columns=[dn_]):
        data_name = dn_bv[dn_]
        print(data_name)
        bv_df_list = au.group_data_frame_columns(dn_df, columns=group_by)
        bv_df_sc_list = list()
        for bv, df in bv_df_list:
            evals = ['nmi', 'ari', 'acc']
            # evals += [s + '_nis' for s in evals]
            sc = Od((s, group_score(df, s)) for s in evals)
            bv_df_sc_list.append((bv, df, sc))
        if sort_group:
            bv_df_sc_list = sorted(bv_df_sc_list, key=lambda item: (
                item[-1]['acc'], item[-1]['ari'], item[-1]['nmi']))
        ret_df = pd.DataFrame()
        for bv, df, sc in bv_df_sc_list:
            # if not (bv[l1_] == '1.0' and bv[l2_] == '1.0'
            #         and bv[l3_] == '0.0'):
            #     continue
            print('<{}>'.format(len(df)), end='    ')
            print(' '.join(['{}={:7}'.format(*x) for x in bv.items()]), end='|  ')
            print('    '.join(['{} {:.4f}'.format(s, g) for s, g in sc.items()]))
            print(df.iloc[:4, :], '\n')
            # print(','.join(df[gid_].tolist()))
            i = len(ret_df)
            for b, v in bv.items():
                ret_df.loc[i, b] = v
            for s, c in sc.items():
                ret_df.loc[i, s] = round(float(c), 4)
            ret_df.loc[i, 'LEN'] = len(df)
            ret_df.loc[i, dn_] = data_name
        print(data_name, '\n' * 3)
        # if comment is not None:
        #     ret_df.to_csv('{}_{}.csv'.format(comment, data_name))
        ret_dfs.append(ret_df)
    pd.concat(ret_dfs).to_csv('{}.csv'.format(comment))


if __name__ == '__main__':
    from argparse import ArgumentParser
    from utils import mu
    from me import *

    parser = ArgumentParser()
    parser.add_argument('-m', action='store_false', default=True, help='using max scores')
    parser.add_argument('-s', action='store_false', default=True, help='if sort tables by nmi')
    parser.add_argument('-c', action='store_true', default=False, help='if prompt to choose file')
    args = parser.parse_args()
    log_paths = iu.list_children('./', iu.DIR, '^log', True)
    log_path = iu.choose_from(log_paths) if args.c else iu.most_recent(log_paths)

    # group_by = [l1_, l2_, l3_, l4_, vs_]
    #     comment = 'tune_l1_l3_mean'
    # if 'mean_pool_tune_l2' in log_path:
    #     group_by = [l1_, l2_, l3_, eps_, worc_]
    #     comment = 'tune_l2_mean'
    # if 'mean_pool_tune_eps' in log_path:
    #     group_by = [l1_, l2_, l3_, eps_, worc_]
    #     comment = 'tune_eps_mean'
    exclude = {ep_, cn_, bs_, ns_, gi_, gp_}
    group_by = comment = None
    if 'l3_eps' in log_path:
        group_by = [l1_, l2_, l3_, eps_, worc_]
        comment = 'l3_eps'
    if 'no_trn_w_or_c' in log_path:
        group_by = [l1_, l2_, l3_, eps_, wtrn_, ctrn_, worc_]
        comment = 'no_trn_w_or_c'
    if 'l1_l2_eps=50' in log_path:
        group_by = [l1_, l2_, l3_, eps_, worc_]
        # comment = 'l1_l2_eps=50'
    if 'l2_eps_over' in log_path:
        group_by = [l1_, l2_, l3_, eps_, worc_]
    # if 'ablation_eps=50' in log_path:
    #     group_by = [l1_, l2_, l3_, eps_, worc_, vs_]
    #     comment = 'ablation_eps=50'
    if 'c_num' in log_path:
        exclude = {ep_, bs_, ns_, gi_, gp_}
        group_by = [l1_, l2_, l3_, eps_, cn_]
        comment = 'c_num'
    # if 'abal_wo_L1L2' in log_path:
    #     group_by = [l1_, l2_, l3_, worc_]
    #     comment = 'abal_wo_L1L2'

    print('logging path:', log_path)
    print('Using {} scores'.format('max' if args.m else 'last'))
    _files = iu.list_children(log_path, pattern='gid.+\.txt$', full_path=True)
    print(len(_files))
    print_groups(args.m, args.s, _files)
