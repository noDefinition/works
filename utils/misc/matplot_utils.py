import matplotlib

matplotlib.use('Agg')
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import utils.io_utils as fu
import re
from pathlib import Path
import utils.file_iterator as fi
from scipy.special import comb, gamma


def figure(X, Y, fig_name):
    # plt.figure(figsize=(13, 7))
    plt.plot(X, Y, color="blue", linewidth=1)
    plt.xlim([-0.03, 1.03])
    plt.ylim([-0.03, 1.03])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title("roc curve")
    plt.legend(loc='lower right')
    plt.savefig(fig_name, format='png')


def binomial(p, n, k_range):
    return [comb(n, k) * np.power(p, k) * np.power(1 - p, n - k) for k in k_range]


def plot_p_eta_given_m_and_epsilon():
    font = {
        # 'family': 'serif',
        # 'color': 'black',
        'weight': 'normal',
        'size': 20,
    }
    
    args = [(M, eta) for M in (30,) for eta in [i / 10 for i in range(1, 10)]]
    
    for M, eta in args:
        full_x = [i for i in range(M + 1)]
        full_y = binomial(eta, M, full_x)
        # split = M + 1
        
        # alpha = 0.05
        # alpha_fractile = 13
        # eta = [i / 1000 + 0.2 for i in range(100 + 1)]
        # for e in eta:
        #     full_y = binomial(e, M, full_x)
        #     s = 0
        #     for idx in range(alpha_fractile, M):
        #         s += full_y[idx]
        #         if s > alpha:
        #             print(s, e)
        #             exit()
        # split = s = 0
        # for idx, y_ in enumerate(full_y[::-1]):
        #     s += y_
        #     if s > alpha:
        #         split = M - idx + 1
        #         print(split)
        #         break
        # exit()
        split = 13
        eta = 0.273
        full_y = binomial(eta, M, full_x)
        
        x1 = full_x[:split]
        y1 = full_y[:split]
        x2 = full_x[split:]
        y2 = full_y[split:]
        
        plt.close()
        plt.figure(figsize=(10, 6))
        plt.xticks(x1 + x2)
        ylim = 0.20
        plt.ylim((0, ylim))
        # plt.yticks([ylim / (i + 1) for i in range(4)])
        plt.xlabel('m\'', fontdict=font)
        plt.ylabel('P(m\'|m,η)', fontdict=font)
        title = 'm={}\nη={}'.format(M, round(eta, 3))
        plt.text(M * 0.72, ylim * 0.72, title, fontsize=40, color="r")
        plt.bar(x1, y1, fc='b')
        plt.bar(x2, y2, fc='g')
        plt.savefig('{}.png'.format(title.replace('\n', ',')))


if __name__ == '__main__':
    plot_p_eta_given_m_and_epsilon()
    exit()
    font = {'weight': 'normal', 'size': 20, }
    
    args = [(M, m_) for M in (20, 30, 40) for m_ in [8 * i for i in range(10) if 0 < 8 * i < M]]
    
    # M = 20
    # m_ = 4
    g = 1000
    
    for M, m_ in args:
        x = [i / g for i in range(g + 1)]
        c = comb(M, m_)
        y = [c * np.power(x_, m_) * np.power(1 - x_, M - m_) for x_ in x]
        print(sum(y), min(x))
        
        plt.close()
        plt.figure(figsize=(10, 6))
        # plt.xticks(x)
        ylim = 0.20
        plt.ylim((0, ylim))
        plt.yticks([i * 0.05 for i in range(6)])
        plt.xlabel('η', fontdict=font)
        plt.ylabel('P(η|m,m\')', fontdict=font)
        print(x[0:10])
        plt.bar(x, y,  fc='b', align='center', width=1/g)
        
        title = 'm={}\nm\'={}'.format(M, m_)
        plt.text(0.75, ylim * 0.8, title, fontsize=40, color="r")
        plt.savefig('{}.png'.format(title.replace('\n', ',')))
    
    exit()
    
    # files = fi.listchildren('/home/nfs/cdong/tw/testdata/output2', children_type=fi.TYPE_DIR, concat=True)
    # ls = list()
    # for f in files:
    #     p = Path(f)
    #     s = p.stat()
    #     digits = re.findall('\d+', p.name)
    #     cluid, clunum = list(map(int, digits))
    #     ls.append((cluid, clunum, s.st_mtime))
    #
    # ls = sorted(ls, key=lambda item: item[0])
    # print(ls)
    # dt = [(ls[0][0], ls[0][1], 0)] + [(ls[i][0], ls[i][1], int(ls[i][2] - ls[i - 1][2])) for i in
    #                                   range(1, len(ls))]
    # print(dt)
    #
    # # x, y_cnum, y_dt = list(zip(*dt[:50]))
    # x1, y_cnum, y_dt = list(zip(*dt))
    # y_cnum_norm = np.array(y_cnum) / np.max(y_cnum)
    # y_dt_norm = np.array(y_dt) / np.max(y_dt)
    # print('time sum: {}'.format(np.sum(y_dt)))
    # print('max cnum:{}, min cnum:{}'.format(max(y_cnum), min(y_cnum)))
    # print('max dt:{}, min dt:{}'.format(max(y_dt), min(y_dt)))
    #
    # plt.figure(figsize=(20, 10))
    # plt.plot(x1, y_cnum_norm, ".-", color="blue", linewidth=0.5)
    # plt.plot(x1, y_dt_norm, "x-", color="red", linewidth=0.5)
    # plt.savefig('test_zhexian.png')
    # plt.close()
    
    # import utils.array_utils as au
    # labelarr, probarr = fu.load_array("/home/nfs/cdong/tw/src/preprocess/filter/prb_lbl_arr.txt")
    # from sklearn.metrics import roc_curve
    # fpr, tpr, thresholds = roc_curve(labelarr, probarr)
    # figure(fpr, tpr, "/home/nfs/cdong/tw/src/preprocess/filter/roc_curve.png")
    # au.precision_recall_threshold(labelarr, probarr, file="/home/nfs/cdong/tw/src/preprocess/filter/performance.csv",
    #                               thres_range=[i / 100 for i in range(1, 10)] + [i / 20 for i in range(2, 20)])
