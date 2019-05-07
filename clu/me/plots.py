# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from data.datasets import Sampler
from me.layers import get_session, tf, np
from utils import au, iu

from argparse import ArgumentParser
from me import *
from me.gen1 import *


def show(name=None):
    if name is not None:
        plt.savefig(name + '.png', dpi=192)
        with PdfPages(name + '.pdf') as pdf:
            pdf.savefig()
    plt.show()
    plt.close()


def hide_spine_and_ticks(axis):
    for a in ['top', 'bottom', 'left', 'right']:
        axis.spines[a].set_visible(False)
    axis.set_xticks([])
    axis.set_yticks([])


def plot_curve(name=None):
    from .analyze import read_scores_from_file
    od, _, _ = read_scores_from_file(name)
    font = {'family': 'sans-serif', 'weight': 'normal', 'size': 12}
    # colors = ['red', 'darkorange', 'lime', 'blue', 'magenta']
    for name, scores in od.items():
        print(name, scores)
        epochs = len(scores)
        ax = plt.figure(figsize=(4, 3)).add_subplot(111)
        ax.tick_params(direction='in', right=True, top=True, labelsize=9)
        # x-axis
        x_ticks = list(range(1, 1 + len(scores)))
        ax.set_xlabel('Iterations', font)
        ax.set_xticks(x_ticks)
        ax.set_xlim(-1, epochs + 2)
        # y-axis
        ax.set_ylabel(name.upper(), font)
        ax.set_yticks(np.arange(0, 1, 0.1))
        ax.set_ylim(0, 1)

        ax.legend(name, loc='lower right', fontsize=9, frameon=False,
                  borderaxespad=0.3, labelspacing=0.3)
        ax.plot(x_ticks, scores)
        show(name)


def plot_scatter(points, labels, name=None):
    labels = au.reindex(labels)
    lbl2pts = dict()
    for point, label in zip(points, labels):
        lbl2pts.setdefault(label, list()).append(point)
    cmap = ColorMapper(len(set(labels)))
    ax = plt.figure(figsize=(10, 10)).add_subplot(111)
    hide_spine_and_ticks(ax)
    for lbl, pts in lbl2pts.items():
        x, y = np.array(pts).T
        ax.scatter(x, y, s=10, c=cmap[lbl])
    show(name)


def visualize_embedding(embeds, labels, name=None):
    from utils import ru
    points = ru.fit_tsne(embeds)
    plot_scatter(points, labels, name)


class ColorMapper:
    def __init__(self, n):
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        cmap = plt.get_cmap('hsv')
        norm = Normalize(vmin=0, vmax=n)
        self.value_set = set(range(0, n))
        self.scalar_map = ScalarMappable(norm=norm, cmap=cmap)

    def __getitem__(self, i):
        assert i in self.value_set
        return self.scalar_map.to_rgba(i)


def per_store_path(store_path):
    print(store_path)
    hyper_file = iu.join(store_path, 'hyper')
    param_file = iu.join(store_path, 'model.ckpt')
    args = iu.load_json(hyper_file)
    print('restore args from file', args)
    model_name = args[vs_]
    data_name = args[dn_]

    sampler = Sampler(data_name)
    w_embed, c_embed = sampler.d_obj.load_word_cluster_embed()
    eval_batches = sampler.eval_batches
    print('sample over')

    model_class = {v.__name__: v for v in [N5]}[model_name]
    model = model_class(args)
    model.build_model(w_embed, c_embed)
    print('model build over')
    sess = get_session(1, 0.1, allow_growth=True, run_init=True)
    model.set_session(sess)
    tf.train.Saver(tf.trainable_variables()).restore(sess, param_file)

    p_maxes = list()
    for batch in eval_batches:
        c_probs = sess.run(model.pc_probs, feed_dict=model.get_fd_by_batch(batch))
        p_maxes.extend(np.max(c_probs, axis=1).reshape(-1))
    sess.close()
    print('clusters get over')

    ax = plt.figure(figsize=(4, 3)).add_subplot(111)
    ax.tick_params(direction='in', right=True, top=True, labelsize=9)
    font = {'family': 'sans-serif', 'weight': 'normal', 'size': 12}
    # x-axis
    ax.set_xlabel('max probability', font)
    # y-axis
    ax.set_ylabel('density', font)
    # ax.set_yticks(np.arange(0, 1, 0.1))
    # ax.set_ylim(0, 1)
    # ax.legend(name, loc='lower right', fontsize=9, frameon=False,
    #           borderaxespad=0.3, labelspacing=0.3)
    span = 10000
    ax.hist(p_maxes, density=False, bins=np.arange(0, span) / span)
    show()


def plot_max_c_probs():
    paths = iu.list_children('./', iu.DIR, '^log', True)
    log_path = (iu.choose_from if False else iu.most_recent)(paths)
    store_paths = iu.list_children(log_path, iu.DIR, pattern='gid=67', full_path=True)
    for store_path in store_paths:
        try:
            per_store_path(store_path)
            tf.reset_default_graph()
        except Exception as e:
            print(e)


if __name__ == '__main__':
    # np.random.randint(low=)
    pass
