from clu.data.datasets import *
from gensim.models import Word2Vec
import utils.multiprocess_utils as mu
import utils.reduction_utils as ru


def train_embedding(data_class):
    docarr = data_class().load_docarr()
    sentences = [d.tokens for d in docarr]
    model = Word2Vec(sentences, size=300, min_count=0, seed=233, workers=8, null_word=None, iter=100)
    doc_avg = np.array([np.mean([model[w] for w in d.tokens], axis=0) for d in docarr])
    return doc_avg


def tsne_multi():
    class_list = [DataTrec, DataGoogle, DataEvent, DataReuters, Data20ng]
    doc_avg_list = mu.multi_process(train_embedding, [[c] for c in class_list])
    for m in doc_avg_list:
        print(m.shape)
    kw_arg = dict(early_exaggeration=12, n_iter=800, n_iter_without_progress=100)
    tsne_point_list = ru.fit_multi(ru.fit_tsne, doc_avg_list, [kw_arg] * len(doc_avg_list))
    for m in tsne_point_list:
        print(m.shape)
    name_point_list = list(zip([c.name for c in class_list], tsne_point_list))
    return name_point_list


def to_gnuplot(name_point_list):
    def labels2indexes(labels):
        sorted_labels = sorted(set(labels))
        label2idx = dict((label, idx) for idx, label in enumerate(sorted_labels))
        return [label2idx[label] for label in labels]
    
    def points_topics2file(points, indexes, outfile):
        lines = ['{:.4f} {:.4f} {}'.format(*p, i) for p, i in zip(points, indexes)]
        iu.write_lines(outfile, lines)
    
    for d_name, points in name_point_list:
        topic_list = name2d_object[d_name].get_topics()
        index_list = labels2indexes(topic_list)
        points_topics2file(points, index_list, '{}_stcc.txt'.format(d_name))


if __name__ == '__main__':
    stcc_tsne_file = 'tsne_stcc.npy'
    # np.save(stcc_tsne_file, np.array(name_point_list, dtype=object))
    to_gnuplot(tsne_multi())
