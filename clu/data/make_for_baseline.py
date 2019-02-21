import gensim

import utils.array_utils as au
import utils.io_utils as iu
import utils.multiprocess_utils as mu

embed_google_news = 'google'
embed_glovec_6B = 'glovec6B'
embed_glovec_27B = 'glovec27B'


def _word2vec_from_lines(lines):
    word2vec = dict()
    for line in lines:
        splits = line.strip().split_length(' ')
        word2vec[splits[0]] = list(map(float, splits[1:]))
    return word2vec


def _load_word2vec_from_glovec_file(glovec_file, process_num):
    lines_parts = mu.split_multi(iu.read_lines(glovec_file), process_num)
    print('lines read & partition over')
    word2vec_parts = mu.multi_process(_word2vec_from_lines, [(lines,) for lines in lines_parts])
    word2vec = dict()
    for word2vec_part in word2vec_parts:
        word2vec.update(word2vec_part)
    return word2vec


def load_word2vec_by_name(name):
    base = "/home/cdong/works/research/input_and_outputs/word_embeddings/"
    embed_name2func = {
        embed_google_news: lambda: gensim.models.KeyedVectors.load_word2vec_format(
            base + "GoogleNews-vectors-negative300.bin", binary=True),
        embed_glovec_6B: lambda: _load_word2vec_from_glovec_file(
            base + "glove.6B.300d.txt", process_num=10),
        embed_glovec_27B: lambda: _load_word2vec_from_glovec_file(
            base + "glove.twitter.27B.200d.txt", process_num=10),
    }
    return embed_name2func[name]()


def matrix2str_list(matrix, delimeter, ndigits):
    return [delimeter.__mul__([str(round(v, ndigits)) for v in vector]) for vector in matrix]


def matrix2str_list_multi(matrix, delimeter, ndigits, process_num):
    matrix_parts = mu.split_multi(matrix, process_num)
    arg_list = [(matrix_part, delimeter, ndigits) for matrix_part in matrix_parts]
    return au.merge(mu.multi_process(matrix2str_list, arg_list))
