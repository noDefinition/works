import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.layers import Conv1D, Dense, Dropout
from keras.layers import Embedding, Input
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

import utils.array_utils as au
from clu.baselines.short_cnn_grid import get_all_i_need

args, logger, d_class = get_all_i_need()
clu_num = args.cn
embed_dim = args.ed
window_size = args.ws
dropout = args.do
batch_size = args.bs

gpu_options = tf.GPUOptions(allow_growth=False, per_process_gpu_memory_fraction=args.gp)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())
set_session(sess)


def median_binarize(matrix):
    median = np.median(matrix, axis=1)[:, None]
    binary = np.zeros(shape=np.shape(matrix))
    binary[matrix > median] = 1
    return binary


# def map_label(trues, preds):
#     label_pair = list(zip(preds, trues))
#     count = tuple(Counter(label_pair).items())
#     mapping = dict()
#     n_label = len(np.unique(trues))
#     # map most likely labels from prediction to ground truth
#     for label in range(n_label):
#         tuples = [tup for tup in count if tup[0][0] == label]
#         likely_tuple = max(tuples, key=itemgetter(1))[0]
#         mapping[likely_tuple[0]] = likely_tuple[1]
#     pred_labels_mapped = [mapping[x] for x in preds]
#     return pred_labels_mapped


# def cluster_quality(trues, preds):
#     # h, c, v = metrics.homogeneity_completeness_v_measure(trues, preds)
#     nmi = metrics.normalized_mutual_info_score(trues, preds)
#     # ari = metrics.adjusted_rand_score(trues, preds)
#     # preds_mapped = map_label(trues, preds)
#     # acc = metrics.accuracy_score(trues, preds_mapped)
#     # if show:
#     #     print()
#     #     print("Homogeneity:     %0.4f" % h)
#     #     print("Completeness:    %0.4f" % c)
#     #     print("V-measure:       %0.4f" % v)
#     print("NMI:             %0.4f" % nmi)
#     #     print("Rand score:      %0.4f" % ari)
#     #     print("Accuracy:        %0.4f" % acc)
#     # return dict(homogeneity=h, completeness=c, vmeasure=v, nmi=nmi, rand=ari, accuracy=acc)
#     return nmi


#################################################
# reading data
#################################################
# EMBED_FILE = '/home/cdong/works/research/input_and_outputs/word_embeddings/GoogleNews-vectors-negative300.bin'
ifd, docarr = d_class.load_ifd_and_docarr()
word_index = ifd.get_word2id()
sequences_full = [d.tokenids for d in docarr]
target = [d.topic for d in docarr]

# text_path = d_class.text_file
# label_path = d_class.gnd_file
# with open(text_path) as f1, open(label_path) as f2:
#     data = [text.strip() for text in f1]
#     target = [int(label.rstrip('\n')) for label in f2.readlines()]
# tokenizer = Tokenizer(char_level=False)
# tokenizer.fit_on_texts(data)
# sequences_full = tokenizer.texts_to_sequences(data)
# tokenizer.fit_on_sequences(sequences_full)
# word_index = tokenizer.word_index

tokenizer = Tokenizer(char_level=False)
tokenizer.word_index = word_index
tokenizer.fit_on_sequences(sequences_full)
seq_lens = [len(s) for s in sequences_full]
MAX_SEQ_LEN = max(seq_lens)
print("Total: %s short texts" % format(len(docarr), ","), ' %s unique tokens.' % len(word_index))
print("Average length: %d" % np.mean(seq_lens), ", Max length: %d" % max(seq_lens))

X = pad_sequences(sequences_full, maxlen=MAX_SEQ_LEN)
y = target

#################################################
# Preparing embedding matrix
#################################################

EMBED_DIM = 300
NB_WORDS = len(word_index) + 1
word2vec = d_class().load_word2vec()
embed_matrix = np.zeros((NB_WORDS, EMBED_DIM))
for word, index in word_index.items():
    if word in word2vec:
        embed_matrix[index] = word2vec[word]
# print('Null word embeddings: %d' % np.sum(np.sum(embed_matrix, axis=1) == 0))
null_cnt = np.sum(np.sum(embed_matrix, axis=1) == 0)
cover_cnt = len(word_index) - null_cnt
print('Embedding coverage: {} / {} = {}'.format(cover_cnt, len(word_index), cover_cnt / len(word_index)))

# embed_npy_file = './short_cnn.npy'
# if fi.exists(embed_npy_file):
#     print('Loading embedding matrix')
#     embed_matrix = np.load(embed_npy_file)
# else:
#     print('Preparing embedding matrix')
#     word2vec = KeyedVectors.load_word2vec_format(EMBED_FILE, binary=True)
#     embed_matrix = np.zeros((NB_WORDS, EMBED_DIM))
#     for word, index in word_index.items():
#         if word in word2vec.vocab:
#             embed_matrix[index] = word2vec.word_vec(word)
#         else:
#             print(word)
#     np.save(embed_npy_file, embed_matrix)
# print('Null word embeddings: %d' % np.sum(np.sum(embed_matrix, axis=1) == 0))


#################################################
# Preparing target using Average embeddings (AE)
# B = embed_matrix normalization using tfidf
#################################################

tfidf = tokenizer.sequences_to_matrix(sequences_full, mode='tfidf')
tfidf_norm = 1 + np.sum(tfidf, axis=1)[:, None]
normed_tfidf = tfidf / tfidf_norm
average_embed = np.dot(normed_tfidf, embed_matrix)
print("Shape of average embedding: ", average_embed.shape)
B = median_binarize(average_embed)
# Last dimension in the CNN, equals VOCAB_SIZE
TARGET_DIM = B.shape[1]

################################################
# construct model
################################################
is_embed_trainable = True
# Embedding layer
pretrained_embedding_layer = Embedding(
    input_dim=NB_WORDS,
    output_dim=EMBED_DIM,
    weights=[embed_matrix],
    input_length=MAX_SEQ_LEN,
)
# Input
sequence_input = Input(shape=(MAX_SEQ_LEN,), dtype='int32')
embedded_sequences = pretrained_embedding_layer(sequence_input)

# 0th Layer
x = Conv1D(embed_dim, window_size, activation='tanh', padding='same')(embedded_sequences)
# 1st Layer
x = GlobalMaxPooling1D()(x)
# 2nd Layer
x = Dropout(dropout)(x)
# 3rd Layer
predictions = Dense(TARGET_DIM, activation='sigmoid')(x)

# model / gen2 / optimizer
model = Model(sequence_input, predictions)
model.layers[1].trainable = is_embed_trainable
adam = Adam(lr=5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-3)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['mae'])
# Fine-tune embeddings or not
model.summary()

if __name__ == '__main__':
    # checkpoint = ModelCheckpoint('models_prev/weights.{epoch:03d}-{val_acc:.4f}.hdf5',
    #                              monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
    for i in range(3):
        hist = model.fit(X, B, validation_split=0.1, epochs=100,
                         batch_size=batch_size, verbose=0, shuffle=True)
        
        # create model that gives penultimate layer
        inputs = model.layers[0].input
        outputs = model.layers[-2].output
        model_penultimate = Model(inputs, outputs)
        # inference of penultimate layer
        H = model_penultimate.predict(X)
        V = normalize(H, norm='l1')
        print("Sample shape: {}".format(H.shape))
        
        # n_clusters = len(np.unique(y))
        # print("Number of classes: %d" % n_clusters)
        km = KMeans(n_clusters=clu_num, n_jobs=4, max_iter=200)
        km.fit(V)
        pred = km.labels_
        # nmi = cluster_quality(y, pred)
        
        d = dict([(s, au.score(y, pred, s)) for s in ['nmi', 'ari']])
        print(d)
        # logger.info(entries2name(d, inter=' ', intra=':', postfix=''))
        # np.save("pred.npy", pred)
        # model.save_weights("model.plk")
