import os
from time import time

import numpy as np
import pandas as pd

import gensim
from gensim.corpora.dictionary import Dictionary
from gensim.models.wrappers.dtmmodel import DtmModel

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import csc_matrix, csr_matrix

from dateutil import parser
import datetime


def dateparser(date_str):
    try:
        date = parser.parse(date_str).replace(tzinfo=None)
    except:
        date = datetime.datetime(year=2014, month=06, day=22).replace(tzinfo=None)  # first date in the corpus
    return date


tic = time()

#loading aggregated comments file
agg_comm_file = '../data/aggregated_comment.pkl'

if os.path.exists(agg_comm_file):
    df_comments = pd.read_pickle(agg_comm_file)
else:
    dfs = list()
    for dirname, dirnames, filenames in os.walk('../data/'):
        for filename in filenames:
            if filename.startswith('2013'):
                dataset = os.path.join(dirname, filename)
                tmp = pd.read_csv(dataset, parse_dates=[6], date_parser=dateparser)
                dfs.append(tmp)
    df = pd.concat(dfs)

    sorted_df = df.sort_values(by=['postTimestamp'])
    df_comments = sorted_df.ix[(sorted_df['edgeType'] == 'Comment')]
    df_comments.to_pickle(agg_comm_file)

text = list(df_comments['commentText'])
times = list(df_comments['postTimestamp'])
comments_text = [comment for comment in text if type(comment) == str]
comments_time = [times[i] for i, comment in enumerate(text) if type(comment) == str]

print("Loading Dataset Done! %f" % (time()-tic))
tic = time()

min_df = 5  # ignore terms that have a document frequency strictly lower than the given threshold

tf_vectorizer = CountVectorizer(min_df=min_df, stop_words='english')
dt_mat = tf_vectorizer.fit_transform(comments_text)

tfidf_vectorizer = TfidfVectorizer(min_df=min_df, stop_words='english')
tfidf_mat = tfidf_vectorizer.fit_transform(comments_text)

median = np.median(tfidf_mat.data)
dt_mat.data[tfidf_mat.data < median] = 0
# dt_mat[tfidf_mat < median] = 0
dt_mat.eliminate_zeros()

print("Vectorization Done! %f" % (time()-tic))
tic = time()

tf = np.array(np.squeeze(np.sum(dt_mat, 0)).tolist()[0])
dlen = np.array(np.squeeze(np.sum(dt_mat, 1)).tolist()[0])
# dt_mat = dt_mat[np.ix_(dlen != 0, tf > 0)]

dt_mat = csr_matrix(dt_mat)
dt_mat = dt_mat[dlen != 0, :]
dt_mat = csc_matrix(dt_mat)
dt_mat = dt_mat[:, tf > 0]
dt_mat = csr_matrix(dt_mat)

print("Dimension Adjustment Done! %f" % (time()-tic))
tic = time()

oldvoca = tf_vectorizer.get_feature_names()
voca = [oldvoca[i] for i, wi in enumerate(tf) if wi > 0]
id2word = {i: word for i, word in enumerate(voca)}

tmp = [comments_time[i] for i, _len in enumerate(dlen) if _len > 0]
comments_time = tmp

corpus = gensim.matutils.Sparse2Corpus(dt_mat.T)
n_terms, n_docs = corpus.sparse.shape
dictionary = Dictionary.from_corpus(corpus, id2word=id2word)

n_time = 12
time_seq = [dictionary.num_docs / n_time] * n_time
time_seq[-1] += dictionary.num_docs - sum(time_seq)
cum_time = np.cumsum(time_seq)

# path to dtm home folder
dtm_home = os.environ.get('DTM_HOME', "dtm-master")
# path to the binary. on my PC the executable file is dtm-master/bin/dtm
dtm_path = os.path.join(dtm_home, 'bin', 'dtm') if dtm_home else None

# num topics
n_topics = 20
alpha = 0.01
top_chain_var = 0.005

# destination for result
dst = "../result/dtm/all/"
if not os.path.exists(dst):
    os.makedirs(dst)
# filename = "maxdf_%.2f_mindf_%.2f_topic_%d_equal_%d_alpha_%.3f_var_%.3f.model" % (
#     max_df, min_df, n_topics, n_time, alpha, top_chain_var)
filename = "median_mindf_%.2f_topic_%d_equal_%d_alpha_%.3f_var_%.3f.model" % (
    min_df, n_topics, n_time, alpha, top_chain_var)

print("DTM init! %f" % (time()-tic))

if not os.path.exists(os.path.join(dst, filename)):
    tic = time()
    model = DtmModel(dtm_path, corpus, time_seq, num_topics=n_topics,
                     id2word=dictionary, initialize_lda=True, alpha=alpha,
                     top_chain_var=top_chain_var, rng_seed=0)
    print('Training DTM Done! %f' % (time() - tic))
    model.save(os.path.join(dst, filename))
else:
    model = DtmModel.load(os.path.join(dst, filename))
