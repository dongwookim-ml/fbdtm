import os
from time import time

import numpy as np
import pandas as pd

import gensim
from gensim.corpora.dictionary import Dictionary
from gensim.models.wrappers.dtmmodel import DtmModel

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from dateutil import parser
import datetime


def dateparser(date_str):
    try:
        date = parser.parse(date_str).replace(tzinfo=None)
    except:
        date = datetime.datetime(year=2014, month=06, day=22).replace(tzinfo=None)  # first date in the corpus
        # date = datetime.datetime.now().replace(tzinfo=None)
    return date


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
text = list(df_comments['commentText'])
times = list(df_comments['postTimestamp'])
comments_text = [comment for comment in text if type(comment) == str]
comments_time = [times[i] for i, comment in enumerate(text) if type(comment) == str]

max_df = 1  # ignore terms that have a document frequency strictly higher than the given threshold
min_df = 5  # ignore terms that have a document frequency strictly lower than the given threshold

tf_vectorizer = CountVectorizer(min_df=min_df, stop_words='english')
dt_mat = tf_vectorizer.fit_transform(comments_text)

corpus = gensim.matutils.Sparse2Corpus(dt_mat.T)
n_terms, n_docs = corpus.sparse.shape
id2word = {i: word for i, word in enumerate(tf_vectorizer.get_feature_names())}
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
filename = "maxdf_%.2f_mindf_%.2f_topic_%d_equal_%d_alpha_%.3f_var_%.3f.model" % (
    max_df, min_df, n_topics, n_time, alpha, top_chain_var)

if not os.path.exists(os.path.join(dst, filename)):
    tic = time()
    model = DtmModel(dtm_path, corpus, time_seq, num_topics=n_topics,
                     id2word=dictionary, initialize_lda=True, alpha=alpha,
                     top_chain_var=top_chain_var, rng_seed=0)
    print(time() - tic)
    model.save(os.path.join(dst, filename))
else:
    model = DtmModel.load(os.path.join(dst, filename))
