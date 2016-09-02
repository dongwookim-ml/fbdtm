import os
from time import time

import numpy as np
import pandas as pd
import codecs

from gensim.models.wrappers.dtmmodel import DtmModel


max_df = 1  # ignore terms that have a document frequency strictly higher than the given threshold
min_df = 5  # ignore terms that have a document frequency strictly lower than the given threshold
n_time = 12
n_topics = 20
alpha = 0.01
top_chain_var = 0.005

# destination of result
dst = "../result/dtm/all/"
filename = "median_mindf_%.2f_topic_%d_equal_%d_alpha_%.3f_var_%.3f.model" % (
    min_df, n_topics, n_time, alpha, top_chain_var)
model = DtmModel.load(os.path.join(dst, filename))

n_terms = 20

with codecs.open("../result/dtm/all/median_topics.csv", 'w', 'utf-8') as f:
    for k in range(n_topics):
        f.write('Topic %d\n' % (k))
        for ti in range(n_time):
            topic = model.show_topic(topicid=k, time=ti, topn=n_terms)
            topwords = ','.join([word for (prob, word) in topic])
            f.write('time %d, %s\n' % (ti, topwords))
