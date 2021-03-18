from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime
import numpy as np
from transformers import BertTokenizer


class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.count_vec = CountVectorizer()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def fit(self, x, y):
        # Description features
        self.count_vec.fit(x.desc.astype(str))
        return self

    def transform(self, x):
        desc_fts = self.count_vec.transform(x.desc.astype(str)).toarray()
        date_fts = x.date.astype(str).apply(lambda x: datetime.fromisoformat(x).isoweekday())
        x = np.hstack(
            (desc_fts, date_fts.to_numpy().reshape(-1, 1), x.val.to_numpy().reshape(-1, 1))
        )
        return x


def get_label_map(a):
    map = {}
    for i, cat in enumerate(a):

        map[cat.strip("\n").strip().lower()] = i

    return map
