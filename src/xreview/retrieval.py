"""Nearest training utterances by TF-IDF cosine similarity, used as worked examples in the request."""

from sklearn.feature_extraction.text import TfidfVectorizer

from .data import Utterance


class Retriever:
    def __init__(self, train: list[Utterance], k: int):
        self.train = train
        self.k = k
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([u.text for u in train])

    def nearest(self, text: str) -> list[Utterance]:
        sims = (self.matrix @ self.vectorizer.transform([text]).T).toarray().ravel()
        order = sims.argsort()[::-1][: self.k]
        return [self.train[i] for i in reversed(order)]
