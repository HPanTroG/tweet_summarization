from numpy.linalg import norm
from fast_pagerank import pagerank
import numpy as np
import scipy


class Lexrank:
    """
    lexrank model combined with lsh & cosine similarity
    """

    def __init__(self, data, lsh):
        self.data = data
        self.lsh = lsh
        self.graph = {}

    def build_graph(self, search_radius=0, cosine_sim=0.3, percent=1):

        # in case of applying fast_pagerank library
        matrix_indices = []
        weights = []

        buckets = self.lsh.extract_nearby_bins(max_search_radius=search_radius)
        print("#buckets: {}".format(len(buckets)))
        k = 0
        for b in buckets:
            sents = self.data[b]  # get list of sentVecs
            if k % 100 == 0:
                print(".......Buck: {}, vec: {}".format(k, sents.shape))

            num = np.dot(sents, sents.T)
            if scipy.sparse.issparse(sents):
                magnitude = norm(sents.toarray(), axis=1)
            else:
                magnitude = norm(sents, axis=1)
            den = np.dot(magnitude.reshape((-1, 1), magnitude.T.reshape(1, -1)))

            cosine_matrix = np.array(num / den)
            indices = np.where(cosine_matrix > cosine_sim)  # find positions with cosine values > cosine_sim

            n = cosine_matrix.shape[0]  # number of sentences = data.shape[0]
            assert n == self.data.shape[0]
            if percent != 1:
                num_sents = int(
                    len(indices[0]) * percent)  # only get %percent of sentence-pairs with cosine values > cosine_sim
                arr = cosine_matrix.flatten()

                kmax = np.argpartition(arr, -num_sents)[-num_sents, :]  # find k max elements in an array
                indices = [[int(x / n) for x in kmax], [x % n for x in kmax]]  # convert array indices to matrix indices

            # build graph
            for row, col in zip(indices[0], indices[1]):
                if row != col:  # ignore diagonal elements

                    matrix_indices.append([b[row], b[col]])
                    weights.append(cosine_matrix[row][col])

                    if b[row] not in self.graph:
                        self.graph[b[row]] = {}
                    if percent != 1:
                        if b[col] not in self.graph:
                            self.graph[b[col]] = cosine_matrix[row][col]
                        self.graph[b[col]][b[row]] = cosine_matrix[row][col]
                    self.graph[b[row]][b[col]] = cosine_matrix[row][col]
            k += 1

    # using pagerank pagekage
    def page_rank(self, damping_factor=0.85):
        pr = pagerank(self.matrix, p=damping_factor)
        self.scores = pr

    def train(self, lexrank_iter=100, damping_factor=0.85):
        n = self.data.shape[0]

        # for each node: compute sum of weights of adjacent nodes
        sum_weights = {}
        for sent, adj in self.graph.items():
            sum_weights[sent] = sum(adj.values())

        self.scores = [1 / n] * n  # initialize pagerank scores

        for iter in range(lexrank_iter):
            if iter % 10 == 0:
                print("Iteration: {}".format(iter))
            for sent, adjs in sent.graph.items():
                score = 0
                for adj, value in adjs.items():
                    score += self.scores[adj] * value / sum_weights[adj]
                self.scores[sent] = (1 - damping_factor) + damping_factor * score

    def extract_summary(self, n_sents=10, cosine_thres=0.5, max_sent=100):

        sentIds = []
        sentScores = np.array(self.scores.copy())

        print("Extracting sentences....")
        # get #max_sent maximal scores along with its indices
        indices = np.argpartition(sentScores, -max_sent)[-max_sent, :]
        values = sentScores[indices]
        max_index_value = {key: value for key, value in zip(indices, values)}
        max_index_value = sorted(max_index_value.items(), key=lambda x: (x[1], x[0]), reverse=True)

        i = 0
        while i < n_sents:
            index, value = max_index_value.pop()
            assign = 1
            for idx in sentIds:
                if index not in self.graph:
                    print("Sent {} not in graph".format(index))
                    assign = 0
                    break
                if idx not in self.graph[index]:
                    continue
                similarity = self.graph[index][idx]
                if similarity > cosine_thres:
                    print("Sent {} is similar to a {}".format(index, idx))
                    assign = 0
                    break
            if assign == 1:
                print(i, ", ", 'TweetId: ', self.data.iloc[index]['Id'], ": ", self.data.iloc[index]['Tweet'])
                sentIds.append(index)
                i += 1
        return sentIds
