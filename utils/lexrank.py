import numpy as np
from numpy.linalg import norm
import sys
class Lexrank:
    def __init__(self, data):
        self.data = data
        self.graph = {}
        self.scores = None 
    
    def build_graph_bertscore_from_file(self, sim_thres, input_file):
        count = 0
        with open(input_file, 'r') as f:
            for i, line in enumerate(f):
                
                content = line.split(",")
                idx = int(content[0])
                if content[1][1:-1]=="":
                    continue
                neighbors = [int(x) for x in content[1][1:-1].split(' ')]
                sim_scores = [float(y) for y in content[2][1:-2].split(" ") if y!='']
                for j, neighbor_idx in enumerate(neighbors):
                    if sim_scores[j] <sim_thres:
                        continue

                    if idx not in self.graph:
                        self.graph[idx] = {}
                    self.graph[idx][neighbor_idx] = sim_scores[j]
                    if neighbor_idx not in self.graph:
                        self.graph[neighbor_idx] = {}
                    self.graph[neighbor_idx][idx] = sim_scores[j]
                if i%200 == 0:
                    print("Line: ", i, idx)
        print("Done, ", i, count)
    def build_graph_cosine(self, cos_thres = 0.3, batch_size = 1000):
        for i in range(0, self.data.shape[0], batch_size):
            if i+batch_size > self.data.shape[0]:
                current_sents = self.data[i:self.data.shape[0]]
            else:
                current_sents = self.data[i: i+batch_size]
            
            current_magnitudes = norm(current_sents.toarray(), axis=1)
            
            for j in range(i, self.data.shape[0], batch_size):
                rightBound = j+batch_size
                if j+batch_size> self.data.shape[0]:
                    rightBound = self.data.shape[0]

                sents = self.data[j:rightBound]

                num = np.dot(current_sents, sents.T)
                magnitudes = norm(sents.toarray(), axis = 1)
                denum = np.dot(current_magnitudes.reshape(-1, 1), magnitudes.T.reshape(1, -1))

                cosine_matrix = np.array(num/denum)
                indices = np.where(cosine_matrix>cos_thres)

                if len(indices[0]) == 0:
                    continue
                for row, col in zip(indices[0], indices[1]):
                    if i+row != j+col:
                        if i+row not in self.graph:
                            self.graph[i+row] = {}
                        if j+col not in self.graph:
                            self.graph[j+col] = {}
                        self.graph[i+row][j+col] = cosine_matrix[row][col]
                        self.graph[j+col][i+row] = cosine_matrix[row][col]
    
    def train(self, lexrank_iter = 100, damping_factor=0.85):
        n = self.data.shape[0]
        sum_weights = {}
        for sent, adjs in self.graph.items():
            sum_weights[sent] = sum(adjs.values())
        self.scores = [1/n] * n

        for iter in range(lexrank_iter):
            if iter%10 == 0:
                print("Iteration: {}".format(iter))
            for sent, adjs in self.graph.items():
                score = 0
                for adj, value in adjs.items():
                    score += self.scores[adj] * value / sum_weights[adj]
                self.scores[sent] = (1 - damping_factor)/n + damping_factor * score
    
    def extract_summary(self, n_sents = 10, cos_thres = 0.85, max_sent = 100, min_len = -1, data = None):
        sentIds = []
        sentScores = np.array(self.scores.copy())
        print("Extracting sentences....")
        indices = np.argpartition(sentScores, -max_sent)[-max_sent:]
        values = sentScores[indices]
        max_index_value = {key: value for key, value in zip(indices, values)}
        max_index_value = sorted(max_index_value.items(), key = lambda x: (x[1], x[0]))

        i = 0
        while i < n_sents:
            index, value = max_index_value.pop()
            if index not in self.graph:
                continue
            if (min_len!=-1):
                if (data.iloc[index]['uniWPercent'] <=min_len):
                    continue
            assign = 1 
            for idx in sentIds:
                if idx not in self.graph[index]: # add to summary
                    continue
                sim = self.graph[index][idx]
                if sim > cos_thres: # compare similarity with selected ones
                    assign = 0
                    break
            if assign == 1:
                sentIds.append(index)
                i+=1 
        return sentIds