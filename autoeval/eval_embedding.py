import numpy as np


class Embed:
    def __init__(self, embed_path):
        file = open(embed_path, encoding='utf-8')
        embed = [line[:-1].split(' ') for line in file.readlines()]
        embed = [(line[0], line[1:]) for line in embed]
        d = {}
        for line in embed:
            d[line[0]] = np.array(line[1])
        vec = d['a']
        d['<unk>'] = np.zeros([vec.shape[0]])
        self.embed_dict = d

    def unk_transfer(self, word):
        if word in self.embed_dict:
            return word
        else:
            return '<unk>'

    def eval_embedding(self, references, candidates):
        score_total = 0
        for i in range(len(references)):
            candidate = [self.unk_transfer(word) for word in candidates[i]]
            candidate = np.array([self.embed_dict[word] for word in candidate])
            candidate = candidate.astype(np.float32)
            avg_can = np.average(candidate, axis=0)
            reference = [self.unk_transfer(word) for word in references[i][0]]
            reference = np.array([self.embed_dict[word] for word in reference])
            reference = reference.astype(np.float32)
            avg_ref = np.average(reference, axis=0)
            dist = np.sum(np.square(avg_ref - avg_can)) / len(avg_can)
            score_total += dist
        return score_total / len(references)
