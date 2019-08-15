import numpy


def judge_same(ngram1, ngram2):
    for i in range(len(ngram1)):
        if ngram1[i] != ngram2[i]:
            return False
    return True


def distinct(candidates, n):
    score_all = 0
    for line in candidates:
        ngram = []
        for i in range(len(line) - n + 1):
            ngram.append(line[i:i+n])
        cnt = 0
        for i in range(len(ngram)):
            for j in range(i+1, len(ngram)):
                if judge_same(ngram[i], ngram[j]):
                    cnt += 1
                    break
        score_all += (len(ngram) - cnt) / len(line)
    return score_all / len(candidates)
