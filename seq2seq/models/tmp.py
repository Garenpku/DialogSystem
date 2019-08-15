dict = {}

f = open(path).readlines()

name = []

def find(name, res):


for line in f:
    res = line.split(' ')
    if res[0] + res[1] not in name:
        name.append([res[0] + res[1]])
        name[-1].append()


