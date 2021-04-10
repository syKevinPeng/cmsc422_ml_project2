from sklearn.tree import DecisionTreeClassifier
import multiclass
import util
from datasets import *

# h = multiclass.OAA(20, lambda: DecisionTreeClassifier(max_depth=1))
# h.train(WineData.X, WineData.Y)
# P = h.predictAll(WineData.Xte)
# print(mean(P == WineData.Yte))
# print(mode(WineData.Y))
# print(WineData.labels[1])
# print(mean(WineData.Yte == 1))
#
# print()
# # use zero and one
# P = h.predictAll(WineData.Xte, useZeroOne=True)
# print(mean(P == WineData.Yte))
#
# print()
# # swithcing to smaller data set
# h = multiclass.OAA(5, lambda: DecisionTreeClassifier(max_depth=3))
# h.train(WineDataSmall.X, WineDataSmall.Y)
# P = h.predictAll(WineDataSmall.Xte)
# print(mean(P == WineDataSmall.Yte))
# print(mean(WineDataSmall.Yte == 1))
# print(WineDataSmall.labels[0])
# util.showTree(h.f[0], WineDataSmall.words)

# h = multiclass.AVA(5, lambda: DecisionTreeClassifier(max_depth=3))
# h.train(WineDataSmall.X, WineDataSmall.Y)
# P = h.predictAll(WineDataSmall.Xte)
# print(mean(P == WineDataSmall.Yte))
# print(mean(WineDataSmall.Yte == 1))
# print(WineDataSmall.labels[0])
# util.showTree(h.f[1][0], WineDataSmall.words)

t = multiclass.makeBalancedTree(range(5))
h = multiclass.MCTree(t, lambda: DecisionTreeClassifier(max_depth=3))
h.train(WineDataSmall.X, WineDataSmall.Y)
P = h.predictAll(WineDataSmall.Xte)
print(mean(P == WineDataSmall.Yte))

