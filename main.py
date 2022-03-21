#!/usr/bin/env python3

# solution for fmlsp22 hw2 problem c
# author: Jinli Xiao

import os

# parse data

f = open("abalone.data", "r")
f1 = open("output/abalone.txt", "w")
f2 = open("output/abalone.t.txt", "w")

cnt = 0
for line in f:
    out = f1 if (cnt < 3133) else f2
    lst = line.split(",")
    parsed = ["-1" if (int(lst[-1]) <= 9) else "1"]
    parsed.append({"M": "1:1 2:0 3:0", "F": "1:0 2:1 3:0", "I": "1:0 2:0 3:1"}[lst[0]])
    for i in range(1, len(lst)-1):
        parsed.append("%d:%s" % (i+3, lst[i]))
    out.write(" ".join(parsed))
    out.write("\n")
    cnt += 1

f.close()
f1.close()
f2.close()


# scale data

os.system("./libsvm-3.25/svm-scale -s output/range.txt output/abalone.txt > output/abalone.scale.txt")
os.system("./libsvm-3.25/svm-scale -r output/range.txt output/abalone.t.txt > output/abalone.t.scale.txt")


# cross validation
os.system("shuf output/abalone.scale.txt > output/abalone.scale.shuffle.txt")
os.system("split -n l/5 -d --additional-suffix=.txt output/abalone.scale.shuffle.txt output/abalone.scale.shuffle.split")


k = 0
for d in range(1, 5):
    for i in range(-k, k+1):
        for s in range(5):
            c = 3 ** i
            train_set = "output/abalone.scale.shuffle.split0%d.txt" % s
            model_file = "output/abalone.model.%d.%d.%d.txt" % (d, i, s)
            train_log = "output/abalone.train.log.%d.%d.%d.txt" % (d, i, s)
            test_file = "output/abalone.t.scale.txt"
            prediction = "output/abalone.prediction.%d.%d.%d.txt" % (d, i, s)
            pred_log = "output/abalone.prediction.log.%d.%d.%d.txt" % (d, i, s)
            os.system("./libsvm-3.25/svm-train -t 1 -d %d -c %f %s %s > %s" % (d, c, train_set, model_file, train_log))
            os.system("./libsvm-3.25/svm-predict %s %s %s > %s" % (test_file, model_file, prediction, pred_log))

            
