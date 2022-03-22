#!/usr/bin/env python3

# solution for fmlsp22 hw2 problem c
# author: Jinli Xiao

import os
import re
import numpy as np
import matplotlib.pyplot as plt

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


# generate training set and testing set for cross validation
os.system("shuf output/abalone.scale.txt > output/abalone.scale.shuffle.txt")
os.system("split -n l/5 -d --additional-suffix=.txt output/abalone.scale.shuffle.txt output/abalone.scale.shuffle.split")

os.system("cat output/abalone.scale.shuffle.split0{1,2,3,4}.txt > output/abalone.train00.txt")
os.system("cat output/abalone.scale.shuffle.split0{0,2,3,4}.txt > output/abalone.train01.txt")
os.system("cat output/abalone.scale.shuffle.split0{0,1,3,4}.txt > output/abalone.train02.txt")
os.system("cat output/abalone.scale.shuffle.split0{0,1,2,4}.txt > output/abalone.train03.txt")
os.system("cat output/abalone.scale.shuffle.split0{0,1,2,3}.txt > output/abalone.train04.txt")

os.system("cat output/abalone.scale.shuffle.split00.txt > output/abalone.vad00.txt")
os.system("cat output/abalone.scale.shuffle.split01.txt > output/abalone.vad01.txt")
os.system("cat output/abalone.scale.shuffle.split02.txt > output/abalone.vad02.txt")
os.system("cat output/abalone.scale.shuffle.split03.txt > output/abalone.vad03.txt")
os.system("cat output/abalone.scale.shuffle.split04.txt > output/abalone.vad04.txt")


# conduct cross validation
k = 9
for d in range(1, 6):
    for i in range(-k, k+1):
        for s in range(5):
            c = 3 ** i
            train_set = "output/abalone.train0%d.txt" % s
            model_file = "output/abalone.model.%d.%d.%d.txt" % (d, i, s)
            train_log = "output/abalone.train.log.%d.%d.%d.txt" % (d, i, s)
            test_file = "output/abalone.vad0%d.txt" % s
            prediction = "output/abalone.prediction.%d.%d.%d.txt" % (d, i, s)
            pred_log = "output/abalone.prediction.log.%d.%d.%d.txt" % (d, i, s)
            os.system("./libsvm-3.25/svm-train -t 1 -d %d -c %f %s %s > %s" % (d, c, train_set, model_file, train_log))
            os.system("./libsvm-3.25/svm-predict %s %s %s > %s" % (test_file, model_file, prediction, pred_log))


# get cross-validation accuracy
result = []
for d in range(1, 6):
    result_d = []
    for i in range(-k, k+1):
        arr = []
        for s in range(5):
            pred_log = "output/abalone.prediction.log.%d.%d.%d.txt" % (d, i, s)
            with open(pred_log, "r") as f:
                for line in f:
                    res = re.findall(r"^Accuracy = ([\d\.]+)%", line)
                    if res:
                        arr.append(1-float(res[0])/100)
        result_d.append(arr)
    result.append(result_d)
result = np.array(result)


# plot cross validation accuracy
fig, axs = plt.subplots(2, 3, sharex=False, sharey=True)
fig.set_size_inches(18, 10)
fig.set_dpi(350)

for d in range(1, 6):
    axs[(d-1)//3, (d-1)%3].set_title("d=%d" % d)

    cs = np.array(range(-k, k+1))
    avgs = np.array([np.mean(result[d-1][i]) for i in range(2*k+1)])
    stds = np.array([np.std(result[d-1][i]) for i in range(2*k+1)])

    axs[(d-1)//3, (d-1)%3].plot(cs, avgs)
    axs[(d-1)//3, (d-1)%3].plot(cs, avgs+stds, '--')
    axs[(d-1)//3, (d-1)%3].plot(cs, avgs-stds, '--')
    axs[(d-1)//3, (d-1)%3].set_xticks(np.arange(-k, k+1, step=2))
    if d in [3, 4, 5]:
        axs[(d-1)//3, (d-1)%3].set_xlabel("log C")
    if d in [1, 4]:
        axs[(d-1)//3, (d-1)%3].set_ylabel("Cross Validation Error")
        
axs[1, 2].set_axis_off()
fig.savefig("question 3.pdf")


# compute test errors for different training sets
k = 5
test_errors = []
for d in range(1, 6):
    arr = []
    for s in range(5):
        model_file = "output/abalone.model.%d.%d.%d.txt" % (d, k, s)
        test_file = "output/abalone.t.scale.txt"
        prediction = "output/abalone.t.prediction.%d.%d.%d.txt" % (d, k, s)
        pred_log = "output/abalone.t.prediction.log.%d.%d.%d.txt" % (d, k, s)
        os.system("./libsvm-3.25/svm-predict %s %s %s > %s" % (test_file, model_file, prediction, pred_log))
        with open(pred_log, "r") as f:
            for line in f:
                res = re.findall(r"^Accuracy = ([\d\.]+)%", line)
                if res:
                    arr.append(1-float(res[0])/100)
    test_errors.append(arr)
test_errors = np.array(test_errors)


# get nSV and nBSV
k = 5
nSV = []
nBSV = []
for d in range(1, 6):
    arr1 = []
    arr2 = []
    for s in range(5):
        train_log = "output/abalone.train.log.%d.%d.%d.txt" % (d, k, s)
        with open(train_log, "r") as f:
            for line in f:
                res = re.findall(r"^nSV = (\d+), nBSV = (\d+)$", line)
                if res:
                    arr1.append(int(res[0][0]))
                    arr2.append(int(res[0][1]))
    nSV.append(arr1)
    nBSV.append(arr2)
nSV = np.array(nSV)
nBSV = np.array(nBSV)


fig, axs = plt.subplots(2, 2, sharex=False, sharey=False)
fig.set_size_inches(18, 10)
fig.set_dpi(350)

# plot for 5-Fold Cross Validation Error
i = 14  # the best c is 3^5 = 243
axs[0, 0].set_ylabel("5-Fold Cross Validation Error")
axs[0, 0].set_xlabel("d")
xs = np.array([1, 2, 3, 4, 5])
avgs = np.array([np.mean(result[d-1][i]) for d in range(1, 6)])
stds = np.array([np.std(result[d-1][i]) for d in range(1, 6)])
axs[0, 0].plot(xs, avgs)
axs[0, 0].plot(xs, avgs + stds, '--')
axs[0, 0].plot(xs, avgs - stds, '--')
axs[0, 0].set_xticks([1, 2, 3, 4, 5])

# plot for Test Error
axs[0, 1].set_ylabel("Test Error")
axs[0, 1].set_xlabel("d")
xs = np.array([1, 2, 3, 4, 5])
avgs = np.array([np.mean(test_errors[d-1]) for d in range(1, 6)])
stds = np.array([np.std(test_errors[d-1]) for d in range(1, 6)])
axs[0, 1].plot(xs, avgs)
axs[0, 1].plot(xs, avgs + stds, '--')
axs[0, 1].plot(xs, avgs - stds, '--')
axs[0, 1].set_xticks([1, 2, 3, 4, 5])

# plot for Number of Support Vectors
axs[1, 0].set_ylabel("Number of Support Vectors")
axs[1, 0].set_xlabel("d")
xs = np.array([1, 2, 3, 4, 5])
avgs = np.array([np.mean(nSV[d-1]) for d in range(1, 6)])
axs[1, 0].plot(xs, avgs)
axs[1, 0].set_xticks([1, 2, 3, 4, 5])

axs[1, 1].set_axis_off()
fig.savefig("question 4.pdf")


# compute training error for each set
k = 5
d = 2

train_errors = []
for s in range(5):
    model_file = "output/abalone.model.%d.%d.%d.txt" % (d, k, s)
    test_file = "output/abalone.train0%d.txt" % s
    prediction = "output/abalone.tr.prediction.%d.%d.%d.txt" % (d, k, s)
    pred_log = "output/abalone.tr.prediction.log.%d.%d.%d.txt" % (d, k, s)
    os.system("./libsvm-3.25/svm-predict %s %s %s > %s" % (test_file, model_file, prediction, pred_log))
    
    with open(pred_log, "r") as f:
        for line in f:
            res = re.findall(r"^Accuracy = ([\d\.]+)%", line)
            if res:
                train_errors.append(1-float(res[0])/100)


# plot training and test errors
fig, ax = plt.subplots()

d = 2
ax.set_ylabel("Error")
ax.set_xlabel("Training Sample No.")
xs = np.array([1, 2, 3, 4, 5])
ax.plot(xs, test_errors[d-1], 'o-', label='test error')
ax.plot(xs, train_errors, 'o-', label='train error')
ax.set_xticks([1, 2, 3, 4, 5])
ax.legend()

fig.savefig("question 5.pdf")

            
