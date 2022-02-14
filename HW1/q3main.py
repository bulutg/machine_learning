import pandas as pd
import numpy as np
from math import log2, inf
import sys
import time

train_f = pd.read_csv('q3/sms_train_features.csv').to_numpy()
train_l = pd.read_csv('q3/sms_train_labels.csv').to_numpy()
test_f = pd.read_csv('q3/sms_test_features.csv').to_numpy()
test_l = pd.read_csv('q3/sms_test_labels.csv').to_numpy()


def train(train_features, train_labels, binomial=False):

    start_training_time = time.time()
    simple_features = np.delete(train_features, 0, 1)

    spam = np.zeros(simple_features[0].shape)
    ham = np.zeros(simple_features[0].shape)

    p_spam = np.zeros(simple_features[0].shape)
    p_ham = np.zeros(simple_features[0].shape)

    spam_count = 0
    ham_count = 0
    spam_ham_count = 0

    if not binomial:
        for i, j in zip(simple_features, train_labels):
            # sum spam ham row by row
            if j[1] == 1:
                spam = np.add(i, spam)
                spam_count += 1
            else:
                ham = np.add(i, ham)
                ham_count += 1
            spam_ham_count += 1

        # calculate count(y)
        sum_spam = np.sum(spam)
        sum_ham = np.sum(ham)

        # feature count
        count = len(p_spam)

        # marginal
        m_spam = spam_count / spam_ham_count
        m_ham = ham_count / spam_ham_count

        for i in range(count):
            p_spam[i] = (spam[i] + 1) / (count + sum_spam)
            p_ham[i] = (ham[i] + 1) / (count + sum_ham)

        #return [p_spam, p_ham, m_spam, m_ham, simple_features]
    # bernoulli
    else:
        # make array binary
        simple_features = np.where(simple_features > 0, 1, 0)
        for i, j in zip(simple_features, train_labels):
            if j[1] == 1:
                spam = np.add(i, spam)
                spam_count += 1
            else:
                ham = np.add(i, ham)
                ham_count += 1
            spam_ham_count += 1

        # calculate count(y)
        # sum_spam = np.sum(spam)
        # sum_ham = np.sum(ham)

        # feature count
        count = len(p_spam)

        # marginal
        m_spam = spam_count / spam_ham_count
        m_ham = ham_count / spam_ham_count

        for i in range(count):
            p_spam[i] = (spam[i] + 1) / (2 + spam_count)
            p_ham[i] = (ham[i] + 1) / (2 + ham_count)

    train_time = time.time() - start_training_time
    return [p_spam, p_ham, m_spam, m_ham, simple_features, train_time]


# predict multinomial
def predict(test_features, test_labels, spam_probability_arr, ham_probability_arr, marginal_spam, marginal_ham):
    spam_prediction = []
    ham_prediction = []
    prediction_result = []
    confusion_matrix = [[0, 0], [0, 0]]
    acc_count = 0

    for row, result in zip(np.delete(test_features, 0, 1), np.delete(test_labels, 0, 1)):
        prob_spam = marginal_spam
        prob_ham = marginal_ham
        for z in range(len(row)):
            prob_spam *= spam_probability_arr[z] ** row[z]
            prob_ham *= ham_probability_arr[z] ** row[z]
        spam_prediction.append(prob_spam)
        ham_prediction.append(prob_ham)
        if prob_spam > prob_ham:
            prediction_result.append(1)
        else:
            prediction_result.append(0)

    # print(prediction_result)
    # print(spam_prediction)
    # print(ham_prediction)

    for i, j in zip(prediction_result, np.delete(test_labels, 0, 1)):
        if (i == j).all():
            acc_count += 1
            if i == 1:
                confusion_matrix[0][0] += 1
            else:
                confusion_matrix[1][1] += 1
        else:
            if i == 1:
                confusion_matrix[1][0] += 1
            else:
                confusion_matrix[0][1] += 1

    accuracy = acc_count / len(test_labels)
    # print(accuracy)
    # print(confusion_matrix)
    return [accuracy, confusion_matrix]


# predict bernoulli
def predict_bi(test_features, test_labels, spam_probability_arr, ham_probability_arr, marginal_spam, marginal_ham):
    spam_prediction = []
    ham_prediction = []
    prediction_result = []
    confusion_matrix = [[0, 0], [0, 0]]
    acc_count = 0

    for row, result in zip(np.delete(test_features, 0, 1), np.delete(test_labels, 0, 1)):
        prob_spam = marginal_spam
        prob_ham = marginal_ham
        for z in range(len(row)):
            if row[z]:
                prob_spam *= spam_probability_arr[z]
                prob_ham *= ham_probability_arr[z]
            else:
                prob_spam *= (1 - spam_probability_arr[z])
                prob_ham *= (1 - ham_probability_arr[z])
        spam_prediction.append(prob_spam)
        ham_prediction.append(prob_ham)
        if prob_spam > prob_ham:
            prediction_result.append(1)
        else:
            prediction_result.append(0)

    # print(prediction_result)
    # print(spam_prediction)
    # print(ham_prediction)

    for i, j in zip(prediction_result, np.delete(test_labels, 0, 1)):
        if (i == j).all():
            acc_count += 1
            if i == 1:
                confusion_matrix[0][0] += 1
            else:
                confusion_matrix[1][1] += 1
        else:
            if i == 1:
                confusion_matrix[1][0] += 1
            else:
                confusion_matrix[0][1] += 1

    accuracy = acc_count / len(test_labels)
    # print(accuracy)
    # print(confusion_matrix)
    return [accuracy, confusion_matrix]


# calculta mutual information
def compute_mi(feature_col, label_col):
    n_00 = 0
    n_01 = 0
    n_10 = 0
    n_11 = 0

    for i, j in zip(feature_col, label_col):
        if i == 0 and j == 0:
            n_00 += 1
        elif i == 0 and j == 1:
            n_01 += 1
        elif i == 1 and j == 0:
            n_10 += 1
        elif i == 1 and j == 1:
            n_11 += 1

    n_0 = n_00 + n_01
    n_1 = n_10 + n_11
    n__0 = n_00 + n_10
    n__1 = n_01 + n_11
    var_n = n_0 + n_1

    val = 0

    try:
        try:
            val += (n_11 / var_n) * log2((var_n * n_11) / (n_1 * n__1))
        except ValueError:
            if n_11 == 0:
                pass
            else:
                return -inf
        try:
            val += (n_01 / var_n) * log2((var_n * n_01) / (n_0 * n__1))
        except ValueError:
            if n_01 == 0:
                pass
            else:
                return -inf
        try:
            val += (n_10 / var_n) * log2((var_n * n_10) / (n_1 * n__0))
        except ValueError:
            if n_10 == 0:
                pass
            else:
                return -inf
        try:
            val += (n_00 / var_n) * log2((var_n * n_00) / (n_0 * n__0))
        except ValueError:
            if n_00 == 0:
                pass
            else:
                return -inf
    except ZeroDivisionError:
        return -inf

    return val


# get best n features 2d numpy array according to sorted feature id list
def get_n_features(features_arr, feature_id_list, number):
    # get row number
    # n_features = np.zeros(train_features.shape[0])

    # print(n_features)
    # print(train_features[:, feature_id_list[541]])

    # for i in range(number):
    #     n_features = np.concatenate([n_features, train_features[:, feature_id_list[i]]], axis=1)
    #     print(n_features)

    return features_arr[:, feature_id_list[:number]]


prob = train(train_f, train_l)
re = predict(test_f, test_l, prob[0], prob[1], prob[2], prob[3])
print("multinomial accuracy: ", re[0], re[1])

mi_results = []

for k in range(train_f.shape[1]):
    mi_results.append([k, compute_mi(train_f[:, k], train_l[:, 1])])

mi_results.sort(key=lambda x: x[1], reverse=True)
# print(mi_results[:100])

transpose_mi_results = list(zip(*mi_results[1:]))
best_features_id_list = transpose_mi_results[0]

# print(best_features_id_list[:100])

prob = train(train_f, train_l, binomial=True)
re = predict_bi(test_f, test_l, prob[0], prob[1], prob[2], prob[3])
print("bernoulli accuracy:", re[0], re[1], "training time:", prob[5])


for n in range(100, 700, 100):
    best_train_features = get_n_features(train_f, best_features_id_list, n)
    best_test_features = get_n_features(test_f, best_features_id_list, n)
    prob = train(best_train_features, train_l, binomial=True)
    re = predict_bi(best_test_features, test_l, prob[0], prob[1], prob[2], prob[3])
    print(n, re[0], re[1], "training time:", prob[5])
