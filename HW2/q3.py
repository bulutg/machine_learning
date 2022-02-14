import pandas as pd
import numpy as np
import random


def f_beta_score(beta, precision, recall):
    return (beta * beta * (precision + recall)) and ((1 + beta * beta) * precision * recall) / (
            beta * beta * (precision + recall)) or 0


def min_max_normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr)), np.min(arr), np.max(arr)


def min_max_normalize_test(arr, minimum, maximum):
    return (arr - minimum) / (maximum - minimum)


def my_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    features_train = np.array(pd.read_csv('question-3-features-train.csv'))
    labels_train = np.array(pd.read_csv('question-3-labels-train.csv'))
    features_test = np.array(pd.read_csv('question-3-features-test.csv'))
    labels_test = np.array(pd.read_csv('question-3-labels-test.csv'))

    features_train_norm, my_min, my_max = min_max_normalize(features_train)

    mean = features_train_norm.mean(axis=0)
    std = features_train_norm.std(axis=0)

    features_train_norm = (features_train_norm - mean) / std

    features_test_norm = min_max_normalize_test(features_test, my_min, my_max)
    features_test_norm = (features_test_norm - mean) / std

    features_train_ones = np.column_stack((np.ones((features_train_norm.shape[0], 1)), features_train_norm))
    features_test_ones = np.column_stack((np.ones((features_test_norm.shape[0], 1)), features_test_norm))

    for n in [0.00001, 0.0001, 0.001, 0.01, 0.1]:
        weight = np.zeros((4, 1))
        acc_count = 0
        confusion_matrix = [[0, 0], [0, 0]]

        for i in range(1000):
            weight += (n * (features_train_ones.T @ (labels_train - my_sigmoid(features_train_ones @ weight))))
        # print(n, weight)

        prediction = my_sigmoid(features_test_ones @ weight)

        for i, j in zip(prediction, labels_test):
            pred = 1 if i >= 0.5 else 0
            if pred == j:
                acc_count += 1
                if pred == 1:
                    confusion_matrix[0][0] += 1
                else:
                    confusion_matrix[1][1] += 1
            else:
                if pred == 1:
                    confusion_matrix[1][0] += 1
                else:
                    confusion_matrix[0][1] += 1

        accuracy = acc_count / len(labels_test)
        print("learning rate:", n, "| accuracy:", accuracy)
        print("confusion matrix:", confusion_matrix)

        true_pos = confusion_matrix[0][0]
        false_pos = confusion_matrix[0][1]
        false_neg = confusion_matrix[1][0]
        true_neg = confusion_matrix[1][1]

        precision = true_pos / (true_pos + false_pos)
        recall = (true_pos + false_neg) and (true_pos / (true_pos + false_neg)) or 0
        npv = (true_neg + false_neg) and (true_neg / (true_neg + false_neg)) or 0
        fpr = (true_neg + false_pos) and (false_pos / (true_neg + false_pos)) or 0
        fdr = false_pos / (false_pos + true_pos)
        f1_score = f_beta_score(1, precision, recall)
        f2_score = f_beta_score(2, precision, recall)

        print("precision: ", precision, " recall:", recall)
        print("npv: ", npv, "fpr:", fpr, "fdr:", fdr)
        print("f1:", f1_score, "f2:", f2_score)
        print("------------------------------------------")

    print("mini batch n = 100")
    selected_learning_rate = 0.001
    # random weights
    batch_size = 100
    new_weights = np.random.normal(0, 0.01, size=(4, 1))
    # mini batch gradient ascent
    for i in range(1000):
        indices = random.sample(range(0, 712), batch_size)
        new_train_array = features_train_ones[indices]
        new_label_array = labels_train[indices]
        new_weights += (selected_learning_rate * (new_train_array.T @ (new_label_array - my_sigmoid(new_train_array @ new_weights))))

    prediction = my_sigmoid(features_test_ones @ new_weights)

    acc_count = 0
    confusion_matrix = [[0, 0], [0, 0]]

    for i, j in zip(prediction, labels_test):
        pred = 1 if i >= 0.5 else 0
        if pred == j:
            acc_count += 1
            if pred == 1:
                confusion_matrix[0][0] += 1
            else:
                confusion_matrix[1][1] += 1
        else:
            if pred == 1:
                confusion_matrix[1][0] += 1
            else:
                confusion_matrix[0][1] += 1

    accuracy = acc_count / len(labels_test)
    print("learning rate:", selected_learning_rate, "| accuracy:", accuracy)
    print("confusion matrix:", confusion_matrix)

    true_pos = confusion_matrix[0][0]
    false_pos = confusion_matrix[0][1]
    false_neg = confusion_matrix[1][0]
    true_neg = confusion_matrix[1][1]

    precision = true_pos / (true_pos + false_pos)
    recall = (true_pos + false_neg) and (true_pos / (true_pos + false_neg)) or 0
    npv = (true_neg + false_neg) and (true_neg / (true_neg + false_neg)) or 0
    fpr = (true_neg + false_pos) and (false_pos / (true_neg + false_pos)) or 0
    fdr = false_pos / (false_pos + true_pos)
    f1_score = f_beta_score(1, precision, recall)
    f2_score = f_beta_score(2, precision, recall)

    print("precision: ", precision, " recall:", recall)
    print("npv: ", npv, "fpr:", fpr, "fdr:", fdr)
    print("f1:", f1_score, "f2:", f2_score)
    print("------------------------------------------")

    # stochastic gradient ascent algorithm
    print("stochastic gradient ascent algorithm")

    # random weights
    batch_size = 1
    new_weights = np.random.normal(0, 0.01, size=(4, 1))
    # mini batch gradient ascent
    for i in range(1000):
        indices = random.sample(range(0, 712), batch_size)
        new_train_array = features_train_ones[indices]
        new_label_array = labels_train[indices]
        new_weights += (selected_learning_rate * (
                    new_train_array.T @ (new_label_array - my_sigmoid(new_train_array @ new_weights))))

    prediction = my_sigmoid(features_test_ones @ new_weights)

    acc_count = 0
    confusion_matrix = [[0, 0], [0, 0]]

    for i, j in zip(prediction, labels_test):
        pred = 1 if i >= 0.5 else 0
        if pred == j:
            acc_count += 1
            if pred == 1:
                confusion_matrix[0][0] += 1
            else:
                confusion_matrix[1][1] += 1
        else:
            if pred == 1:
                confusion_matrix[1][0] += 1
            else:
                confusion_matrix[0][1] += 1

    accuracy = acc_count / len(labels_test)
    print("learning rate:", selected_learning_rate, "| accuracy:", accuracy)
    print("confusion matrix:", confusion_matrix)

    true_pos = confusion_matrix[0][0]
    false_pos = confusion_matrix[0][1]
    false_neg = confusion_matrix[1][0]
    true_neg = confusion_matrix[1][1]

    precision = true_pos / (true_pos + false_pos)
    recall = (true_pos + false_neg) and (true_pos / (true_pos + false_neg)) or 0
    npv = (true_neg + false_neg) and (true_neg / (true_neg + false_neg)) or 0
    fpr = (true_neg + false_pos) and (false_pos / (true_neg + false_pos)) or 0
    fdr = false_pos / (false_pos + true_pos)
    f1_score = f_beta_score(1, precision, recall)
    f2_score = f_beta_score(2, precision, recall)

    print("precision: ", precision, " recall:", recall)
    print("npv: ", npv, "fpr:", fpr, "fdr:", fdr)
    print("f1:", f1_score, "f2:", f2_score)
    print("------------------------------------------")


if __name__ == '__main__':
    main()
