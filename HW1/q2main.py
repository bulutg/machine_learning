import pandas as pd
import numpy as np
import time

features_attr = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                 'DiabetesPedigreeFunction',
                 'Age']

labels_attr = ['Outcome']

train_features_df = pd.read_csv('q2/diabetes_train_features.csv', names=features_attr, skiprows=1)
train_labels_df = pd.read_csv('q2/diabetes_train_labels.csv', names=labels_attr, skiprows=1)
test_features_df = pd.read_csv('q2/diabetes_test_features.csv', names=features_attr, skiprows=1)
test_labels_df = pd.read_csv('q2/diabetes_test_labels.csv', names=labels_attr, skiprows=1)

train_f = train_features_df.to_numpy()
train_l = train_labels_df.to_numpy()
test_f = test_features_df.to_numpy()
test_l = test_labels_df.to_numpy()


def backward_elimination(train_features, train_labels,
                         test_features, test_labels,
                         attr_list, accuracy=-1):
    results = []
    column_count = train_features.shape[1]

    start = time.time()

    new_accuracy = predict(train_features, train_labels,
                           test_features, test_labels)[1]

    if accuracy <= new_accuracy:
        accuracy = new_accuracy

    for i in range(column_count):
        result = predict(np.delete(train_features, i, 1), train_labels,
                         np.delete(test_features, i, 1), test_labels)
        results.append([i, result[1], result[2], result[3]])

    results.sort(key=lambda x: x[1], reverse=True)

    if results[0][1] >= accuracy:
        validation_time = time.time() - start

        feature_id = results[0][0]
        print("Del", attr_list[feature_id],
              "|", results[0][1], ">", accuracy,
              "| f_no:", column_count - 1)
        print(
              "| T_prediction:", results[0][2],
              "| T_validation:", validation_time,
              "| c_matrix:", results[0][3])
        attr_list.pop(feature_id)
        backward_elimination(np.delete(train_features, results[0][0], 1), train_labels,
                             np.delete(test_features, results[0][0], 1), test_labels, attr_list)


def predict(train_features, train_labels,
            test_features, test_labels):
    confusion_matrix = [[0, 0], [0, 0]]
    prediction = []
    acc_count = 0

    start = time.time()
    for item in test_features:
        distances = np.linalg.norm(train_features - item, axis=1).argsort()
        first9 = distances[:9]
        result = 0
        for index in first9:
            result += train_labels[index]
        if result >= 5:
            prediction.append(1)
        else:
            prediction.append(0)

    for i, j in zip(prediction, test_labels):
        if i == j:
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

    prediction_time = time.time() - start

    return [prediction, accuracy, prediction_time, confusion_matrix]


backward_elimination(train_f, train_l, test_f, test_l, features_attr)
