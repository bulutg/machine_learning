import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    features_arr = np.array(pd.read_csv('question-2-features.csv'))
    labels_arr = np.array(pd.read_csv('question-2-labels.csv'))

    # q2.2
    rank_xtx = np.linalg.matrix_rank(np.matmul(features_arr.transpose(), features_arr))
    print("rank is", rank_xtx)

    lstat = np.array(features_arr[:, 12])
    x_arr = np.column_stack((np.ones(lstat.shape), lstat))

    # q2.3
    optimal_params = np.matmul(np.matmul(np.linalg.inv(np.matmul(x_arr.transpose(), x_arr)), x_arr.transpose()),
                               labels_arr)
    print("Bias and LSTAT", optimal_params)

    prediction = np.matmul(x_arr, optimal_params)
    # print("prediction", prediction)

    plt.xlabel("LSTAT")
    plt.ylabel("PRICE")
    plt.plot(lstat, labels_arr, 'r.')
    plt.plot(lstat, prediction, 'g.')
    plt.show()

    mse = np.square(np.subtract(labels_arr, prediction)).mean()

    print("MSE", mse)

    # q2.4
    poly_arr = np.column_stack((x_arr, np.square(lstat)))
    optimal_params = np.matmul(np.matmul(np.linalg.inv(np.matmul(poly_arr.transpose(), poly_arr)), poly_arr.transpose()),
                               labels_arr)

    print("Bias and LSTAT and LSTAT2", optimal_params)

    prediction = np.matmul(poly_arr, optimal_params)
    # print("prediction", prediction)

    plt.xlabel("LSTAT")
    plt.ylabel("PRICE")
    plt.plot(lstat, labels_arr, 'r.')
    plt.plot(lstat, prediction, 'g.')
    plt.show()

    mse = np.square(np.subtract(labels_arr, prediction)).mean()

    print("MSE", mse)


if __name__ == '__main__':
    main()
