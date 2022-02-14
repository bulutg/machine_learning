import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    k = 500
    arr = np.array(pd.read_csv('images.csv'))
    # mean center the data
    mean_centered = arr - arr.mean()
    # calculate eigenvalues and eigenvectors of covariance matrix
    w, v = np.linalg.eig(np.cov(mean_centered.T))
    # Eigenvector with the largest eigenvalue λn is nth principal component
    sorted_eigenvectors = v[:, np.argsort(w)[::-1]]

    rows = 5
    columns = 2

    k_subset = sorted_eigenvectors[:, 0:k]

    pves = []
    k_s =[1,10,50,100,500]

    fig = plt.figure(figsize=(10, 10))
    #for i in range(k):
    for i in k_s:
        #fig.add_subplot(rows, columns, i + 1)
        #plt.imshow(k_subset.T[i, :].reshape((48, 48)))
        print("PVE for k", i, w[i]/np.sum(w))
        pves.append(w[i]/np.sum(w))

    plt.xlabel("k")
    plt.ylabel("PVE")
    plt.plot(k_s,pves)
    plt.show()

    # q1.3
    reconstruct = np.matmul(k_subset, np.matmul(k_subset.T, arr[0].reshape((2304, 1))))
    plt.imshow(reconstruct.reshape((48, 48)))
    plt.show()


if __name__ == '__main__':
    main()
