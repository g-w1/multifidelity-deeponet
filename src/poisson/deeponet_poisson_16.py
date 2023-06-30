import sys

import deepxde as dde
import numpy as np
import tensorflow as tf

dde.config.set_default_float("float16")
tf.keras.mixed_precision.set_global_policy("mixed_float16")


def get_data(
    fname_train, fname_test, residual=False, stackbranch=False, stacktrunk=False
):
    N = 500
    # i = 0
    # idx = np.arange(i * N, (i + 1) * N)
    idx = np.random.choice(100000, size=N, replace=False)

    d = np.load(fname_train)
    X_branch = d["X0"][idx]
    X_trunk = d["X1"][idx]
    if stackbranch:
        X_branch = np.hstack((d["X0"][idx], d["y_low"][idx]))
    if stacktrunk:
        X_trunk = np.hstack((d["X1"][idx], d["y_low_x"][idx]))
    X_train = (X_branch, X_trunk)
    y_train = d["y"][idx].astype(np.float16)
    if residual:
        y_train -= d["y_low_x"][idx]

    d = np.load(fname_test)
    X_branch = d["X0"]
    X_trunk = d["X1"]
    if stackbranch:
        X_branch = np.hstack((d["X0"], d["y_low"])).astype(np.float16)
    if stacktrunk:
        X_trunk = np.hstack((d["X1"], d["y_low_x"])).astype(np.float16)
    X_test = (X_branch, X_trunk)
    y_test = d["y"].astype(np.float16)
    if residual:
        y_test -= d["y_low_x"]
    return X_train, y_train, X_test, y_test


def run(data, net, lr, epochs):
    model = dde.Model(data, net)
    model.compile("adam", lr=lr)
    losshistory, train_state = model.train(epochs=epochs)
    dde.saveplot(losshistory, train_state, issave=False, isplot=True)


def main():
    fname_train = "../data/poisson_train.npz"
    fname_test = "../data/poisson_train.npz"
    X_train, y_train, X_test, y_test = get_data(
        fname_train, fname_test, residual=True, stackbranch=False, stacktrunk=False
    )
    data = dde.data.Triple(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    m = 100
    dim_x = 1
    width = 5
    net = dde.nn.DeepONetCartesianProd(
        [m, width, width],
        [dim_x, width],
        "selu",
        "LeCun normal",
    )

    lr = 0.0001
    epochs = 50000
    run(data, net, lr, epochs)


if __name__ == "__main__":
    main()
