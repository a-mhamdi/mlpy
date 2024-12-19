import marimo

__generated_with = "0.10.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Machine Learning

        **Textbook is available @ [https://www.github.com/a-mhamdi/mlpy](https://www.github.com/a-mhamdi/mlpy)**

        ---
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Artificial neural networks (ANN) are commonly used for classification tasks because they are able to learn complex relationships between the input features and the target class. They are particularly useful when the relationship is non-linear, as they are able to learn and model the inputs-outputs mapping using multiple hidden layers of interconnected neurons.

        ANN are also able to handle large amounts of data and can learn from it without being explicitly programmed with a set of rules or a decision tree. This allows them to be very flexible and adaptable, and makes them well-suited for tasks that are difficult to define using traditional programming techniques.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Binary Classification using ANN""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        There are several advantages to using neural networks for classification tasks:

        1. They are able to learn complex relationships between the input features and the target class;
        1. They are able to handle large amounts of data;
        1. They can learn from unstructured data;
        1. They are flexible and adaptable;
        1. They can be trained to perform well on a wide range of classification tasks.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Importing the libraries""")
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    return np, pd


@app.cell
def _(np):
    np.set_printoptions(precision=2)
    return


@app.cell
def _():
    from matplotlib import pyplot as plt
    plt.style.use('dark_background')
    plt.rc('figure', figsize=(6, 4))

    from matplotlib import rcParams
    rcParams['font.family'] = 'Comfortaa'
    rcParams['font.size'] = 8
    rcParams['axes.unicode_minus'] = False
    return plt, rcParams


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Importing the dataset""")
    return


@app.cell
def _(pd):
    df = pd.read_csv("../Datasets/Churn_Modelling.csv")
    return (df,)


@app.cell
def _(df):
    df_1 = df.dropna(how='any', axis=0)
    return (df_1,)


@app.cell
def _(df_1):
    df_1.head()
    return


@app.cell
def _(df_1):
    df_1.info()
    return


@app.cell
def _(df_1):
    df_1.describe()
    return


@app.cell
def _(df_1):
    df_1.Exited.value_counts()
    return


@app.cell
def _():
    from random import sample
    return (sample,)


@app.cell
def _(df_1, sample):
    target = df_1.Exited
    param = 7963 - 2037
    records_to_drop = sample(list(target[target == 0].index), param)
    df_1.drop(records_to_drop, axis=0, inplace=True)
    return param, records_to_drop, target


@app.cell
def _(df_1):
    X = df_1.iloc[:, 3:-1].values
    y = df_1.iloc[:, -1].values
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Data preprocessing""")
    return


@app.cell
def _():
    from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
    return OneHotEncoder, OrdinalEncoder


@app.cell
def _(OneHotEncoder, OrdinalEncoder):
    oe = OrdinalEncoder()
    ohe = OneHotEncoder()
    return oe, ohe


@app.cell
def _():
    from sklearn.compose import ColumnTransformer
    return (ColumnTransformer,)


@app.cell
def _(ColumnTransformer, oe, ohe):
    ct = ColumnTransformer([("ohe", ohe, [1]), ("oe", oe, [2])], remainder='passthrough')
    return (ct,)


@app.cell
def _(ct):
    ct
    return


@app.cell
def _(X, ct):
    X_1 = ct.fit_transform(X)
    return (X_1,)


@app.cell
def _(X_1):
    X_1[:5]
    return


@app.cell
def _():
    from sklearn.model_selection import train_test_split
    return (train_test_split,)


@app.cell
def _(X_1, train_test_split, y):
    (X_train, X_test, y_train, y_test) = train_test_split(X_1, y, train_size=0.8, random_state=123, stratify=y)
    return X_test, X_train, y_test, y_train


@app.cell
def _():
    from sklearn.preprocessing import MinMaxScaler
    return (MinMaxScaler,)


@app.cell
def _(MinMaxScaler):
    mms = MinMaxScaler()
    return (mms,)


@app.cell
def _(X_test, X_train, mms):
    X_train_1 = mms.fit_transform(X_train)
    X_test_1 = mms.transform(X_test)
    return X_test_1, X_train_1


@app.cell
def _(X_train_1):
    print(X_train_1[:5, :])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Build the classifier `clf`""")
    return


@app.cell
def _():
    from keras.models import Sequential
    from keras.layers import Input, Dense
    return Dense, Input, Sequential


@app.cell
def _(Dense, Input, Sequential, X_train_1):
    clf = Sequential()
    ndim = X_train_1.shape[1]
    clf.add(Input(shape=(ndim,)))
    clf.add(Dense(units=16, activation='relu'))
    clf.add(Dense(units=8, activation='relu'))
    clf.add(Dense(units=4, activation='relu'))
    clf.add(Dense(units=1, activation='sigmoid'))
    return clf, ndim


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Insights about `clf`""")
    return


@app.cell
def _():
    from keras.utils import plot_model
    return (plot_model,)


@app.cell
def _(clf, plot_model):
    plot_model(clf, show_shapes=True)
    return


@app.cell
def _(clf):
    clf.summary()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Compile `clf`""")
    return


@app.cell
def _():
    from keras.optimizers import Adam
    from keras.losses import binary_crossentropy
    return Adam, binary_crossentropy


@app.cell
def _(mo):
    eta = mo.ui.slider(.001, .1, .01, .01)
    eta
    return (eta,)


@app.cell
def _(Adam, eta):
    opt = Adam(learning_rate=eta.value)
    return (opt,)


@app.cell
def _():
    import tensorflow as tf
    tf.keras.metrics.Precision
    return (tf,)


@app.cell
def _(binary_crossentropy, clf, opt):
    clf.compile(optimizer=opt, 
                loss=binary_crossentropy, 
                metrics=['Accuracy', 'Precision', 'Recall'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Train and evaluate `clf`""")
    return


@app.cell
def _(mo):
    batchsize = mo.ui.slider(2, 64, 4, 32)
    batchsize
    return (batchsize,)


@app.cell
def _(mo):
    epochs = mo.ui.slider(1, 100, 2, 32)
    epochs
    return (epochs,)


@app.cell
def _(X_train_1, batchsize, clf, epochs, y_train):
    classifier_history = clf.fit(X_train_1, y_train, validation_split=0.1, batch_size=batchsize.value, epochs=epochs.value)
    return (classifier_history,)


@app.cell
def _():
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    return ConfusionMatrixDisplay, confusion_matrix


@app.cell
def _(X_test_1, clf):
    y_pred = clf.predict(X_test_1)
    y_pred = (y_pred > 0.5).astype(int)
    return (y_pred,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Print the confusion matrix""")
    return


@app.cell
def _(confusion_matrix, y_pred, y_test):
    cm = confusion_matrix(y_test, y_pred)
    return (cm,)


@app.cell
def _(ConfusionMatrixDisplay, cm):
    ConfusionMatrixDisplay(cm).plot();
    return


@app.cell
def _(y_test):
    print(y_test.shape)
    return


@app.cell
def _(y_pred):
    print(y_pred.shape)
    y_pred_1 = y_pred.reshape(len(y_pred))
    print(y_pred_1.shape)
    return (y_pred_1,)


@app.cell
def _(pd, y_pred_1, y_test):
    pd.crosstab(y_test, y_pred_1, rownames=['Expected'], colnames=['Predicted'], margins=True)
    return


@app.cell
def _(np, y_pred_1, y_test):
    y_test_1 = y_test.reshape(len(y_test), 1)
    y_pred_2 = y_pred_1.reshape(len(y_pred_1), 1)
    print(np.concatenate((y_test_1[:10], y_pred_2[:10]), axis=1))
    return y_pred_2, y_test_1


@app.cell
def _():
    from sklearn.metrics import classification_report
    return (classification_report,)


@app.cell
def _(classification_report, y_pred_2, y_test_1):
    print(classification_report(y_test_1, y_pred_2))
    return


@app.cell
def _(classifier_history):
    metrics = list(classifier_history.history.keys())
    metrics
    return (metrics,)


@app.cell
def _(classifier_history, metrics, plt):
    (fig, axs) = plt.subplots(1, 4)
    plt.rc('figure', figsize=(12, 6))
    axs[0].plot(classifier_history.history['loss'])
    axs[0].plot(classifier_history.history['val_loss'])
    axs[1].plot(classifier_history.history['Accuracy'])
    axs[1].plot(classifier_history.history['val_Accuracy'])
    axs[2].plot(classifier_history.history['Precision'])
    axs[2].plot(classifier_history.history['val_Precision'])
    axs[3].plot(classifier_history.history['Recall'])
    axs[3].plot(classifier_history.history['val_Recall'])

    for (idx, metric) in enumerate(metrics[:4]):
        ax = axs[idx]
        ax.legend(['train', 'val'], loc='lower right')
        ax.set_title(metric)
        ax.set_xlabel('epoch')

    plt.show()
    return ax, axs, fig, idx, metric


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""It is important to note that neural networks can be more computationally intensive to train and may require more data and more time to achieve better performance compared to some other classification algorithms. Furthermore, because they discover patterns in the data using the network's weights and biases rather than explicit rules, they may be more challenging to read and comprehend.""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
