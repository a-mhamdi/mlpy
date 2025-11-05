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
    mo.md(r"""## K-Nearest Neighbors (K-NN)""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        $k$-nearest neighbors ($k$-NN) is a type of instance-based learning, a method of supervised machine learning. It is used for classification and regression tasks.

        In $k$-NN, the algorithm is given a labeled training dataset and a set of test data. To make a prediction for a test instance, the algorithm looks at the $k$ nearest neighbors in the training dataset, based on the distance between the test instance and the training instances. The prediction is then made based on the majority class among the $k$ nearest neighbors. For classification tasks, the prediction is the class with the most neighbors. For regression tasks, the prediction is the mean or median of the values of the $k$ nearest neighbors.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Importing the libraries""")
    return


@app.cell
def _():
    import pandas as pd
    return (pd,)


@app.cell
def _():
    from matplotlib import pyplot as plt
    plt.rc('figure', figsize=(6, 4))
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Importing the dataset""")
    return


@app.cell
def _(pd):
    df = pd.read_csv('../Datasets/Social_Network_Ads.csv')
    df.head()
    return (df,)


@app.cell
def _(df):
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Splitting the dataset into the Training set and Test set""")
    return


@app.cell
def _(X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)
    return X_test, X_train, train_test_split, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Feature Scaling""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""$k$-NN is sensitive to the scale of the features, and it may not perform well if the features have very different scales.""")
    return


@app.cell
def _():
    from sklearn.preprocessing import StandardScaler
    return (StandardScaler,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""In order to avoid *information leakage*, it is highly important to keep in mind that only the `transform` method has to be applied on the `X_test`. $(\mu,\ \sigma)$ are of `X_train` set.""")
    return


@app.cell
def _(StandardScaler, X_test, X_train):
    sc = StandardScaler()
    X_train_1 = sc.fit_transform(X_train)
    X_test_1 = sc.transform(X_test)
    return X_test_1, X_train_1, sc


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Training the k-NN model on the training set""")
    return


@app.cell
def _():
    from sklearn.neighbors import KNeighborsClassifier
    return (KNeighborsClassifier,)


@app.cell
def _(mo):
    slider = mo.ui.slider(1, 10, 1, 5)
    slider
    return (slider,)


@app.cell
def _(KNeighborsClassifier, slider):
    clf = KNeighborsClassifier(n_neighbors=slider.value, metric='minkowski', p=2)
    return (clf,)


@app.cell
def _(X_train_1, clf, y_train):
    clf.fit(X_train_1, y_train)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Predicting a new result""")
    return


@app.cell
def _(clf, sc):
    clf.predict(sc.transform([[30,87000]]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Predicting the test set results""")
    return


@app.cell
def _(X_test_1, clf):
    y_pred = clf.predict(X_test_1)
    return (y_pred,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Displaying the Confusion Matrix""")
    return


@app.cell
def _():
    from sklearn.metrics import confusion_matrix
    return (confusion_matrix,)


@app.cell
def _(clf, confusion_matrix, y_pred, y_test):
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    return (cm,)


@app.cell
def _(cm):
    cm
    return


@app.cell
def _():
    from sklearn.metrics import ConfusionMatrixDisplay
    return (ConfusionMatrixDisplay,)


@app.cell
def _(ConfusionMatrixDisplay, clf, cm):
    ConfusionMatrixDisplay(cm, display_labels=clf.classes_).plot();
    return


@app.cell
def _():
    from sklearn.metrics import accuracy_score
    return (accuracy_score,)


@app.cell
def _(accuracy_score, y_pred, y_test):
    print(f'Accuracy = {accuracy_score(y_test, y_pred):.2f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""[https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.crosstab.html](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.crosstab.html)""")
    return


@app.cell
def _(pd, y_pred, y_test):
    pd.crosstab(y_test, y_pred, rownames=['Expected'], colnames=['Predicted'], margins=True)
    return


@app.cell
def _():
    from sklearn.metrics import classification_report
    return (classification_report,)


@app.cell
def _(classification_report, y_pred, y_test):
    print(classification_report(y_test, y_pred))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""$k$-NN is a simple and effective method for classification and regression tasks, and it is easy to understand and implement. However, it can be computationally expensive to find the $k$ nearest neighbors for each test instance, especially for large datasets.""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
