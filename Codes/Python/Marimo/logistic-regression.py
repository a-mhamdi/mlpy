import marimo

__generated_with = "0.9.31"
app = marimo.App()


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # Machine Learning

        **Textbook is available @ [https://www.github.com/a-mhamdi/mlpy](https://www.github.com/a-mhamdi/mlpy)**

        ---
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Logistic Regression""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        Logistic regression is a statistical method used for classification tasks. It is used to predict the probability that an instance belongs to a particular class. Logistic regression is named for the function used at the core of the method, the logistic function.

        In logistic regression, the goal is to find the best fitting model that represents the relationship between the independent variables and the dependent variable. The dependent variable is binary, meaning it can take on only two values (such as `yes` or `no`), and the independent variables can be continuous or categorical.

        The logistic function is used to model the probability that an instance belongs to a particular class. The logistic function takes the form:

        $$y \;=\; p(z=1|X) \;=\; \sigma(z) \;=\; \dfrac{1}{1 + \mathrm{e}^{-z}}$$

        where $p$ is the probability that the instance belongs to the positive class, and $z$ is the linear combination of the independent variables and the model coefficients. The linear combination is calculated as:

        $$z \;=\; \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_{m-1} x_{m-1} \;=\; x^T\theta$$

        where $x_1$, $x_2$, ..., $x_{m-1}$ are the independent variables, and $\theta_0$, $\theta_1$, $\theta_2$, ..., $\theta_{m-1}$ are the coefficients that represent the influence of each variable on the dependent variable. The coefficients are estimated using the data, and the resulting equation is used to make predictions on new data.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Importing the libraries""")
    return


@app.cell
def __():
    import numpy as np
    import pandas as pd
    return np, pd


@app.cell
def __():
    from matplotlib import pyplot as plt
    plt.style.use('dark_background')
    plt.rc('figure', figsize=(6, 4))

    from matplotlib import rcParams
    rcParams['font.family'] = 'Comfortaa'
    rcParams['font.size'] = 8
    rcParams['axes.unicode_minus'] = False
    return plt, rcParams


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Importing the dataset""")
    return


@app.cell
def __(pd):
    df = pd.read_csv('../Datasets/Social_Network_Ads.csv')
    return (df,)


@app.cell
def __(df):
    df.head()
    return


@app.cell
def __(df):
    df.info()
    return


@app.cell
def __(df):
    df.describe()
    return


@app.cell
def __(df):
    df.Purchased.value_counts()
    return


@app.cell
def __():
    143/(143+257)
    return


@app.cell
def __(df):
    X = df.iloc[:, :-1].values; X[:, -1] = X[:, -1] / 1000
    y = df.iloc[:, -1].values
    return X, y


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Splitting the dataset into training and test sets""")
    return


@app.cell
def __(X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=123, 
                                                        stratify=y)
    return X_test, X_train, train_test_split, y_test, y_train


@app.cell
def __(np, y):
    np.sum(y == 1) / len(y)
    return


@app.cell
def __(np, y_train):
    np.sum(y_train == 1) / len(y_train)
    return


@app.cell
def __(np, y_test):
    np.sum(y_test == 1) / len(y_test)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Training the Logistic Regression model on the Training set""")
    return


@app.cell
def __():
    from sklearn.linear_model import LogisticRegression
    return (LogisticRegression,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""The following code will create a logistic regression model that fits a curve to the training data. We will need it later in order to make predictions on the test data.""")
    return


@app.cell
def __(LogisticRegression, X_train, y_train):
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    return (clf,)


@app.cell
def __(clf):
    clf.coef_, clf.intercept_
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Predicting a sample input""")
    return


@app.cell
def __(np):
    sample = np.array([[30, 87]])
    return (sample,)


@app.cell
def __(clf, np, sample):
    z = np.sum(clf.coef_ * sample) + clf.intercept_
    o = 1/(1+np.exp(-z))
    o
    return o, z


@app.cell
def __(clf, sample):
    clf.predict_proba(sample) # p(y=0|X), p(y=1|X)
    return


@app.cell
def __(clf, sample):
    clf.predict(sample)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Predicting the test set input""")
    return


@app.cell
def __(X_test, clf):
    y_prob = clf.predict_proba(X_test)
    y_prob[:5]
    return (y_prob,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""The predicted class will be the class with the higher probability, as determined by the logistic function.""")
    return


@app.cell
def __(np, y_prob):
    np.argmax(y_prob, axis=1)
    return


@app.cell
def __(X_test, clf):
    y_pred = clf.predict(X_test)
    y_pred
    return (y_pred,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Making the Confusion Matrix""")
    return


@app.cell
def __():
    from sklearn.metrics import confusion_matrix, accuracy_score
    return accuracy_score, confusion_matrix


@app.cell
def __(confusion_matrix, y_pred, y_test):
    cm = confusion_matrix(y_test, y_pred)
    cm
    return (cm,)


@app.cell
def __(accuracy_score, y_pred, y_test):
    accuracy_score(y_test, y_pred)
    return


@app.cell
def __():
    from sklearn.metrics import ConfusionMatrixDisplay
    return (ConfusionMatrixDisplay,)


@app.cell
def __(ConfusionMatrixDisplay, clf, cm):
    ConfusionMatrixDisplay(cm, display_labels=clf.classes_).plot();
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Another fancy way to display the confusion matrix is to call the builtin method `crosstab` form `pandas` library as shown below.""")
    return


@app.cell
def __(pd, y_pred, y_test):
    pd.crosstab(y_test, y_pred, rownames=['Expected'], colnames=['Predicted'], margins=True)
    return


@app.cell
def __():
    from sklearn.metrics import classification_report
    return (classification_report,)


@app.cell
def __(classification_report, y_pred, y_test):
    print(classification_report(y_test, y_pred))
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Logistic regression handles both continuous and categorical independent variables, and outputs probabilities for each class, which can be useful for tasks such as fraud detection or medical diagnosis. However, it's important to note that the independent variables must be linearly related to the log odds of the dependent variable in order for logistic regression to be appropriate. If the relationship is non-linear, we may need to use a different type of classification algorithm.""")
    return


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
