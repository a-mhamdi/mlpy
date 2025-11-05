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
    mo.md(r"""## Multiple Linear Regression""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Multiple linear regression is a type of regression analysis in which there are multiple independent variables that have an effect on the dependent variable. In multiple linear regression, the goal is to find the linear equation that best explains the relationship between the outcome and the features in $X$.

        The equation takes the form:

        $$y \;=\; \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_{m-1} x_{m-1}$$

        where $y$ is the dependent variable, $x_1$, $x_2$, ..., $x_{m-1}$ are the independent variables, and $\theta_0$, $\theta_1$, $\theta_2$, ..., $\theta_{m-1}$ are the coefficients that represent the influence of each variable on the output $y$. The coefficients are estimated using the data, and the resulting equation can be used later to make predictions on new data.
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
    np.set_printoptions(precision=3)
    return


@app.cell
def _():
    from matplotlib import pyplot as plt
    plt.rc('figure', figsize=(6, 4))
    return (plt,)


@app.cell(hide_code=True)
def _():
    # Show plots in an interactive format, e.g., zooming, saving, etc
    # %matplotlib widget
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Importing the dataset""")
    return


@app.cell
def _(pd):
    df = pd.read_csv('../Datasets/50_Startups.csv')
    return (df,)


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df):
    df.info()
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Extract features $X$ and target $y$ from the dataset. **Profit** is the dependant variable.""")
    return


@app.cell
def _(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Check the first five observations within $X$""")
    return


@app.cell
def _(X):
    X.head()
    return


@app.cell
def _(X):
    X_1 = X.values
    type(X_1)
    return (X_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Check the corresponding first five values from **Profit** column.""")
    return


@app.cell
def _(y):
    y.head()
    return


@app.cell
def _(y):
    y_1 = y.values
    type(y_1)
    return (y_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Encoding categorical data""")
    return


@app.cell
def _():
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    return ColumnTransformer, OneHotEncoder


@app.cell
def _(ColumnTransformer, OneHotEncoder, X_1, np):
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
    X_2 = np.array(ct.fit_transform(X_1))
    return X_2, ct


@app.cell
def _(X_2):
    print(X_2[:5])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Splitting the dataset into training set and test set""")
    return


@app.cell
def _():
    from sklearn.model_selection import train_test_split
    return (train_test_split,)


@app.cell
def _(X_2, train_test_split, y_1):
    (X_train, X_test, y_train, y_test) = train_test_split(X_2, y_1, test_size=0.2, random_state=123)
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Training the multiple linear regression model on the training set""")
    return


@app.cell
def _():
    from sklearn.linear_model import LinearRegression
    return (LinearRegression,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""This code will create a linear regression model that fits a line to the training data, in order to make future predictions on the test data.""")
    return


@app.cell
def _(LinearRegression, X_train, y_train):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return (lr,)


@app.cell
def _(lr):
    theta = lr.coef_
    theta
    return (theta,)


@app.cell
def _(lr):
    b = lr.intercept_
    b
    return (b,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Consider the sample `tst` as follows:""")
    return


@app.cell
def _(np):
    tst = np.array([1, 0, 0, 15e+3, 10e+2, 5e+6])
    return (tst,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Predict the outcome if `tst` is the input.""")
    return


@app.cell
def _(b, theta, tst):
    pred = theta @ tst + b
    print('%.3f' % pred)
    return (pred,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""By calling our `lr`, we get the same result:""")
    return


@app.cell
def _(lr, tst):
    lr.predict(tst.reshape(1, -1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""If we don't want to do the encoding of state feature by ourselves, we can invoke the previous `ct` object.""")
    return


@app.cell
def _(ct, np):
    tst_new = [[15e+3, 10e+2, 5e+6, 'California']]
    arr = np.array(ct.transform(tst_new))
    arr
    return arr, tst_new


@app.cell
def _(arr, lr):
    lr.predict(arr)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Evaluation and Visualization""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Make predictions using the $X$ test set and visualize the results""")
    return


@app.cell
def _(X_test, lr):
    y_pred = lr.predict(X_test)
    return (y_pred,)


@app.cell
def _(plt, y_pred, y_test):
    # y_pred vs. y_test
    plt.scatter(y_test, y_pred, c='red')
    plt.plot(y_test, y_test, '-*')
    plt.grid()
    return


@app.cell
def _():
    from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
    return (
        mean_absolute_error,
        mean_absolute_percentage_error,
        mean_squared_error,
    )


@app.cell
def _(
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    y_pred,
    y_test,
):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) # relative error: *100
    return mae, mape, mse


@app.cell
def _(mae, mape, mse):
    mae, mse, mape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Multiple linear regression can be used to understand the relationship between multiple independent variables and a single dependent variable, and can be used to make predictions about the dependent variable given new data. However, it's important to note that the independent variables must be linearly related to the dependent variable in order for multiple linear regression to behave appropriately. If the relationship is non-linear, we need to use a different type of regression analysis such as polynomial regression.""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
