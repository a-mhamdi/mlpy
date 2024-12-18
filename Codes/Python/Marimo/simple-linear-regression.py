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
    mo.md(r"""Linear regression is a statistical method used to model the linear relationship between a dependent variable and one or more independent variables. Simple linear regression is a type of linear regression that involves only one independent variable.""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Simple Linear Regression""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""The goal is to find the line of best fit that represents the relationship between the independent variable (aka the predictor or explanatory variable) and the dependent variable (aka the response or outcome variable). The line of best fit is a line that is as close as possible to the data points in the scatterplot of the variables.""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Libraries and settings""")
    return


@app.cell
def __():
    import numpy as np
    import pandas as pd
    return np, pd


@app.cell
def __(np):
    np.set_printoptions(precision=2)
    return


@app.cell
def __():
    from matplotlib import pyplot as plt
    plt.style.use('dark_background')
    plt.rc('figure', figsize=(6, 4))

    from matplotlib import rcParams
    rcParams['font.family'] = 'Monospace'
    rcParams['font.size'] = 8
    rcParams['axes.unicode_minus'] = False
    return plt, rcParams


@app.cell
def __():
    #%matplotlib widget
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""`%matplotlib widget` is a magic command in _Jupyter_ Notebooks which enables interactive features such as panning, zooming of plots, as well as the ability to hover over data point to display their values.""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Importing the dataset""")
    return


@app.cell
def __(pd):
    df = pd.read_csv('../Datasets/Salary_Data.csv')
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
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Data split""")
    return


@app.cell
def __():
    from sklearn.model_selection import train_test_split
    return (train_test_split,)


@app.cell
def __(X, train_test_split, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=123)
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Training""")
    return


@app.cell
def __():
    from sklearn.linear_model import LinearRegression
    return (LinearRegression,)


@app.cell
def __(LinearRegression, X_train, y_train):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return (lr,)


@app.cell
def __(lr):
    _theta_1 = lr.coef_
    _theta_1
    return


@app.cell
def __(lr):
    _b = lr.intercept_
    _b
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Prediction and evaluation""")
    return


@app.cell
def __(X_test, lr):
    y_pred = lr.predict(X_test)
    return (y_pred,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""#### Evaluation Metrics""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""First of all, let's compute, using the basic operations, the mean absolute, mean squared and mean absolute percentage errors:""")
    return


@app.cell
def __(np, y_pred, y_test):
    _mae = np.abs(y_pred - y_test)
    _mae.mean()
    return


@app.cell
def __(y_pred, y_test):
    _mse = (y_pred - y_test) ** 2
    _mse.mean()
    return


@app.cell
def __(np, y_pred, y_test):
    _mape = np.abs((y_pred - y_test) / y_test)
    _mape.mean()
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""We can also evaluate the mean absolute error, the mean squared error and the mean absolute percentage error, denoted here `mae`, `mse` and `mape`, by calling the built-in methods `mean_absolute_error`, `mean_squared_error` and `mean_absolute_percentage_error`""")
    return


@app.cell
def __():
    from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
    return (
        mean_absolute_error,
        mean_absolute_percentage_error,
        mean_squared_error,
    )


@app.cell
def __(mean_absolute_error, y_pred, y_test):
    _mae = mean_absolute_error(y_test, y_pred)
    _mae
    return


@app.cell
def __(mean_squared_error, y_pred, y_test):
    _mse = mean_squared_error(y_test, y_pred)
    _mse
    return


@app.cell
def __(mean_absolute_percentage_error, y_pred, y_test):
    _mape = mean_absolute_percentage_error(y_test, y_pred)
    _mape
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""#### Visualization""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""We can also use _Matplotlib_ to visualize the line of best fit, to which, we add the original test data points in a `scatterplot`.""")
    return


@app.cell
def __(X_test, plt, y_pred, y_test):
    plt.scatter(X_test, y_test, color='green')
    plt.plot(X_test, y_pred, color='red')
    plt.title('Test Set')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.grid()
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### BONUS""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""The following cells are to check the validity of `lr` against the normal equation, aka, **Ordinary Least Squares (OLS)** method as seen in class. For the sake of simplicity, we suppose we don't fit the intercept.""")
    return


@app.cell
def __(LinearRegression, X_train, y_train):
    lr_1 = LinearRegression(fit_intercept=False)
    lr_1.fit(X_train, y_train)
    return (lr_1,)


@app.cell
def __(lr_1):
    _theta_1 = lr_1.coef_
    _theta_1
    return


@app.cell
def __(lr_1):
    _b = lr_1.intercept_
    _b
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Check the value of `theta_1` using **OLS**:""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""$$\hat{\theta} \;=\; \displaystyle\dfrac{\sum_{i=0}^{n-1} y_i x_i}{\sum_{i=0}^{n-1} x_i^2}$$""")
    return


@app.cell
def __(X_train, y_train):
    theta_hat = (y_train @ X_train)/(X_train.T @ X_train)
    theta_hat
    return (theta_hat,)


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
