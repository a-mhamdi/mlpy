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
    mo.md(r"""## Polynomial Regression""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Polynomial regression is a type of regression analysis in which the relationship between the independent variables in $X$ and the dependent variable $y$ is modeled as a $p^\text{th}$ degree polynomial. It helps in modeling relationships between variables that are not linear.""")
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
def __(np):
    np.set_printoptions(precision=3)
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


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Importing the dataset""")
    return


@app.cell
def __(pd):
    df = pd.read_csv('../Datasets/Position_Salaries.csv')
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
    df.columns
    return


@app.cell
def __(df):
    X = df.iloc[:, 1].values # Level
    y = df.iloc[:, -1].values # Salary
    return X, y


@app.cell
def __(X):
    print(type(X), X[:5], sep='\n')
    return


@app.cell
def __(y):
    print(type(y), y[:5], sep='\n')
    return


@app.cell
def __(X):
    X.shape
    return


@app.cell
def __(X):
    X_1 = X.reshape(-1, 1)
    return (X_1,)


@app.cell
def __(y):
    y.shape
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Training the linear regression model on the whole dataset""")
    return


@app.cell
def __():
    from sklearn.linear_model import LinearRegression
    return (LinearRegression,)


@app.cell
def __(LinearRegression, X_1, y):
    lr_1 = LinearRegression()
    lr_1.fit(X_1, y)
    return (lr_1,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Visualizing the linear regression predictions""")
    return


@app.cell
def __(X_1, lr_1, plt, y):
    plt.scatter(X_1, y, color='green')
    plt.plot(X_1, lr_1.predict(X_1.reshape(-1, 1)), color='red')
    plt.title('Predictions made by `lr_1`')
    plt.xlabel('Level')
    plt.ylabel('Salary')
    plt.grid()
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        The **Taylor** series of a real or complex-valued function `f(x)` that is infinitely differentiable at a real or complex number `a` is the power series [[Wikipedia]](https://en.wikipedia.org/wiki/Taylor_series):

        $$\displaystyle f(x) \;=\; f(a)+{\frac {f'(a)}{1!}}(x-a)+{\frac {f''(a)}{2!}}(x-a)^{2}+{\frac {f'''(a)}{3!}}(x-a)^{3}+\cdots \;=\; \sum_{i=0}^{+\infty}\dfrac{f^{(i)}(a)}{i!}~\left(x-a\right)^i$$
        """
    )
    return


@app.cell
def __(X_1, np):
    Xnew = np.concatenate((X_1, X_1 ** 2, X_1 ** 3, X_1 ** 4), axis=1)
    Xnew
    return (Xnew,)


@app.cell
def __(LinearRegression, Xnew, y):
    lr_2 = LinearRegression()
    lr_2.fit(Xnew, y)
    return (lr_2,)


@app.cell
def __(X_1, Xnew, lr_2, plt, y):
    plt.scatter(X_1, y, color='green')
    plt.plot(X_1, lr_2.predict(Xnew), color='red')
    plt.title('Predictions made by `lr_2`')
    plt.xlabel('Level')
    plt.ylabel('Salary')
    plt.grid()
    return


@app.cell
def __():
    from sklearn.preprocessing import PolynomialFeatures
    return (PolynomialFeatures,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""The goal is to find the polynomial equation that best fits the data. The polynomial equation can be of any degree, but typically the degree is chosen based on the shape of the curve in the data.""")
    return


@app.cell
def __(PolynomialFeatures, X_1):
    poly_reg = PolynomialFeatures(degree=4)
    X_poly = poly_reg.fit_transform(X_1)
    print(X_poly[:5])
    return X_poly, poly_reg


@app.cell
def __(LinearRegression, X_poly, y):
    lr_3 = LinearRegression()
    lr_3.fit(X_poly, y)
    return (lr_3,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""The code shown above has created a polynomial transformer that transforms the input data into a polynomial representation, and then creates a linear regression model that fits a polynomial to the transformed data. The polynomial degree is specified by the `degree` parameter in the `PolynomialFeatures` transformer.""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Visualizing the polynomial regression predictions""")
    return


@app.cell
def __(X_1, lr_3, plt, poly_reg, y):
    plt.scatter(X_1, y, color='green')
    plt.plot(X_1, lr_3.predict(poly_reg.fit_transform(X_1)), color='red')
    plt.title('Predictions made by `lr_3`')
    plt.xlabel('Level')
    plt.ylabel('Salary')
    plt.grid()
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Predicting a new result using the linear regressor""")
    return


@app.cell
def __(lr_1):
    lr_1.predict([[6.5]])
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Predicting a new result using the polynomial regressor""")
    return


@app.cell
def __(lr_2, np):
    lr_2.predict(np.array([[6.5, 6.5**2, 6.5**3, 6.5**4]]))
    return


@app.cell
def __(lr_3, poly_reg):
    lr_3.predict(poly_reg.fit_transform([[6.5]]))
    return


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
