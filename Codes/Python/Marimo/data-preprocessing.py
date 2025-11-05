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
    mo.md(r"""## Data Preprocessing Template""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        It is important to carefully consider the preprocessing steps that are appropriate for the specific dataset and machine learning task. Preprocessing of data helps to ensure that the data is in a suitable format to use, and can also help to improve the generalization ability of the model.

        There are several reasons why data preprocessing is important in machine learning:

        1. Cleaning and formatting the data;
        1. Normalizing the data;
        1. Reducing the dimensionality of the data; and
        1. Enhancing the interpretability of the model.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Introduction to Data Scaling""")
    return


@app.cell
def _():
    import numpy as np
    return (np,)


@app.cell
def _(np):
    X = np.array([[1, -1], [0, 2], [4.5, -3], [0, 9], [1.3, -2], [5, 4]])
    X
    return (X,)


@app.cell
def _():
    import pandas as pd
    return (pd,)


@app.cell
def _(X, pd):
    df = pd.DataFrame(X, columns=['Col #1', 'Col #2'])
    df
    return (df,)


@app.cell
def _(df):
    df.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### MinMaxScaler""")
    return


@app.cell
def _(X):
    X_pg = (X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0))
    X_pg
    return (X_pg,)


@app.cell
def _():
    from sklearn.preprocessing import MinMaxScaler
    return (MinMaxScaler,)


@app.cell
def _(MinMaxScaler, X):
    X_mms = MinMaxScaler().fit_transform(X)
    X_mms
    return (X_mms,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### StandardScaler""")
    return


@app.cell
def _(X):
    X_ms = (X-X.mean(axis=0))/(X.std(axis=0))
    X_ms
    return (X_ms,)


@app.cell
def _():
    from sklearn.preprocessing import StandardScaler
    return (StandardScaler,)


@app.cell
def _(StandardScaler, X):
    X_sc = StandardScaler().fit_transform(X)
    X_sc
    return (X_sc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Data Preprocessing Template""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Importing the libraries""")
    return


@app.cell
def _():
    from matplotlib import pyplot as plt
    return (plt,)


@app.cell
def _(np):
    np.set_printoptions(precision=3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Importing the dataset""")
    return


@app.cell
def _(pd):
    df_1 = pd.read_csv('../Datasets/Data.csv')
    df_1.head()
    return (df_1,)


@app.cell
def _(df_1):
    df_1.describe()
    return


@app.cell
def _(df_1):
    df_1['Purchased'].value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Extracting independent and dependent variables""")
    return


@app.cell
def _(df_1):
    X_1 = df_1.iloc[:, :-1].values
    y = df_1.iloc[:, -1].values
    return X_1, y


@app.cell
def _(X_1, y):
    print('***** Features *****', X_1, sep='\n')
    print('***** Target *****', y, sep='\n')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Imputation transformer for completing missing values""")
    return


@app.cell
def _(mo):
    mo.md(r"""[https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)""")
    return


@app.cell
def _():
    from sklearn.impute import SimpleImputer
    return (SimpleImputer,)


@app.cell
def _(SimpleImputer, X_1, np):
    si = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_1[:, 1:] = si.fit_transform(X_1[:, 1:])
    return (si,)


@app.cell
def _(X_1, y):
    print('***** Features *****', X_1, sep='\n')
    print('***** Target *****', y, sep='\n')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### How to encode categorical data?""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""##### Case of two categories""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""[https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)""")
    return


@app.cell
def _():
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    return LabelEncoder, le


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Fit `le` and return encoded labels""")
    return


@app.cell
def _(le, y):
    y_1 = le.fit_transform(y)
    print(y_1)
    return (y_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **MARGINAL NOTE**

        Try to fit and transform a new `LabelEncoder` instance on the `Country` column.

        >```python
        >ce = LabelEncoder()
        >country = ce.fit_transform(X[:, 0]) # You can use `df.Country` instead
        >```

        We can access the original values by simply writing:

        >```python
        >X[:, 0] = ce.inverse_transform(country) # X[:, 0].astype(int)
        >```
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""##### Case of multiple categories""")
    return


@app.cell
def _():
    from sklearn.compose import ColumnTransformer
    return (ColumnTransformer,)


@app.cell
def _(mo):
    mo.md(r"""[https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)""")
    return


@app.cell
def _():
    from sklearn.preprocessing import OneHotEncoder
    return (OneHotEncoder,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""[https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder)""")
    return


@app.cell
def _(ColumnTransformer, OneHotEncoder):
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
    ct
    return (ct,)


@app.cell
def _(X_1, ct, np):
    X_2 = np.array(ct.fit_transform(X_1))
    return (X_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Display `X` after being encoded""")
    return


@app.cell
def _(X_2):
    print(X_2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""_REMARK_""")
    return


@app.cell
def _():
    z = [['Python'], ['Julia'], ['Rust'], ['JavaScript']]
    Z = [3 * _ for _ in z]
    Z
    return Z, z


@app.cell
def _():
    from sklearn.preprocessing import OrdinalEncoder
    return (OrdinalEncoder,)


@app.cell
def _(ColumnTransformer, OneHotEncoder, OrdinalEncoder):
    ctz = ColumnTransformer(transformers=[('oe', OrdinalEncoder(), [2]), ('ohe', OneHotEncoder(), [0])], remainder='passthrough')
    ctz
    return (ctz,)


@app.cell
def _(Z, ctz):
    ctz.fit_transform(Z)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Splitting the dataset into training set and test set""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""[https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)""")
    return


@app.cell
def _():
    from sklearn.model_selection import train_test_split
    return (train_test_split,)


@app.cell
def _(X_2, train_test_split, y_1):
    (X_train, X_test, y_train, y_test) = train_test_split(X_2, y_1, train_size=0.8, random_state=123)
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Print `y_train` which is $80\%$ of the target variable `y`""")
    return


@app.cell
def _(y_train):
    print(y_train)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Print `y_test` which is $20\%$ of the target variable `y`""")
    return


@app.cell
def _(y_test):
    print(y_test)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Scaling of features""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""[https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)""")
    return


@app.cell
def _(StandardScaler):
    sc = StandardScaler()
    return (sc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""`X_train` after scaling *(fit & transform, mean $\mu$ & standard deviation $\sigma$ are stored to be later used to transform the test set)*""")
    return


@app.cell
def _(X_train, sc):
    X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
    print(X_train)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""`X_test` after scaling *(only transform)*""")
    return


@app.cell
def _(X_test, sc):
    X_test[:, 3:] = sc.transform(X_test[:, 3:])
    print(X_test)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
