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
        In unsupervised learning, the algorithm is given a dataset and is asked to learn the underlying structure of the data. The goal is to find patterns or relationships in the data that can be used to group the data points into clusters or to reduce the dimensionality of the data.

        Some examples of unsupervised learning algorithms include:

        1. $K$-Means clustering;
        1. Principal Component Analysis (PCA); and
        1. Autoencoders.

        These algorithms can be used for tasks such as image compression, anomaly detection, and customer segmentation.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## $K$-Means Clustering""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""$K$-Means clustering is a method of unsupervised machine learning used to partition a dataset into $k$ clusters, where $k$ is a user-specified number. The goal of $K$-Means clustering is to minimize the sum of squared distances between the points in each cluster and its centroid.""")
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
    plt.style.use('dark_background')
    plt.rc('figure', figsize=(6, 4))

    from matplotlib import rcParams
    rcParams['font.family'] = 'Comfortaa'
    rcParams['font.size'] = 8
    rcParams['axes.unicode_minus'] = False
    return plt, rcParams


@app.cell(hide_code=True)
def _():
    # Show plots in an interactive format, e.g., zooming, saving, etc
    #%matplotlib widget
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Importing the dataset""")
    return


@app.cell
def _(pd):
    df = pd.read_csv('../Datasets/Mall_Customers.csv')
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


@app.cell
def _(df):
    df.rename(columns={'Annual Income (k$)':'Annual Income', 'Spending Score (1-100)': 'Spending Score'}, inplace=True)
    return


@app.cell
def _(df):
    X = df.drop(columns=['CustomerID', 'Age', 'Gender']).values
    X[:10, :]
    return (X,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Import `KMeans` class""")
    return


@app.cell
def _():
    from sklearn.cluster import KMeans
    return (KMeans,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**OPTIONAL: IF NOT FAMILIAR WITH `KMEANS`, FEEL FREE TO SKIP THE FOLLOWING CELL**""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""---""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**Using the elbow method to find the optimal number of clusters**""")
    return


@app.cell
def _(KMeans, X, plt):
    wcss = []

    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i,
                        init='k-means++', # Init method
                        n_init=5)

        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    plt.plot(range(1, 11), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.grid()
    return i, kmeans, wcss


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""---""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Training the K-Means model on the dataset""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""This code will create a $K$-Means model with $5$ clusters and fit it to the data. It will then make predictions about which cluster each data point belongs to.""")
    return


@app.cell
def _(mo):
    slider = mo.ui.slider(1, 10, 1, 5)
    slider
    return (slider,)


@app.cell
def _(KMeans, X, slider):
    kmeans_1 = KMeans(n_clusters=slider.value, init='k-means++', n_init='auto')
    y_pred = kmeans_1.fit_predict(X)
    return kmeans_1, y_pred


@app.cell
def _(kmeans_1):
    centers = kmeans_1.cluster_centers_
    centers
    return (centers,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Visualizing the clusters""")
    return


@app.cell
def _(X, centers, plt, y_pred):
    fig, ax = plt.subplots()
    scatter =  ax.scatter(X[:, 0], X[:, 1], c=y_pred, s=100)
    legend = ax.legend(*scatter.legend_elements(), title='Clusters')
    ax.add_artist(legend)
    ax.scatter(centers[:, 0], centers[:, 1], c='red', s=200)
    ax.set_title('Clusters of customers')
    ax.set_xlabel('Annual Income')
    ax.set_ylabel('Spending Score')
    ax.grid()
    plt.show()
    return ax, fig, legend, scatter


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Unsupervised learning can be useful when there is no labeled training data available, or when the goal is to discover patterns or relationships in the data rather than to make predictions. However, it can be more difficult to evaluate the performance of unsupervised learning algorithms, as there is no ground truth to compare the predictions to.

        $K$-Means clustering is a fast and efficient method for clustering large datasets, and is often used as a baseline method for comparison with other clustering algorithms. However, it can be sensitive to the initial selection of centroids, and may not always find the optimal clusters if the data is not well-separated or has a non-convex shape. It is also limited to spherical clusters and may not work well for clusters with more complex shapes.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
