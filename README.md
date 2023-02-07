## Machine Learning with Python

This repository contains slides, labs and code examples for using `Python` to implement some **machine learning** related algorithms. Codes run on top of a `Docker` image, ensuring a consistent and reproducible environment.

[![MLPY CI](https://github.com/a-mhamdi/mlpy/actions/workflows/docker-image.yml/badge.svg)](https://github.com/a-mhamdi/mlpy/actions/workflows/docker-image.yml)
[![Docker Version](https://img.shields.io/docker/v/abmhamdi/mlpy?sort=semver)](https://hub.docker.com/r/abmhamdi/mlpy)
[![Docker Pulls](https://img.shields.io/docker/pulls/abmhamdi/mlpy)](https://hub.docker.com/r/abmhamdi/mlpy)
[![Docker Stars](https://img.shields.io/docker/stars/abmhamdi/mlpy)](https://hub.docker.com/r/abmhamdi/mlpy)

To run the code, you will need to first pull the `Docker` image by running the following command:

```
docker pull abmhamdi/mlpy
```

This may take a while, as it will download and install all necessary dependencies.

### Included Algorithms
The repository includes implementation of the following algorithms:
>1. Linear Regression
>1. Logistic Regression
>1. k-NN
>1. K-MEANS
>1. ANN

### Prerequisites
You will need to have `Docker` installed on your machine. You can download it from the [Docker website](https://hub.docker.com).

### License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
