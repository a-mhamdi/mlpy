## Machine Learning with Python

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=main&skip_quickstart=true&machine=standardLinux32gb&repo=537615866&devcontainer_path=.devcontainer%2Fdevcontainer.json&geo=EuropeWest)

[![MLPY CI](https://github.com/a-mhamdi/mlpy/actions/workflows/docker-image.yml/badge.svg)](https://github.com/a-mhamdi/mlpy/actions/workflows/docker-image.yml)
[![Docker Version](https://img.shields.io/docker/v/abmhamdi/mlpy?sort=semver)](https://hub.docker.com/r/abmhamdi/mlpy)

This repository contains slides, labs, and code samples for using `Python` to implement some **machine learning** related algorithms. 

### Included Algorithms
The repository includes the implementation of the following algorithms:
>1. Linear Regression
>1. Logistic Regression
>1. k-NN
>1. K-MEANS
>1. ANN

### Prerequisites

Codes run on top of a `Docker` image, ensuring a consistent and reproducible environment. 

<img src="Attention.svg" alt="Attention" width="16"/> You will need to have `Docker` installed on your machine. You can download it from the [Docker website](https://hub.docker.com).

To run the code, you will need to first pull the `Docker` image by running the following command:

```zsh
docker pull abmhamdi/mlpy
```

This may take a while, as it will download and install all necessary dependencies.

### License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
