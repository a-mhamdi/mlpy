# Machine Learning with Python

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=main&skip_quickstart=true&machine=standardLinux32gb&repo=537615866&devcontainer_path=.devcontainer%2Fdevcontainer.json&geo=EuropeWest)

[![MLPY-CI](https://github.com/a-mhamdi/mlpy/actions/workflows/mlpy.yml/badge.svg)](https://github.com/a-mhamdi/mlpy/actions/workflows/mlpy.yml)
[![Docker Version](https://img.shields.io/docker/v/abmhamdi/mlpy?sort=semver)](https://hub.docker.com/r/abmhamdi/mlpy)
[![Open in Codeanywhere](https://codeanywhere.com/img/open-in-codeanywhere-btn.svg)](https://app.codeanywhere.com/#https://github.com/a-mhamdi/mlpy)

This repository contains slides, labs, and code samples for using `Python` to implement some **machine learning** related algorithms. 

## Included Algorithms
The repository includes the implementation of the following algorithms:
>1. Linear Regression
>1. Logistic Regression
>1. k-NN
>1. K-MEANS
>1. ANN

## Prerequisites

Codes run on top of a `Docker` image, ensuring a consistent and reproducible environment. 

<img src="https://raw.githubusercontent.com/a-mhamdi/mlpy/main/Attention.svg" alt="Attention" width="16"/> You will need to have `Docker` installed on your machine. You can download it from the [Docker website](https://hub.docker.com).

To run the code, you will need to first pull the `Docker` image by running the following command:

```zsh
docker pull abmhamdi/mlpy
```

This may take a while, as it will download and install all necessary dependencies.

## How to control the containers:

* ```docker-compose up -d``` starts the container in detached mode
* ```docker-compose down``` stops and destroys the container

Services can be run by typing the command `docker-compose up`. This will start the `Jupyter Lab` on [http://localhost:2468](http://localhost:2468) and you should be able to use `Python` from within the notebook by starting a new `Python` notebook. You can parallelly start `Marimo` on [http://localhost:1357](http://localhost:1357).

## License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
