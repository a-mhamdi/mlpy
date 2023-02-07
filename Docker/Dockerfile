# Base image
FROM python:3.9
# Metadata
LABEL org.opencontainers.image.authors="A. Mhamdi"
LABEL version="latest"
# Expose port 1357
EXPOSE 1357
# Settings
ENV USERNAME=mhamdi
ENV USER_HOME_DIR=/home/"${USERNAME}"
ARG WORKING_DIR="${USER_HOME_DIR}/repo/"

# Add user and set ownership
RUN useradd -ms /bin/bash  -p $(echo "${USERNAME}" | openssl passwd -1 -stdin) "${USERNAME}"

# Change user & current working directory
USER "${USERNAME}"
WORKDIR "${USER_HOME_DIR}"
# Copy needed files and datasets
RUN mkdir -p "${WORKING_DIR}"
COPY ./Codes/Python/datasets "${WORKING_DIR}"/datasets
# Create Python virtual env
ENV VIRTUAL_ENV="${USER_HOME_DIR}/venv"
RUN python3 -m venv "${VIRTUAL_ENV}"
# Change working directory
WORKDIR "${WORKING_DIR}"
# Activate venv
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"
# Update `pip` NOT(REQUIRED FOR PYTHON 3.10)
RUN python3 -m pip install --upgrade pip
# Install packages: Interactive tools, ML and ANN APIs
RUN python3 -m pip install ipython jupyterlab numpy scipy pandas matplotlib seaborn sklearn tensorflow keras
# Default command: Jupyter Lab
ENTRYPOINT jupyter lab --NotebookApp.token='' --ip 0.0.0.0 --port 1357 --allow-root --no-browser

