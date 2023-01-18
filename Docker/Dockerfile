# Base image
FROM python:3.9
# Metadata
LABEL org.opencontainers.image.authors="A. Mhamdi"
LABEL version="latest"
# Export port 1357
EXPOSE 1357
# Settings
ENV USER=mhamdi
ENV USER_HOME_DIR=/home/${USER}
ARG WORKING_DIR=${USER_HOME_DIR}/repo/
# Add user and set ownership
RUN useradd --create-home --shell /bin/bash ${USER} \
	&& chown -R ${USER} ${USER_HOME_DIR} \
	&& mkdir -p ${WORKING_DIR}
# Change user & current working directory
USER ${USER}
WORKDIR ${WORKING_DIR}
# Copy needed files and datasets
COPY ./datasets/ ${WORKING_DIR}/datasets/
# CREATE PYTHON VIRTUAL ENV
ENV VIRTUAL_ENV=${USER_HOME_DIR}/venv
RUN python3 -m venv ${VIRTUAL_ENV}
# Activate venv
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"
# Update `pip` NOT(REQUIRED FOR PYTHON 3.10)
RUN python3 -m pip install --upgrade pip
# Install packages
RUN python3 -m pip install ipython jupyterlab
RUN python3 -m pip install numpy matplotlib scipy pandas seaborn
RUN python3 -m pip install --use-pep517 sklearn tensorflow keras
# Default command: Jupyter Lab
CMD jupyter lab --NotebookApp.token=''  --ip 0.0.0.0 --port 1357 --allow-root --no-browser
