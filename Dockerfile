FROM --platform=${BUILDPLATFORM} python:latest

EXPOSE 1234

ENV USER ml
ENV USER_HOME_DIR /home/${USER}
ENV WORKING_DIR ${USER_HOME_DIR}/repo

RUN python3 -m pip install jupyterlab ipython numpy matplotlib scipy pandas sklearn tensorflow keras

RUN useradd --create-home --shell /bin/bash ${USER} \
	&& chown -R ${USER} ${USER_HOME_DIR} \
	&& mkdir -p ${WORKING_DIR}

USER ${USER}

WORKDIR ${WORKING_DIR}

COPY ./codes/datasets/ ./datasets/

COPY ./codes/ /codes/

# Default command: Jupyter Notebook
CMD jupyter notebook --NotebookApp.token=''  --ip 0.0.0.0 --port 1234 --allow-root --no-browser

