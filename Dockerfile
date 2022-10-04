FROM python:latest
EXPOSE 1234

ENV USER ml
ENV USER_HOME_DIR /home/${USER}
ENV WORKING_DIR ${USER_HOME_DIR}/repo

RUN echo "apt-get update \
	&& apt-get install -y apt-utils nano vim \
	&& apt-get install -y python3 python3-pip"

RUN python3 -m pip install jupyterlab ipython numpy matplotlib scipy pandas sklearn tensorflow keras

RUN useradd --create-home --shell /bin/bash ${USER} \
	&& mkdir -p ${WORKING_DIR}
	
WORKDIR ${WORKING_DIR}

COPY sl/datasets/ lab/datasets/ ./datasets/

COPY sl /abmvol/
COPY lab /abmvol/

CMD jupyter notebook --NotebookApp.token=''  --ip 0.0.0.0 --port 1234 --allow-root --no-browser

