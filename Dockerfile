FROM ubuntu:latest
EXPOSE 1234
RUN apt-get update \
	&& apt-get install -y apt-utils nano vim \
	&& apt-get install -y python3 python3-pip

RUN python3 -m pip install jupyterlab ipython numpy matplotlib scipy pandas sklearn tensorflow keras

RUN useradd --create-home --shell /bin/bash ml \
	&& mkdir /home/ml/repo/ 
	
WORKDIR /home/ml/repo/

COPY sl/datasets/ lab/datasets/ ./datasets/

COPY sl /abmvol/
COPY lab /abmvol/

CMD jupyter notebook --NotebookApp.token=''  --ip 0.0.0.0 --port 1234 --allow-root --no-browser

