FROM ubuntu:latest
EXPOSE 1234
RUN apt-get update \
	&& apt-get install -y apt-utils nano vim \
	&& apt-get install -y python3 python3-pip

RUN python3 -m pip install jupyterlab ipython numpy matplotlib scipy pandas \
	&& python3 -m pip install sklearn \
	&& python3 -m pip install tensorflow keras\

RUN useradd --create-home --shell /bin/bash ml

RUN mkdir /home/ml/repo/ 
WORKDIR /home/ml/repo/

COPY sl/datasets/ ./datasets/
COPY lab/datasets/* ./datasets/*

COPY sl/ /abmvol/sl/ 
COPY lab/ /abmvol/lab/

CMD jupyter notebook --NotebookApp.token=''  --ip 0.0.0.0 --port 1234 --allow-root --no-browser

