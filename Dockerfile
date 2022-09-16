FROM ubuntu:latest
EXPOSE 1234
RUN apt-get update \
	&& apt-get install -y apt-utils nano vim
RUN apt-get install -y python3 python3-pip

# RUN python3 -m venv /opt/venv
### Make sure we use the virtualenv:
# ENV PATH="/opt/venv/bin:$PATH"

### """ NOT WORKING """ ###
# ENV PATH=$PATH:$HOME/.local/bin
# RUN echo "export PATH=\"$PATH:$HOME/.local/bin\"" >> .bashrc
# RUN source .bashrc

RUN python3 -m pip install jupyterlab ipython numpy matplotlib scipy pandas
RUN python3 -m pip install sklearn
RUN python3 -m pip install tensorflow keras

RUN useradd --create-home --shell /bin/bash ml
# RUN usermod -a -G sudo ml

RUN mkdir /home/ml/repo/ 
WORKDIR /home/ml/repo/

COPY sl/datasets/ ./datasets/
# RUN chmod 555 -R datasets/

COPY sl/ /abmvol/sl/ 
COPY lab/ /abmvol/lab/

# USER ml

CMD jupyter notebook --NotebookApp.token=''  --ip 0.0.0.0 --port 1234 --allow-root --no-browser

### CMD IPYTHON
# CMD python3 -m IPython

### BUILD
# docker build -t pyml:latest .

### PUSH
# docker tag pyml:latest abmhamdi/pyml:1.0
# docker push abmhamdi/pyml:1.0

