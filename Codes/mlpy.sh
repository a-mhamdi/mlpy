#!/usr/bin/bash

evince /home/isetbz/mlpy/PDF-Files/'An Introduction To ML'.pdf &
evince /home/isetbz/mlpy/PDF-Files/Lab-ML.pdf &
docker ps -aq | xargs -r docker stop | xargs -r docker rm &&
cd /home/isetbz/mlpy/Docker && 
docker-compose down && 
docker-compose up -d && 
bash /home/student/Desktop/mlpy/dcp.sh &&
cd .. && 
firefox --private-window localhost:2468  # 1357
