#!/usr/bin/bash

evince ~/mlpy/Slides-Labs/Lab-ML.pdf & \
evince ~/mlpy/Slides-Labs/'An Introduction To ML'.pdf & \
cd ~/mlpy/Docker && docker-compose down && docker-compose up -d && cd .. && firefox --private-window localhost:1357
