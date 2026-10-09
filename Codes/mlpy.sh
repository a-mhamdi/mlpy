#!/usr/bin/env bash
set -e

BASE=/home/isetbz/mlpy

evince "$BASE/PDF-Files/An Introduction To ML.pdf" &
evince "$BASE/PDF-Files/Lab-ML.pdf" &

# Remove ALL containers
docker ps -aq | xargs -r docker rm -f

cd "$BASE/Docker"
docker compose up -d
bash "$BASE/dcp.sh"

sleep 3
firefox --private-window localhost:2468 
