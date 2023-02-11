cd %USERPROFILE%\mlpy

start "" "Lab-Machine Learning (L3)".pdf
REM LAUNCH CHROME
start chrome /incognito 192.168.99.100:1357
REM START DOCKER DEAMON
docker-machine start
REM RUN "docker compose down" (CHECK IS THERE ANY RUNNING CONTAINER FROM PREVIOUS SESSION)
docker-compose down
REM RUN "docker compose up"
docker-compose up -d

echo Please, type `docker-compose down` before leaving.

REM EXIT UPON COMPLETION
:: exit
