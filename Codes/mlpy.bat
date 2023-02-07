cd %USERPROFILE%\mlpy

start "" "Lab-Machine Learning (L3)".pdf
::LAUNCH CHROME
start chrome /incognito 192.168.99.100:1357
::START DOCKER DEAMON
docker-machine start
::RUN "docker compose down" (CHECK IS THERE ANY RUNNING CONTAINER FROM PREVIOUS SESSION)
docker-compose down
::RUN "docker compose up"
docker-compose up -d
::EXIT UPON COMPLETION
exit

