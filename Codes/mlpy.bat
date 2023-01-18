start "" c:\mlpy\"Lab-Machine Learning (L3)".pdf
::LAUNCH CHROME
start chrome /incognito 192.168.99.100:1357
::START DOCKER DEAMON
docker-machine start
::RUN A CONTAINER "l3" FROM IMAGE "mlpy"
docker run --rm --name l3 -p 1357:1357 mlpy
::EXIT UPON COMPLETION
exit

