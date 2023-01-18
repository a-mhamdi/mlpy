start "" c:\mlpy\"Lab-Machine Learning (L3)".pdf
::LAUNCH CHROME
start chrome /incognito 192.168.99.100:4321
::START DOCKER DEAMON
docker-machine start
::RUN A CONTAINER "l3" FROM IMAGE "mlpy"
docker run --rm --name l3 -p 4321:4321 mlpy
::EXIT UPON COMPLETION
exit

