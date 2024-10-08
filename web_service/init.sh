#!/bin/bash

sudo apt-get update

sudo apt install python3-pip

#install docker
sudo apt install docker.io

#start docker service
sudo systemctl start docker
sudo systemctl enable docker

#install docker compose
sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose


#build and run docker containers
sudo docker-compose up --build
