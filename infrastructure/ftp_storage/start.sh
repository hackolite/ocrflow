#!/bin/bash
source $1
docker build . -t ftp 
docker run --env-file $1 -v $FTP_FOLDER:/home/$FTP_USER:rw -d -p 20:20 -p 21:21 -p 40000-40009:40000-40009 ftp
nohup  ./storage_daemon.sh &

