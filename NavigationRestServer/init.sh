#!/bin/bash
set -e
docker build -t navi-image .
docker run -d --name navi-container -p 6080:80 navi-image
