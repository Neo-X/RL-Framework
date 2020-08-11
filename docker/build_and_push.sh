#!/bin/bash

docker build -f Dockerfile_trl -t rlframe:latest .
docker tag rlframe:latest gberseth/rlframe:latest
docker push gberseth/rlframe:latest

