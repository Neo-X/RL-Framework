#!/bin/bash

# docker build --no-cache -f Dockerfile_trl -t rlframe:latest .
docker build --no-cache -f Dockerfile_trl -t rlframe:latest .
docker tag rlframe:latest gberseth/rlframe:latest
docker push gberseth/rlframe:latest

