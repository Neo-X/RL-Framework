FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

RUN apt-get install -y python3-dev
RUN apt-get update --fix-missing
RUN apt-get update -y
RUN apt-get install -y wget bzip2 ca-certificates git vim
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        premake4 \
        git \
        curl \
        vim 

CMD ["/bin/bash"]
