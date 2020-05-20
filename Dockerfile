FROM ubuntu:18.04

RUN apt update
RUN apt install -y git python3.7 python3-pip
WORKDIR /usr/src
RUN git clone https://github.com/zhuoranzoeli/CodeHonestly.git
WORKDIR CodeHonestly
RUN pip3 install -r requirements.txt
RUN ln -s $(which python3) /usr/bin/python
RUN ln -s $(which pip3) /usr/bin/pip
