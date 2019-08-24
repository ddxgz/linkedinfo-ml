FROM ubuntu:19.04

# backup old sources.list
#RUN mv /etc/apt/sources.list /etc/apt/sources.list.bk

#ADD sources.list /etc/apt/sources.list

RUN apt-get -qqy update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip 

# RUN mkdir /code
ADD . /code
# ADD ../requirements.txt /code/

# for copy requirements.txt to container
# COPY . /server/docker-compose

WORKDIR /code

RUN pip3 install -r requirements.txt

CMD gunicorn -b 0.0.0.0:8070 -k gevent webapp:wsgiapp
