FROM ubuntu:19.04
# FROM ubuntu:18.04

# backup old sources.list
#RUN mv /etc/apt/sources.list /etc/apt/sources.list.bk

#ADD sources.list /etc/apt/sources.list
WORKDIR /code
COPY . /code
# RUN apk add --no-cache gcc musl-dev linux-headers make automake g++ subversion python3-dev
RUN apt-get -qqy update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip  && \ 
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
# RUN python -m pip install -U --force-reinstall pip
RUN pip3 install -r requirements.txt --no-cache-dir

# CMD gunicorn -b 0.0.0.0:8070 -k gevent webapp:wsgiapp

ENTRYPOINT gunicorn -b 0.0.0.0:80 -k gevent webapp:wsgiapp