FROM python:3.8-slim-buster

# set the working directory in the container
WORKDIR /home/canonical

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1 # Prevents Python from writing pyc files to disc
ENV PYTHONUNBUFFERED 1 # Prevents Python from buffering stdout and stderr

RUN apt-get update \
&& apt-get install gcc -y \
&& apt-get clean

RUN apt-get -y install libc-dev
RUN apt-get -y install build-essential
RUN pip install --upgrade pip

# copy the requirements.txt file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip install -r requirements.txt

# copy the dependencies file to the working directory
COPY data/* ./data/
COPY static/css/* ./static/css/
COPY static/images/* ./static/images/
COPY templates/* ./templates/
COPY operations/* ./operations/
COPY notebooks/* ./notebooks/
COPY models/* ./models/
COPY logs/ ./logs/

COPY main.py .

# command to run on container start
CMD [ "python", "main.py" ]

