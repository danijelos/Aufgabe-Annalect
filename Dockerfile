# parent base image
FROM python:3.12-slim

# set work directory
WORKDIR /src/flask_backend

# copy requirements.txt
COPY ./requirements.txt /src/flask_backend/requirements.txt
# Copy the CSV file from the local filesystem to the Docker container
COPY ./data/source_segments_angepasst.csv /app/data/source_segments_angepasst.csv

# install system dependencies
RUN apt-get update \
    # && apt-get -y install gcc make \
    && rm -rf /var/lib/apt/lists/* \
    &&  pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# copy project
COPY ./src/flask_backend.py /src/flask_backend/

# set flask_backend port
EXPOSE 8080

ENTRYPOINT [ "python" ] 

# Run flask_backend.py when the container launches
CMD [ "flask_backend.py","run","--host","0.0.0.0","--port","8080"] 