FROM python:3.5

ENV PYTHONUNBUFFERED 1

# Requirements have to be pulled and installed here, otherwise caching won't work
COPY ./requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
WORKDIR /app
