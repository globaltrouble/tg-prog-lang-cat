FROM debian:10

RUN apt-get update
RUN apt-get install -yqq build-essential python3 gcc cmake

RUN mkdir /app

WORKDIR /app/bin

ENTRYPOINT ["/app/bin/run-tglang.py"]
