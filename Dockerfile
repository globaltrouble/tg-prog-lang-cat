FROM debian:10

RUN apt-get update
RUN apt-get install -yqq python3 gcc cmake

WORKDIR /app

ENTRYPOINT ["./build.py", "--build-dir", "/app/build-docker"]
