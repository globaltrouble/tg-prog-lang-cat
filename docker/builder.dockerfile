FROM debian:10

RUN apt-get update
RUN apt-get install -yqq build-essential python3 gcc cmake libre2-dev

WORKDIR /app

ENTRYPOINT ["./build.py", "--build-dir", "/app/build-docker"]
