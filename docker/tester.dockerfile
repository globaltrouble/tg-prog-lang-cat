FROM debian:10

RUN apt-get update
RUN apt-get install -yqq libstdc++6 libc6 libgcc1 libre2-dev

RUN mkdir /app

WORKDIR /app/bin

ENTRYPOINT ["/app/bin/run-tglang.py"]
