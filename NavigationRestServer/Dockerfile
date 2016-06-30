From ubuntu:14.04
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
COPY . /usr/src/app/
RUN mv graph-tool.list /etc/apt/sources.list.d && \
    apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    apt-get update && apt-get install -y --force-yes libev-dev python-pip python-graph-tool && \
    pip install -r requirements.txt
EXPOSE 80
CMD ["python", "/usr/src/app/navigator.py"]
