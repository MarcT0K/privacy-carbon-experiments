FROM redis:7.4.1-bookworm
USER root
RUN apt-get update && apt-get upgrade -y &&\
    apt-get install -y git default-jdk python3 python3-pip pipx telnet
WORKDIR /SWiSSSE
RUN pipx install nltk
CMD ["redis-cli -h localhost -p 6379"]