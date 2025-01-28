#!bin/bash
# shellcheck disable=SC2164
cd config_files/
javac -d ./server/bin/ --release 8 -classpath ".:./server/jedis-3.3.0.jar:" ./server/src/Server.java
cd ../plaintext_benchmark/server/bin
nohup java -cp ".:../jedis-3.3.0.jar" Server &
cd ../../
javac -d ./client/bin/ --release 8 ./client/src/*.java ./client/src/client/*.java ./client/src/parser/*.java
cd client/bin
java Controller ../../../config_files/plaintext.conf
