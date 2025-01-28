#!bin/bash
# shellcheck disable=SC2164
cd swissse_benchmark/server/bin
nohup java -cp ".:../jedis-3.3.0.jar" Server &
cd ../../client/bin/
#java Controller ../../../config_files/SWiSSSE_test.conf
java Controller ../../../config_files/SWiSSSE_search.conf
