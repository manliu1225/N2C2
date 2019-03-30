#!/bin/bash

start=202
end=287
for ((i=start; i<=end; i++))
do
   echo "process txt: $i"
   ./NER/CliNER_master/cliner predict --txt ./test_tmp/${i}.txt  --out ./test_tm  --model ./NER/CliNER_master/models/silver.crf --format i2b2
done
