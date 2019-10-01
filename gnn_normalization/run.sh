#!/usr/bin/env bash

datetime=`date +"%y%m%d_%H%M%S"`
seed=$RANDOM
iteration=100
trial=200

mkdir -p log/$datetime
git rev-parse HEAD > log/$datetime/git_commit.txt
cp $0 log/$datetime/
echo $seed > log/$datetime/seed.txt

s=10.0
gpu=3

for c in 1 3 5 7 9
do
    log_dir=log/$datetime/$c/$s
    mkdir -p $log_dir
    if [ "s" = "unnormalized" ]
    then
	python app/train.py -c $c -u 500 -i $iteration -l 0.01 -t $trial -p 3 -g $gpu -d $dataset -s $seed > $log_dir/stdout.txt 2> $log_dir/stderr.txt &
    else
	python app/train.py -c $c -u 500 -i $iteration -l 0.01 -t $trial -p 3 -g $gpu -I $s -n $s -d noisy-pubmed -s $seed > $log_dir/stdout.txt 2> $log_dir/stderr.txt &
    fi
    wait
done
