#!/usr/bin/env bash


function run () {
    seed=$1
    if [ ! -d "$seed" ]; then
	rm -rf $seed
	mkdir $seed
    fi
    for S in 0.5 1.0 1.2 1.5 2.0 4.0
    do
	if [ "$seed/$S" ]; then
	    rm -rf $seed/$S
	    mkdir $seed/$S
	fi
	python main.py -s $seed -S $S -O $seed/$S > log.txt
	cp $seed/$S/streamplot.pdf $seed/$S.pdf
	mv log.txt $seed/$S
    done
}


seed=15
run $seed


seed=4
run $seed
