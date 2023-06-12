#!/bin/bash

DN=5
N0=10
NF=25
NXARRAY=($(seq $N0 $DN $NF))  # START, STEP, LAST
NELEMENTS=${#NXARRAY[@]}



conda activate fenics2020
rm *.dat
for ((i=0; i <$NELEMENTS; i++)); do
	python spectral_plots.py ${NXARRAY[${i}]} & process_id=$!
	wait ${process_id}
	cd data
	mkdir N${NXARRAY[${i}]}
	cd N${NXARRAY[${i}]}
	rm *.dat
	mv ../../*.dat ./
	cd ../..
done

conda deactivate
