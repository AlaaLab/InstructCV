#!/bin/sh

TRAIN_SIZE=10000
VAL_SIZE=500
TEST_SIZE=500

for nblocks in 4 8 12 
do

   echo "Creating a maze with $nblocks blocks"
   python scripts/construct_maze.py --nx $nblocks --ny $nblocks --n_train $TRAIN_SIZE --n_val $VAL_SIZE --n_test $TEST_SIZE
       
done
