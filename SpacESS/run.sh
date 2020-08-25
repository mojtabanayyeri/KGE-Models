#!/bin/sh

#python -u -c 'import torch; print(torch.__version__)'

CODE_PATH=codes
DATA_PATH=data
SAVE_PATH=models

#The first four parameters must be provided
MODE=$1
MODEL=$2
DATASET=$3
GPU_DEVICE=$4
SAVE_ID=$5

FULL_DATA_PATH=$DATA_PATH/$DATASET
SAVE=$SAVE_PATH/"$MODEL"_"$DATASET"_"$SAVE_ID"

#Only used in training
BATCH_SIZE=$6
NEGATIVE_SAMPLE_SIZE=$7
HIDDEN_DIM=$8
GAMMA=$9
ALPHA=${10}
LEARNING_RATE=${11}
MAX_STEPS=${12}
TEST_BATCH_SIZE=${13}
GAMMA1=${14}
DIFF=${15}
OPT=${16}

if [ $MODE == "grid" ]
then
    HIDDEN_DIM=$7
    GAMMA=$8
    ALPHA=$9
    LEARNING_RATE=${10}
    TEST_BATCH_SIZE=${11}

    CUDA_VISIBLE_DEVICES=$GPU_DEVICE python3 -u $CODE_PATH/run.py --cuda \
    --do_grid \
    --do_valid \
    --data_path $FULL_DATA_PATH \
    --model $MODEL \
    -b $BATCH_SIZE -d $HIDDEN_DIM \
    -g $GAMMA -a $ALPHA -adv \
    -lr $LEARNING_RATE \
    --log_steps 5000 \
    --valid_steps 10000 \
    -save $SAVE --test_batch_size $TEST_BATCH_SIZE \
    ${12} ${13} ${14} ${15}



elif [ $MODE == "train" ]
then

echo "Start Training......"

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python3 -u $CODE_PATH/run.py --do_train \
    --cuda \
    --do_test \
    --data_path $FULL_DATA_PATH \
    --model $MODEL \
    -n $NEGATIVE_SAMPLE_SIZE -b $BATCH_SIZE -d $HIDDEN_DIM \
    -g $GAMMA -a $ALPHA -adv \
    -lr $LEARNING_RATE --max_steps $MAX_STEPS \
    -save $SAVE --test_batch_size $TEST_BATCH_SIZE \
    --gamma1 $GAMMA1 --diff $DIFF \
    --opt $OPT \
    --valid_steps 25000 \
    ${17} ${18} ${19} ${20} ${21} 

elif [ $MODE == "valid" ]
then

echo "Start Evaluation on Valid Data Set......"

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python3 -u $CODE_PATH/run.py --do_valid --cuda -init $SAVE
    
elif [ $MODE == "test" ]
then

echo "Start Evaluation on Test Data Set......"

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python3 -u $CODE_PATH/run.py --do_test --cuda -init $SAVE

elif [ $MODE == "experiment" ]
then
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python3 -u $CODE_PATH/run.py --do_experiment \
    --cuda \
    --do_valid \
    --data_path $FULL_DATA_PATH \
    --model $MODEL \
    -n $NEGATIVE_SAMPLE_SIZE -b $BATCH_SIZE -d $HIDDEN_DIM \
    -g $GAMMA -a $ALPHA -adv \
    -lr $LEARNING_RATE --max_steps $MAX_STEPS \
    --log_steps 2000 \
    -save $SAVE --test_batch_size $TEST_BATCH_SIZE \
    --gamma1 $GAMMA1 \
    --diff $DIFF \
    --opt $OPT \
    ${17} ${18} ${19} ${20} 

else
   echo "Unknown MODE" $MODE
fi
