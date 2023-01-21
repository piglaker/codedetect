# My run.sh
d=`date`
datetime=${d// /-}

# Automatically search available gpus
echo "Searching available gpus..."

gpu_memory=(`nvidia-smi -q -d Memory |grep -A4 GPU|grep Free  | awk -F" "    '{ print $3 }'`)

NUM_GPUS=1

count=0

gtx1080=10240
gtx3090=23480

available_gpus=""

if [ ! -n "$5" ] ;then
    BATCH_SIZE=48 #0.005 0.05 
else
    BATCH_SIZE=$5
fi

for i in "${!gpu_memory[@]}";   
do   
    if [ "${gpu_memory[$i]}" -gt "$gtx3090" ]
    then
        available_gpus="$available_gpus$i,"
        let count+=1
    fi
    
    if [ "${gpu_memory[$i]}" -gt "$gtx3090" ]
    then
        batch_size=128
    fi

    if [ $count -ge $NUM_GPUS ] 
    then
        break
    fi 
done  

if [ $count -lt $NUM_GPUS ]
then 
    echo "Error: No enough GPUs!"
    exit
fi

echo "Use GPUs: "${available_gpus}


TASK="codedetect"
MODEL_NAME="textcnn"

if [ ! -d "./logs/${TASK}/${DATASET}" ]; then
    mkdir ./logs/${TASK}/${DATASET}
fi

PRETRAINED_NAME=bert # pretrain bert type

EPOCH=10
STRATEGY=epoch
EVAL_STEPS=2300
LEARNING_RATE=7e-5
WEIGHT_DECAY=0.02
WARMUP_STEPS=2500
CKPT_LIMIT=1

if [ ! -n "$4" ] ;then
    SEED=3471 #0.005 0.05 
else
    SEED=$4
fi

VALUE=1

name=${MODEL_NAME}"_eval"${EVAL_DATASET}"_epoch"${EPOCH}"_bs"${BATCH_SIZE}"_seed"${SEED}

OUTPUT_DIR=./tmp/${DATASET}//${PRETRAINED_NAME}/${name}

LOG_PATH=logs/${TASK}/${DATASET}/${MODEL_NAME}/${PRETRAINED_NAME}/${name}.log

#seed 153603 27 3472
#lr  5e-5 7e-5 6e-5

rm Recent_Note.log
echo "LOG_PATH: "${LOG_PATH}
ln -s ${LOG_PATH} Recent_Note.log

CUDA_VISIBLE_DEVICES=${available_gpus} OMP_NUM_THREADS=${VALUE} CUDA_LAUNCH_BLOCKING=0 torchrun --nproc_per_node=${NUM_GPUS} --master_port 6501 --nnodes=1 --node_rank=0 \
    main.py \
    --sharded_ddp zero_dp_2 \
    --seed ${SEED} \
    --do_train \
    --do_eval \
    --do_predict \
    --fp16 False \
    --disable_tqdm False \
    --dataloader_num_workers 0 \
    --output_dir ${OUTPUT_DIR} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --eval_accumulation_steps 1 \
    --num_train_epochs ${EPOCH} \
    --evaluation_strategy ${STRATEGY} \
    --save_strategy ${STRATEGY} \
    --load_best_model_at_end \
    --dataloader_pin_memory True \
    --metric_for_best_model Acc \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay ${WEIGHT_DECAY} \
    --warmup_steps ${WARMUP_STEPS} \
    --save_steps ${EVAL_STEPS} \
    --eval_steps ${EVAL_STEPS} \
    --save_total_limit ${CKPT_LIMIT} \
    --model_name ${MODEL_NAME} \
    --num_gpus ${NUM_GPUS} \
    --pretrained_name ${PRETRAINED_NAME} \
    --log_path ${LOG_PATH} \
> note.log 2>&1

#cat tmp.log

echo "Finish training"
echo ${LOG_PATH}
