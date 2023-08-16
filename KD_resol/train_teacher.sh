BASE_PATH='/media/hticimg/data1/Data/MRI/datasets/calgary_dataset'
MODEL='Resol'
DATASET_TYPE='calgary'
MODEL_TYPE='teacher'

BATCH_SIZE=8
NUM_EPOCHS=150
DEVICE='cuda:0'
ACC_FACTOR='4x' # using this to get the fs data in the acc_4x path 
EXP_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/sr3/'${MODEL}'_'${MODEL_TYPE}
TRAIN_PATH=${BASE_PATH}'/train/acc_'${ACC_FACTOR}
VALIDATION_PATH=${BASE_PATH}'/validation/acc_'${ACC_FACTOR}

echo python train_sr_base_model.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --model_type ${MODEL_TYPE}

python train_sr_base_model.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --model_type ${MODEL_TYPE} 