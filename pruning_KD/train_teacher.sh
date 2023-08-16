MODEL_TYPE='student'
# ACC_FACTOR='4x'
BASE_PATH='/media/hticimg/data1/Data/MRI'
MODEL='prune_kd'
DATASET_TYPE='mrbrain_t1'
MASK='cartesian'
ACC_FACTOR='4x'


BATCH_SIZE=2
NUM_EPOCHS=150
DEVICE='cuda:0'
SPARSITY=0.975

EXP_DIR=${BASE_PATH}'/exp/prune_kd/'${DATASET_TYPE}'/'${MASK}'/acc_'${ACC_FACTOR}'/'${MODEL}'_'${MODEL_TYPE}'_0.975'


TRAIN_PATH=${BASE_PATH}'/datasets/'
VALIDATION_PATH=${BASE_PATH}'/datasets/'
USMASK_PATH=${BASE_PATH}'/usmasks/'

echo python train_base_model.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp_dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --model_type ${MODEL_TYPE} --sparsity ${SPARSITY}
python train_base_model.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp_dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --model_type ${MODEL_TYPE} --mask_type ${MASK} --sparsity ${SPARSITY}