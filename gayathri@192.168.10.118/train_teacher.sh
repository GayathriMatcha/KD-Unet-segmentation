

MODEL_TYPE='teacher'
# ACC_FACTOR='4x'
BASE_PATH='/media/hticimg/data1/Data/MRI'
MODEL='attention_imitation'
DATASET_TYPE='knee'
MASK='cartesian'
ACC_FACTOR='4x'


BATCH_SIZE=4
NUM_EPOCHS=150
DEVICE='cuda:0'

# EXP_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'_'${MODEL_TYPE}
EXP_DIR='/home/hticimg/gayathri/reconstruction/exp/'${DATASET_TYPE}'/'${MASK}'/acc_'${ACC_FACTOR}'/'${MODEL}'_'${MODEL_TYPE}

# TRAIN_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/train/acc_'${ACC_FACTOR}
# VALIDATION_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/validation/acc_'${ACC_FACTOR}
# USMASK_PATH=${BASE_PATH}'/KD-MRI/us_masks/'${DATASET_TYPE}

TRAIN_PATH=${BASE_PATH}'/datasets/'
VALIDATION_PATH=${BASE_PATH}'/datasets/'
USMASK_PATH=${BASE_PATH}'/usmasks/'
# TEACHER_WTS='best_model.pt'

echo python train_base_model.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp_dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --model_type ${MODEL_TYPE}
python train_base_model.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp_dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --model_type ${MODEL_TYPE} --mask_type ${MASK}
