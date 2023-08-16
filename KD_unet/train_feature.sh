BASE_PATH='/media/hticimg/data1/Data/MRI'
MODEL='attention_imitation'
DATASET_TYPE='mrbrain_t1'
MODEL_TYPE='featureSFTN'
ACC_FACTOR='4x'
MASK='cartesian'

BATCH_SIZE=4
NUM_EPOCHS=150
DEVICE='cuda:0'


EXP_DIR='/home/hticimg/gayathri/KD_unet/exp/'${DATASET_TYPE}'/'${MASK}'/acc_'${ACC_FACTOR}'/'${MODEL}'_'${MODEL_TYPE}
TRAIN_PATH=${BASE_PATH}'/datasets/'
VALIDATION_PATH=${BASE_PATH}'/datasets/'
USMASK_PATH=${BASE_PATH}'/usmasks/'

TEACHER_CHECKPOINT='/home/hticimg/gayathri/KD_unet/exp/'${DATASET_TYPE}'/'${MASK}'/acc_'${ACC_FACTOR}'/'${MODEL}'_teacherSFTN/best_model.pt'
STUDENT_CHECKPOINT='/home/hticimg/gayathri/KD_unet/exp/'${DATASET_TYPE}'/'${MASK}'/acc_'${ACC_FACTOR}'/'${MODEL}'_studentSFTN/best_model.pt'


echo python train_feature.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --teacher_checkpoint ${TEACHER_CHECKPOINT} --mask_type ${MASK} --student_checkpoint ${STUDENT_CHECKPOINT}

python train_feature.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --teacher_checkpoint ${TEACHER_CHECKPOINT} --mask_type ${MASK} --student_checkpoint ${STUDENT_CHECKPOINT}

