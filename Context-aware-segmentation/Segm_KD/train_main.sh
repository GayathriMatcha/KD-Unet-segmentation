MODEL_TYPE='studentSFTN'
# ACC_FACTOR='4x'
BASE_PATH='/media/hticimg/data1/Data/MRI'
MODEL='Segm'
OBJECT_TYPE='cardiac'

BATCH_SIZE=4
NUM_EPOCHS=150
DEVICE='cuda:0'

EXP_DIR='/home/hticimg/gayathri/Context-aware-segmentation/Segm_KD/exp/'${MODEL}'_'${MODEL_TYPE}

TRAIN_PATH=${BASE_PATH}'/cardiac_mri_acdc_dataset/train/'
VALIDATION_PATH=${BASE_PATH}'/cardiac_mri_acdc_dataset/test/'

TEACHER_CHECKPOINT='/home/hticimg/gayathri/Context-aware-segmentation/Segm_KD/exp/Segm_teacherSFTN2/best_model.pt'

echo python train_main.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp_dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --object_type ${OBJECT_TYPE} --model_type ${MODEL_TYPE} --teacher_checkpoint ${TEACHER_CHECKPOINT}

python train_main.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp_dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --object_type ${OBJECT_TYPE} --model_type ${MODEL_TYPE} --teacher_checkpoint ${TEACHER_CHECKPOINT}