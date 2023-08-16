MODEL_TYPE='kd_CC'
# ACC_FACTOR='4x'
BASE_PATH='/media/hticimg/data1/Data/MRI'
MODEL='Segm'
OBJECT_TYPE='cardiac'

BATCH_SIZE=5
NUM_EPOCHS=150
DEVICE='cuda:0'

# EXP_DIR='/home/hticimg/gayathri/reconstruction/exp/'${DATASET_TYPE}'/'${MASK}'/acc_'${ACC_FACTOR}'/'${MODEL}'_'${MODEL_TYPE}
EXP_DIR='/home/hticimg/gayathri/Context-aware-segmentation/Segm_KD/exp/'${MODEL}'_'${MODEL_TYPE}
#'_final'


TRAIN_PATH=${BASE_PATH}'/cardiac_mri_acdc_dataset/train/'
VALIDATION_PATH=${BASE_PATH}'/cardiac_mri_acdc_dataset/test/'

TEACHER_CHECKPOINT='/home/hticimg/gayathri/Context-aware-segmentation/Segm_KD/exp/Segm_teacher/best_model.pt'
# STUDENT_CHECKPOINT='/home/hticimg/gayathri/Context-aware-segmentation/Segm_KD/exp/Segm_studentSFTN/best_model.pt'

echo python train_kd.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp_dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --object_type ${OBJECT_TYPE} --model_type ${MODEL_TYPE} --teacher_checkpoint ${TEACHER_CHECKPOINT} #--student_checkpoint ${STUDENT_CHECKPOINT} #--imitation_required

python train_kd.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp_dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --object_type ${OBJECT_TYPE} --model_type ${MODEL_TYPE} --teacher_checkpoint ${TEACHER_CHECKPOINT} #--student_checkpoint ${STUDENT_CHECKPOINT} #--imitation_required
