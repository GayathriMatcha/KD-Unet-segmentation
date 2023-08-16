BASE_PATH='/media/hticimg/data1/Data/MRI/datasets/calgary_dataset'
MODEL='attention_imitation'
DATASET_TYPE='calgary'
MODEL_TYPE='kdSFTN_fsp'

BATCH_SIZE=4
NUM_EPOCHS=150
DEVICE='cuda:0'
ACC_FACTOR='4x'

EXP_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/sr2/'${MODEL}'_'${MODEL_TYPE}
TRAIN_PATH=${BASE_PATH}'/train/acc_'${ACC_FACTOR}
VALIDATION_PATH=${BASE_PATH}'/validation/acc_'${ACC_FACTOR}
# USMASK_PATH=${BASE_PATH}'/MRI-KD/us_masks/'${DATASET_TYPE}'/'${ACC_TYPE}

TEACHER_CHECKPOINT=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/sr2/'${MODEL}'_teacherSFTN/best_model.pt'
STUDENT_CHECKPOINT=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/sr2/'${MODEL}'_featureSFTN_fsp/best_model.pt'

echo python train_sr_kd.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE}  --teacher_checkpoint ${TEACHER_CHECKPOINT} --student_checkpoint ${STUDENT_CHECKPOINT} --student_pretrained --imitation_required #--usmask_path ${USMASK_PATH}

python train_sr_kd.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE}  --teacher_checkpoint ${TEACHER_CHECKPOINT} --student_checkpoint ${STUDENT_CHECKPOINT} --student_pretrained --imitation_required #--usmask_path ${USMASK_PATH}