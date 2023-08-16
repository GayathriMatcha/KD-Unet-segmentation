BASE_PATH='/media/hticimg/data1/Data/MRI'
MODEL='prune_kd'
DATASET_TYPE='mrbrain_t1'
MODEL_TYPE='kd_at3'
ACC_FACTOR='4x'
MASK='cartesian'

BATCH_SIZE=4
NUM_EPOCHS=150
DEVICE='cuda:0'
SPARSITY=0.9

EXP_DIR=${BASE_PATH}'/exp/prune_kd/'${DATASET_TYPE}'/'${MASK}'/acc_'${ACC_FACTOR}'/'${MODEL}'_'${MODEL_TYPE}
TRAIN_PATH=${BASE_PATH}'/datasets/'
VALIDATION_PATH=${BASE_PATH}'/datasets/'
USMASK_PATH=${BASE_PATH}'/usmasks/'

TEACHER_CHECKPOINT=${BASE_PATH}'/exp/prune_kd/'${DATASET_TYPE}'/'${MASK}'/acc_'${ACC_FACTOR}'/'${MODEL}'_teacher/best_model.pt'
STUDENT_CHECKPOINT=${BASE_PATH}'/exp/prune_kd/'${DATASET_TYPE}'/'${MASK}'/acc_'${ACC_FACTOR}'/'${MODEL}'_feature_at/best_model.pt'

echo python train_kd.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp_dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --teacher_checkpoint ${TEACHER_CHECKPOINT} --student_checkpoint ${STUDENT_CHECKPOINT} --mask_type ${MASK} --student_pretrained # --imitation_required --sparsity ${SPARSITY}
python train_kd.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp_dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --teacher_checkpoint ${TEACHER_CHECKPOINT} --student_checkpoint ${STUDENT_CHECKPOINT} --mask_type ${MASK} --student_pretrained #--imitation_required --sparsity ${SPARSITY}