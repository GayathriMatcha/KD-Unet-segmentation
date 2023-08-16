BASE_PATH='/media/hticimg/data1/Data/MRI'
MODEL='prune_kd'
DATASET_TYPE='mrbrain_t1'
MODEL_TYPE='student' #teacher,student,kd
MASK='cartesian'

ACC_FACTOR='4x'
BATCH_SIZE=1
DEVICE='cuda:0'

VALIDATION_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/'${MASK}'/validation/acc_'${ACC_FACTOR}
USMASK_PATH=${BASE_PATH}'/usmasks/'

CHECKPOINT=${BASE_PATH}'/exp/prune_kd/'${DATASET_TYPE}'/'${MASK}'/acc_'${ACC_FACTOR}'/'${MODEL}'_'${MODEL_TYPE}'_0.95/best_model.pt'

OUT_DIR=${BASE_PATH}'/exp/prune_kd/'${DATASET_TYPE}'/'${MASK}'/acc_'${ACC_FACTOR}'/'${MODEL}'_'${MODEL_TYPE}'_0.95/results'

echo python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --model_type ${MODEL_TYPE} --usmask_path ${USMASK_PATH} --mask_type ${MASK}
python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --model_type ${MODEL_TYPE} --usmask_path ${USMASK_PATH} --mask_type ${MASK}

