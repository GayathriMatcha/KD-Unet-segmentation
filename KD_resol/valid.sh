BASE_PATH='/media/hticimg/data1/Data/MRI/datasets/calgary_dataset'
MODEL='Resol'
DATASET_TYPE='calgary'
MODEL_TYPE='kdSFTN_FactorTransfer'

ACC_FACTOR='4x'
BATCH_SIZE=1
DEVICE='cuda:0'
DATA_PATH=${BASE_PATH}'/validation/acc_'${ACC_FACTOR}
CHECKPOINT=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/sr3/'${MODEL}'_'${MODEL_TYPE}'/best_model.pt'
OUT_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/sr3/'${MODEL}'_'${MODEL_TYPE}'/results'

echo python valid_sr.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --model_type ${MODEL_TYPE}
python valid_sr.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --model_type ${MODEL_TYPE}