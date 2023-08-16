BASE_PATH='/media/hticimg/data1/Data/MRI/datasets/calgary_dataset'
MODEL='attention_imitation'
DATASET_TYPE='calgary'
MODEL_TYPE='kdSFTN_fsp'

ACC_FACTOR='4x'
TARGET_PATH=${BASE_PATH}'/validation/acc_'${ACC_FACTOR}
PREDICTIONS_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/sr2/'${MODEL}'_'${MODEL_TYPE}'/results'
REPORT_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/sr2/'${MODEL}'_'${MODEL_TYPE}'/'
python evaluate.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} 