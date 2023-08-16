BASE_PATH='/media/hticimg/data1/Data/MRI/datasets/calgary_dataset'
MODEL='Resol'
DATASET_TYPE='calgary'
MODEL_TYPE='kd_'

ACC_FACTOR='4x'
BATCH_SIZE=1
DEVICE='cuda:0'

TARGET_PATH=${BASE_PATH}'/validation/acc_'${ACC_FACTOR}

PREDICTIONS_PATH='/media/hticimg/data1/Data/MRI/datasets/calgary_dataset/experiments/calgary/sr3/'${MODEL}'_'${MODEL_TYPE}'/results'
REPORT_PATH='/media/hticimg/data1/Data/MRI/datasets/calgary_dataset/experiments/calgary/sr3/'${MODEL}'_'${MODEL_TYPE}'/'

echo python measures.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} 
python measures.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH}  
