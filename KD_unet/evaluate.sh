BASE_PATH='/media/hticimg/data1/Data/MRI'
MODEL='attention_imitation'
DATASET_TYPE='mrbrain_t1'
MODEL_TYPE='student' #student,kd
MASK='cartesian'

ACC_FACTOR='4x'
BATCH_SIZE=1
DEVICE='cuda:0'

TARGET_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/'${MASK}'/validation/acc_'${ACC_FACTOR}

PREDICTIONS_PATH='/home/hticimg/gayathri/KD_unet/exp/'${DATASET_TYPE}'/'${MASK}'/acc_'${ACC_FACTOR}'/'${MODEL}'_'${MODEL_TYPE}'/results'
REPORT_PATH='/home/hticimg/gayathri/KD_unet/exp/'${DATASET_TYPE}'/'${MASK}'/acc_'${ACC_FACTOR}'/'${MODEL}'_'${MODEL_TYPE}'/'

echo python evaluate.py --target_path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} 
python evaluate.py --target_path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH}  


 