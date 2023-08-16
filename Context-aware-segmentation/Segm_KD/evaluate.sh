BASE_PATH='/media/hticimg/data1/Data/MRI'
MODEL='Segm'
DATASET_TYPE='cardiac_mri_acdc_dataset'
MODEL_TYPE='kd_SP' #student,kd,teacherSFTN
MASK='cartesian'

ACC_FACTOR='4x'
BATCH_SIZE=1
DEVICE='cuda:0'

TARGET_PATH=${BASE_PATH}'/'${DATASET_TYPE}'/test/'

# PREDICTIONS_PATH='/home/hticimg/gayathri/Context-aware-segmentation/KD_seg/cardiac_DCStudentUNet_SFTN/models_local/results'
# REPORT_PATH='/home/hticimg/gayathri/Context-aware-segmentation/KD_seg/cardiac_DCStudentUNet_SFTN/models_local/'

PREDICTIONS_PATH='/home/hticimg/gayathri/Context-aware-segmentation/Segm_KD/exp/'${MODEL}'_'${MODEL_TYPE}'/results'
REPORT_PATH='/home/hticimg/gayathri/Context-aware-segmentation/Segm_KD/exp/'${MODEL}'_'${MODEL_TYPE}

echo python evaluate.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} 
python evaluate.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} 