BASE_PATH='/media/hticimg/data1/Data/MRI'
MODEL='Segm'
DATASET_TYPE='cardiac_mri_acdc_dataset'
MODEL_TYPE='kd_SP' #student,kd,teacherSFTN
MASK='cartesian'

ACC_FACTOR='4x'
BATCH_SIZE=1
DEVICE='cuda:0'


VALIDATION_PATH=${BASE_PATH}'/'${DATASET_TYPE}'/test/'


# CHECKPOINT='/home/hticimg/gayathri/Context-aware-segmentation/KD_seg/cardiac_DC'${MODEL_TYPE}'UNet/models_local/150.pt'
# OUT_DIR='/home/hticimg/gayathri/Context-aware-segmentation/KD_seg/cardiac_DC'${MODEL_TYPE}'UNet/models_local/results'


CHECKPOINT='/home/hticimg/gayathri/Context-aware-segmentation/Segm_KD/exp/'${MODEL}'_'${MODEL_TYPE}'/best_model.pt'
OUT_DIR='/home/hticimg/gayathri/Context-aware-segmentation/Segm_KD/exp/'${MODEL}'_'${MODEL_TYPE}'/results'

echo python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --model_type ${MODEL_TYPE} --mask_type ${MASK}
python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --model_type ${MODEL_TYPE} --mask_type ${MASK} 
