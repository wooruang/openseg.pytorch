#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $SCRIPTPATH
. config.profile

export PYTHONPATH="$PWD":$PYTHONPATH

DATA_DIR="${DATA_ROOT}/ade20k"
SAVE_DIR="${DATA_ROOT}/seg_result/ade20k/"
BACKBONE="hrnet48"
CONFIGS_TEST="configs/ade20k/H_48_D_4_TEST.json"

MODEL_NAME="hrnet_w48_ocr"

CHECKPOINTS_NAME="${MODEL_NAME}_${BACKBONE}_ohem_2"

${PYTHON} -u video.py --configs ${CONFIGS_TEST} \
                       --data_dir ${DATA_DIR} \
                       --backbone ${BACKBONE} \
                       --model_name ${MODEL_NAME} \
                       --checkpoints_name ${CHECKPOINTS_NAME} \
                       --phase test \
                       --gpu 0 \
                       --resume ./checkpoints/ade20k/${CHECKPOINTS_NAME}_latest.pth \
                       --test_dir ${DATA_DIR}/video \
                       --log_to_file n \
                       --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_val_ms \
                       --input_video /home/bogonets/Projects/openseg.pytorch/2020113000407.avi
