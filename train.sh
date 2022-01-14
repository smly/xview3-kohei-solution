#!/bin/bash

echo " ******************** Preprocessing"
# input:
# - data/input/xview3/downloaded/*.tar.gz
# - data/input/xview3/validation.csv
# - data/input/xview3/train.csv
# output:
# - data/working/xview3/preprocess_vh_vv_bathymetry_v2/validation/*.png
# - data/working/xview3/images/ppv6/validation/
# - data/working/xview3/images/ppv6/train/
# - data/working/xview3/images/ppv6/thumb_validation/
# - data/working/xview3/images/ppv6/thumb_train/
PYTHONPATH=. poetry run python -m xd.xview3.preproc

echo " ******************** Training (instance segmentation model)"
# Build instance segmentation models (configs/localization/v13)
# input:
# - data/working/xview3/preprocess_vh_vv_bathymetry_v2/validation/*.png
# output:
# - data/working/xview3/models/v13_iim_ppv2_augv2/fold0/ep59.pth
PYTHONPATH=. poetry run python -m xd.xview3.localization.trainer \
    -c configs/localization/v13_iim_ppv2_augv2.yml \
    -f 0
mkdir -p v13 && cp data/working/xview3/models/v13_iim_ppv2_augv2/fold0/ep59.pth v13/

echo " ******************** Training (classification model)"
# Crop images & Build classification models (configs/vessel_class/v77)
# input:
# - data/working/xview3/images/ppv6/validation/*.png
# - data/working/xview3/images/ppv6/train/*.png
# output:
# - data/working/xview3/images/ppv6/crop_validation/*
# - data/working/xview3/images/ppv6/crop_train/*
for fold_idx in {1..9};
do
    PYTHONPATH=. poetry run python -m xd.xview3.vessel_class.trainer \
        -c configs/vessel_class/v77_r50d_ppv6_v13loc_crop128_augv7.yml \
        -f ${fold_idx}
    mkdir -p v77 && cp data/working/xview3/models/v77_r50d_ppv6_v13loc_crop128_augv7/fold${fold_idx}/v77f${fold_idx}ep9.pth v77/
done

echo " ******************** Training (regression model)"
# Build regression models (configs/length_estimation/v16)
# - PPV2ShipCropTrainValDataset, PPV2ShipCropInferenceDataset
