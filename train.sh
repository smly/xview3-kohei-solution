#!/bin/bash

# Preprocessing
# - data/working/xview3/preprocess_vh_vv_bathymetry_v2/validation/
# - data/working/xview3/images/ppv6/validation/
# - data/working/xview3/images/ppv6/thumb_validation/
echo " ******************** Preprocessing"
PYTHONPATH=. poetry run python -m xd.xview3.preproc

# Build instance segmentation models (configs/localization/v13)
# - PPV2VO
echo " ******************** Training (instance segmentation model)"
PYTHONPATH=. poetry run python -m xd.xview3.localization.trainer \
    -c configs/localization/v13_iim_ppv2_augv2.yml \
    -f 0

# Build classification models (configs/vessel_class/v77)
# - PPV6, PPV2ShipCropInferenceDataset
echo " ******************** Training (classification model)"

# Build regression models (configs/length_estimation/v16)
# - PPV2ShipCropTrainValDataset, PPV2ShipCropInferenceDataset
echo " ******************** Training (regression model)"

