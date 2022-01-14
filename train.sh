#!/bin/bash

# Preprocessing
# - data/working/xview3/preprocess_vh_vv_bathymetry_v2
# - data/working/xview3/images/ppv6
poetry run python scripts/preproc.py || (echo "> Error in scripts/preproc.py"; exit 1)

# Build instance segmentation models (configs/localization/v13)
# - PPV2VO

# Build classification models (configs/vessel_class/v77)
# - PPV6, PPV2ShipCropInferenceDataset

# Build regression models (configs/length_estimation/v16)
# - PPV2ShipCropTrainValDataset, PPV2ShipCropInferenceDataset

