#!/bin/bash



# Install L5Kit
echo "Installing L5kit..."
pip install --progress-bar off --quiet -U l5kit pyyaml

# Set dataset dir path
DATASET_DIR=${HOME}"/l5kit_data"
echo "${DATASET_DIR}" > "dataset_dir.txt"

#######################################################################################################################

# Make a temporary download folder
TEMP_DOWNLOAD_DIR=$(mktemp -d)

echo "DATASET_DIR = ""${DATASET_DIR}"
# Download sample zarr
echo "Downloading sample zarr dataset..."
wget https://lyft-l5-datasets-public.s3-us-west-2.amazonaws.com/prediction/v1.1/sample.tar \
    -q --show-progress -P "${TEMP_DOWNLOAD_DIR}"

mkdir -p "${DATASET_DIR}"/scenes
tar xf "${TEMP_DOWNLOAD_DIR}"/sample.tar -C "${DATASET_DIR}"/scenes

# Download semantic map
echo "Downloading semantic map..."
wget https://lyft-l5-datasets-public.s3-us-west-2.amazonaws.com/prediction/v1.1/semantic_map.tar \
    -q --show-progress -P "${TEMP_DOWNLOAD_DIR}"
mkdir -p "${DATASET_DIR}"/semantic_map
tar xf "${TEMP_DOWNLOAD_DIR}"/semantic_map.tar -C "${DATASET_DIR}"/semantic_map
cp "${DATASET_DIR}"/semantic_map/meta.json "${DATASET_DIR}"/meta.json

# Download aerial maps
echo "Downloading aerial maps (this can take a while)..."
wget https://lyft-l5-datasets-public.s3-us-west-2.amazonaws.com/prediction/v1.1/aerial_map.tar \
    -q --show-progress -P "${TEMP_DOWNLOAD_DIR}"
tar xf "${TEMP_DOWNLOAD_DIR}"/aerial_map.tar -C "${DATASET_DIR}"

# Download sample configuration
wget https://raw.githubusercontent.com/lyft/l5kit/master/examples/visualisation/visualisation_config.yaml -q
wget https://raw.githubusercontent.com/lyft/l5kit/master/examples/RL/gym_config.yaml -q


echo "Datasets downloaded!"
