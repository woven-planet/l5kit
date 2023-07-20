#!/bin/bash

# Make a temporary download folder
TEMP_DOWNLOAD_DIR=$(mktemp -d)
TEMP_DATASET_DIR=$(mktemp -d)

# Download sample zarr
echo "Downloading sample zarr dataset..."
wget https://d20lyvjneielsk.cloudfront.net/prediction-sample.tar \
    -q --show-progress -P $TEMP_DOWNLOAD_DIR -O $TEMP_DOWNLOAD_DIR/sample.tar

mkdir -p $TEMP_DATASET_DIR/scenes
tar xf $TEMP_DOWNLOAD_DIR/sample.tar -C $TEMP_DATASET_DIR/scenes

# Download semantic map
echo "Downloading semantic map..."
wget https://d20lyvjneielsk.cloudfront.net/prediction-semantic_map.tar \
    -q --show-progress -P $TEMP_DOWNLOAD_DIR -O $TEMP_DOWNLOAD_DIR/semantic_map.tar
mkdir -p $TEMP_DATASET_DIR/semantic_map
tar xf $TEMP_DOWNLOAD_DIR/semantic_map.tar -C $TEMP_DATASET_DIR/semantic_map
cp $TEMP_DATASET_DIR/semantic_map/meta.json $TEMP_DATASET_DIR/meta.json

# Download aerial maps
echo "Downloading aerial maps (this can take a while)..."
wget https://d20lyvjneielsk.cloudfront.net/prediction-aerial_map.tar \
    -q --show-progress -P $TEMP_DOWNLOAD_DIR -O $TEMP_DOWNLOAD_DIR/aerial_map.tar
tar xf $TEMP_DOWNLOAD_DIR/aerial_map.tar -C $TEMP_DATASET_DIR

# Dowload sample configuration
wget https://raw.githubusercontent.com/lyft/l5kit/master/examples/visualisation/visualisation_config.yaml -q
wget https://raw.githubusercontent.com/lyft/l5kit/master/examples/RL/gym_config.yaml -q

# Install L5Kit
echo "Installing L5kit..."
pip install --progress-bar off --quiet -U l5kit pyyaml

echo "Dataset and L5kit are ready !"
echo $TEMP_DATASET_DIR > "dataset_dir.txt"
