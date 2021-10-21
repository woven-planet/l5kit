#!/bin/bash

# Make a temporary download folder
TEMP_DOWNLOAD_DIR=$(mktemp -d)

echo "Dataset location: $L5KIT_DATA_FOLDER"

# Download sample zarr
echo "... Downloading sample zarr dataset"
wget https://lyft-l5-datasets-public.s3-us-west-2.amazonaws.com/prediction/v1.1/sample.tar \
    -q -P $TEMP_DOWNLOAD_DIR
echo "Downloaded sample zarr dataset"

echo "... Extracting sample zarr dataset"
mkdir -p $L5KIT_DATA_FOLDER/scenes
tar xf $TEMP_DOWNLOAD_DIR/sample.tar -C $L5KIT_DATA_FOLDER/scenes
echo "Extracted sample zarr dataset"

# Download semantic map
echo "... Downloading semantic map"
wget https://lyft-l5-datasets-public.s3-us-west-2.amazonaws.com/prediction/v1.1/semantic_map.tar \
    -q -P $TEMP_DOWNLOAD_DIR
echo "Downloaded semantic map"

echo "... Extracting semantic map"
mkdir -p $L5KIT_DATA_FOLDER/semantic_map
tar xf $TEMP_DOWNLOAD_DIR/semantic_map.tar -C $L5KIT_DATA_FOLDER/semantic_map
echo "Extracted semantic map"

echo "... Copying semantic map meta"
cp $L5KIT_DATA_FOLDER/semantic_map/meta.json $L5KIT_DATA_FOLDER/meta.json
echo "Copied semantic map meta"

# Download aerial maps
echo "... Downloading aerial maps (this can take a while)"
wget https://lyft-l5-datasets-public.s3-us-west-2.amazonaws.com/prediction/v1.1/aerial_map.tar \
    -q -P $TEMP_DOWNLOAD_DIR
echo "Downloaded aerial maps"

echo "... Copying aerial maps"
tar xf $TEMP_DOWNLOAD_DIR/aerial_map.tar -C $L5KIT_DATA_FOLDER
echo "Copied aerial maps"

# Dowload sample configuration
wget https://raw.githubusercontent.com/lyft/l5kit/master/examples/visualisation/visualisation_config.yaml -q
wget https://raw.githubusercontent.com/lyft/l5kit/master/examples/RL/gym_config.yaml -q

echo "Dataset is ready !"
