#!/bin/bash

export KCVUC_RAW_DATA="./data/raw"

# First we must go to Kaggle and create an API token that will download the "kaggle.json" which contains our API credentials.
# Then we must move it to our ~/.kaggle folder
mv ~/kaggle.json ~/.kaggle/

echo "Moved kaggle API credentials"

# So that other users can't read our credentials, we will define the following permissions
chmod 600 ~/.kaggle/kaggle.json

echo "We limited the permissions to out API credentials so that other users can't read them"

# Download the data from the Kaggle system
kaggle competitions download -c understanding_cloud_organization -p $KCVUC_RAW_DATA/

echo "Downloaded the data from the Kaggle system"

# unzip the file we downloaded
unzip $KCVUC_RAW_DATA/understanding_cloud_organization.zip -d $KCVUC_RAW_DATA/

echo "Unziped the file we downloaded"

# We want this data to remain immutable so we run
# #(chmod a+rwx,u-x,g-wx,o-wx) sets permissions so that, 
# # (U)ser / owner can read, can write and can't execute. 
# # (G)roup can read, can't write and can't execute.
# # (O)thers can read, can't write and can't execute.
chmod 644 $KCVUC_RAW_DATA/*

echo "Limited permissions to the raw data"
