#! /bin/bash

script_path=$(dirname "$0")
echo "Creating data folders"
mkdir -p $script_path/data/amass/{body_models,datasets}

if [ ! -d "$script_path/data/all_asfamc" ]; then
    echo "Downloading CMU dataset"
    wget http://mocap.cs.cmu.edu/allasfamc.zip -O allasfamc.zip 
    unzip allasfamc.zip -d data
    rm allasfamc.zip
fi

echo "Done!"