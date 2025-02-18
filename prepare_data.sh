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

if [ ! -d "$script_path/data/lafan1" ]; then
    echo "Downloading LaFAN"
    sudo apt install git-lfs
    git lfs install
    git clone https://github.com/ubisoft/ubisoft-laforge-animation-dataset.git ./data/ubisoft-laforge-animation-dataset
    mv ./data/ubisoft-laforge-animation-dataset/lafan1 ./data/lafan1
    unzip ./data/lafan1/lafan1.zip -d ./data/lafan1/data
    rm -rf ./data/ubisoft-laforge-animation-dataset
fi

echo "Done!"