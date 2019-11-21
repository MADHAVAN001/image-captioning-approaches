#!/bin/bash

echo "Starting to download Google Comceptual Captions metadata"

BASE_DIR="/dataset/google-cc/"

if [ ! -d "${BASE_DIR}" ]
then
    echo "Directory BASE_DIR does not exist."
    mkdir -p "${BASE_DIR}"
fi

cd "${BASE_DIR}" || exit
wget https://storage.cloud.google.com/gcc-data/Train/GCC-training.tsv?_ga=2.191230122.-1896153081.1529438250