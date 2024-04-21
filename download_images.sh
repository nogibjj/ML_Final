#!/bin/bash

# Read each line from dishes_test.txt
while IFS= read -r dish_name; do
    # Construct the download command
    download_command="gsutil -m cp -r 'gs://nutrition5k_dataset/nutrition5k_dataset/imagery/realsense_overhead/${dish_name}/rgb.png' dishes"

    # Execute the download command
    echo "Downloading ${dish_name}..."
    eval $download_command

    # Rename the downloaded file to ${dish_name}.png
    mv dishes/rgb.png dishes/${dish_name}.png

    echo "Downloaded and renamed ${dish_name}.png"
done < dishes_test.txt
