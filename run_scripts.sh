#!/bin/bash

# Define dataset configurations
datasets=("BrainTumor" "Covid")

# Loop through each dataset configuration
for dataset in "${datasets[@]}"; do
    # Set number of epochs based on dataset
    if [ "$dataset" == "BrainTumor" ]; then
        epochs=20
    elif [ "$dataset" == "Covid" ]; then
        epochs=40
    else
        echo "Unknown dataset: $dataset"
        continue
    fi

    # Build the datasets 
    echo "Building dataset: $dataset"
    python cli.py --name Process --ds "$dataset" --batch_size 64 --epochs "$epochs"
    echo "Dataset: $dataset built"

    # Train the model
    echo "Training for dataset: $dataset"
    python cli.py --name Train --batch_size 64 --epochs "$epochs" --ds "$dataset"
    echo "Training for dataset: $dataset complete"

    # Run inference on the trained model
    echo "Running inference for dataset: $dataset"
    python cli.py --name Saliency --ds "$dataset" --batch_size 64 --epochs "$epochs"
done


# mkdir -p ./Graphics && for file in ./Figures/*.tiff; do convert "$file" ./Graphics/$(basename "${file%.tiff}").png; done

# mkdir -p ./Plots && for file in ./Figures/*.tiff; do convert "$file" ./Plots/$(basename "${file%.tiff}").png; done