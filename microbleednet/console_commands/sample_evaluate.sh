#!/bin/bash

# Sample evaluate command, update the arguments before use
export MICROBLEEDNET_PRETRAINED_MODEL_PATH='/path-to-pretrained-model-checkpoints'
microbleednet evaluate -i /directory-containing-files-preprocessed-using-microbleednet-preprocess-function -o /output-directory -p=True -m pre -int=True -v=True

