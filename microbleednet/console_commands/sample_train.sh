#!/bin/bash

# Sample train command, update the arguments before use
export MICROBLEEDNET_PRETRAINED_MODEL_PATH='/path-to-pretrained-model-checkpoints'
microbleednet train -i /directory-containing-files-preprocessed-using-microbleednet-preprocess-function -l /this-argument-is-not-actually-used-but-make-sure-to-use-a-dummy-directory -m /output-directory-for-trained-checkpoints -v=True