#!/bin/bash

# Sample finetune command, update the arguments before use
export MICROBLEEDNET_PRETRAINED_MODEL_PATH='/path-to-pretrained-model-checkpoints'
microbleednet fine_tune -i /directory-containing-files-preprocessed-using-microbleednet-preprocess-function -l /this-argument-is-not-actually-used-but-make-sure-to-use-a-dummy-directory -o /output-directory-for-finetuned-checkpoints -m pre -v=True