#!/bin/bash

# Sample preprocess command, update the arguments before use
export MICROBLEEDNET_PRETRAINED_MODEL_PATH='/path-to-pretrained-model-checkpoints'
microbleednet preprocess -i /path-to-input-images -l /path-to-input-labels-optional -o /output-path -r common-expression-in-every-image-name-eg-brain_restore -v=True