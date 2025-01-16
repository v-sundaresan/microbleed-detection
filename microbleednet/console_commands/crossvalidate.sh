#!/bin/bash

# Sample crossvalidate command, update the arguments before use
export MICROBLEEDNET_PRETRAINED_MODEL_PATH='/path-to-pretrained-model-checkpoints'
microbleednet cross_validate -i /home/deepika/gouriworkshere/CerebralMicrobleed/Data/CMB_Preprocessed_FRST_sum_2_3 -l /home/deepika/gouriworkshere/CerebralMicrobleed/Data/CMB_Preprocessed_FRST_sum_2_3 -o /home/deepika/gouriworkshere/CerebralMicrobleed/forked_microbleednet_outputs -int=True -v=True

