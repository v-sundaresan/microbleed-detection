#!/bin/bash

# Sample evaluate command, update the arguments before use
export MICROBLEEDNET_PRETRAINED_MODEL_PATH='/home/deepika/gouriworkshere/CerebralMicrobleed/retrained_microbleednet_checkpoints'
microbleednet fine_tune -i /home/deepika/gouriworkshere/CerebralMicrobleed/Data/CMB_Preprocessed_FRST_sum_2_3 -l /home/deepika/gouriworkshere/CerebralMicrobleed/Data/CMB_Preprocessed_FRST_sum_2_3 -o /home/deepika/gouriworkshere/CerebralMicrobleed/retrained_microbleednet_checkpoints -m pre -v=True