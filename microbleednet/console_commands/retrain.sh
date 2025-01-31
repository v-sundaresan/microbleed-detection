#!/bin/bash

# Sample preprocess command, update the arguments before use
export MICROBLEEDNET_PRETRAINED_MODEL_PATH='/home/deepika/gouriworkshere/CerebralMicrobleed/fullset_retrained_checkpoints'

microbleednet preprocess -i /home/deepika/gouriworkshere/CerebralMicrobleed/Data/CMB_Monash -l /home/deepika/gouriworkshere/CerebralMicrobleed/Data/CMB_Monash -o /home/deepika/gouriworkshere/CerebralMicrobleed/Data/FullSet -r _space-T2S_T2S -v=True

# microbleednet preprocess -i /home/deepika/gouriworkshere/CerebralMicrobleed/Data/CMB_Data/input -l /home/deepika/gouriworkshere/CerebralMicrobleed/Data/CMB_Data/label -o /home/deepika/gouriworkshere/CerebralMicrobleed/Data/FullSet -r _brain -f=True -v=True

# microbleednet preprocess -i /home/deepika/gouriworkshere/CerebralMicrobleed/Data/Rob_Data/raw_input -l /home/deepika/gouriworkshere/CerebralMicrobleed/Data/Rob_Data/label -o /home/deepika/gouriworkshere/CerebralMicrobleed/Data/FullSet -r _brain_restore -f=True -v=True