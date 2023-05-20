'''
Cerebral Microbleed detection network 
For training, run:
microbleednet train -i <input_directory> -m <model_directory>
For testing, run:
microbleednet evaluate -i <input_directory> -m <model_directory> -o <output_directory>
For leave-one-out validation, run:
microbleednet loo_validate -i <input_directory> -o <output_directory>
for fine-tuning, run:
microbleednet fine_tune -i <input_directory> -m <model_directory> -o <output_directory>
'''
