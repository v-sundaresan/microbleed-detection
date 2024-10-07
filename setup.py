from setuptools import setup, find_packages

requirements_path = 'requirements.txt'
with open(requirements_path) as file:
    install_requires = [line.strip() for line in file.readlines()]

setup(
    name='microbleednet',
    version='1.0.1',
    description='DL method for cerebral microbleed segmentation',
    author='Vaanathi Sundaresan',
    install_requires=install_requires,
    scripts=[
        'microbleednet/scripts/argument_parser',
        'microbleednet/scripts/prepare_microbleednet_data',
	],
	packages=find_packages(),
    include_package_data=True,
)