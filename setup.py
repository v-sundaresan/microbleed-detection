from setuptools import setup,find_packages
with open('requirements.txt', 'rt') as f:
    install_requires = [l.strip() for l in f.readlines()]

setup(name='microbleednet',
	version='1.0.1',
	description='DL method for cerebral microbleed segmentation',
	author='Vaanathi Sundaresan',
	install_requires=install_requires,
    scripts=['microbleednet/scripts/microbleednet', 'microbleednet/scripts/prepare_microbleednet_data'],
	packages=find_packages(),
	include_package_data=True)
