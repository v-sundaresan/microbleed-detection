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
        'microbleednet/console_commands/microbleednet',
	],
	packages=find_packages(),
    include_package_data=True,
)
