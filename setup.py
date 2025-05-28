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
    entry_points={
        'console_scripts': [
            'microbleednet = microbleednet.console_commands.microbleednet:main',
        ],
    },
    scripts=[
        'microbleednet/console_commands/skull_strip_bias_field_correct.sh',
    ],
    packages=find_packages(),
    include_package_data=True,
)