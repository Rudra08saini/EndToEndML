from setuptools import setup, find_packages
from typing import List
hyphen_e_dot = '-e .'
def get_requirements(file_path):
    with open(file_path) as file:
        requirements = file.readlines()
        require = [req.replace('\n', '') for req in requirements if req.strip()]
        if hyphen_e_dot in require:
            require.remove(hyphen_e_dot)
    return require


setup(
    name='MLProject',
    version='0.1.0',
    author='Rudra',
    author_email='rudra08saini@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirement.txt'))