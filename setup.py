'''
Setup script for the Predictive Maintenance package. 
This script reads the dependencies from requirements.txt and configures the package. 
It uses setuptools for packaging. Generally, this file should not be modified frequently. 
It is primarily used during the initial setup of the project environment.
'''

# Import the 'setup' function (used to define the package metadata)
# and 'find_packages' (used to automatically discover Python packages)
from setuptools import setup, find_packages

# Import List from typing for better type annotations
from typing import List


def get_requirements(file_path: str) -> List[str]:
    """
    Read the dependencies from a requirements file and return them as a list.

    Args:
        file_path (str): The path to the requirements file (e.g., 'requirements.txt').

    Returns:
        List[str]: A list of dependency strings (e.g., ['pandas==1.5.0', 'numpy>=1.21']).
    """
    # Initialize an empty list to store the requirements
    requirements_list: List[str] = []
    try:
        # Open the file in read mode
        with open(file_path, 'r') as file:
            # Read all lines from the file into a list
            lines = file.readlines()

            # Loop through each line in the requirements file
            for line in lines:
                # Remove leading/trailing whitespace and newline characters
                requirement = line.strip()

                # Exclude empty lines and the special '-e .' line 
                # '-e .' is often used in requirements.txt for editable installs, 
                # but it shouldn't be passed to install_requires.
                if requirement and requirement != '-e .':
                    # Add the cleaned requirement to the list
                    requirements_list.append(requirement)

    # Handle the case where the requirements file does not exist
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")

    # Return the final list of requirements
    return requirements_list


# Call the setup() function to define the package metadata and configuration.
# This information is used when installing or distributing the package.
setup(
    name='Predictive Maintenance',               # The name of the package
    version='0.0.1',                             # The current version of the package
    author='Menge Okioma',                       # Author's name
    author_email="Mengeokioma3@gmail.com",       # Author's contact email

    # Automatically find all sub-packages in the project folder.
    # e.g., if you have predictive_maintenance/utils, it will include it automatically.
    packages=find_packages(),

    # Read the dependencies from requirements.txt and include them as install requirements.
    install_requires=get_requirements('requirements.txt')
)

