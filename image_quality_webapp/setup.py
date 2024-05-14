from setuptools import setup, find_packages

setup(
    name='multiscaleiqa',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        # list all packages that your package needs in requirements.txt
        # and read them in here or write them directly as a list
    ],
    entry_points={
        'console_scripts': [
            'myproject=myproject.main:main',
        ],
    },
)