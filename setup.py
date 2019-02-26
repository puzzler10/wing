import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="wing",
    version="0.0.4",
    author="Tom Roth",
    author_email="tom.roth@live.com.au",
    description="Utility functions for data science",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/puzzler10/wing",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)