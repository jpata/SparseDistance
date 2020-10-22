import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="jpata",
    version="0.1",
    author="JOosep Pata",
    author_email="joosep.pata@cern.cj",
    description="Compute sparse distance/adjacency matrices using tensorflow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jpata/SparseDistance",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD 3-clause License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
