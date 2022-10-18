import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "hecalib",
    version = "1.0.0",
    author = "Ikenna Enebuse",
    author_email = "i.enebuse@gmail.com",
    description = "A simple python library for different methods of handeye calibration for vision guided robots",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/ienebuse/hecalib",
    packages = ["hecalib"],
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    zip_safe = False
)