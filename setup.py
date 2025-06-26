from setuptools import setup, find_packages

setup(
    name="neuraltrain",
    version="0.6.0",
    author="Alexandre Geraldo",
    author_email="alexgeraldo@gmail.com",
    description="A neural network training framework leveraging Optuna for hyperparameter optimization and parallelism.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/FCUL-LibMPC/neuraltrain",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=open("requirements.txt").read().splitlines(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
)
