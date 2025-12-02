"""
setup script for continual learning via sparse memory finetuning
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

setup(
    name="continual-learning-via-sparse-memory-finetuning",
    version="0.1.0",
    author="dtunai",
    description="Continual Learning via Sparse Memory Finetuning Lin et al., 2025",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dtunai/continual_learning_via_sparse_memory_finetuning",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
)
