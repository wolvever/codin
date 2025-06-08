#!/usr/bin/env python3
"""Setup script for codin."""

from setuptools import setup, find_packages

setup(
    name="codin",
    version="0.1.0",
    description="A coding agent framework",
    author="Codin Team",
    author_email="info@codin.ai",
    url="https://github.com/codin-team/codin",
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.13",
    install_requires=[
        "click>=8.0.0",
        "python-dotenv>=0.19.0",
        "openai>=1.0.0",
        "pydantic>=2.0.0",
    ],
    entry_points={
        "console_scripts": [
            "codin=codin.cli.commands:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.13",
    ],
) 