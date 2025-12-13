#!/usr/bin/env python3
"""
ContextBox Setup Script
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

# Read requirements
requirements = []
if (this_directory / "requirements.txt").exists():
    requirements = (this_directory / "requirements.txt").read_text().splitlines()

setup(
    name="contextbox",
    version="0.1.0",
    author="ContextBox Team",
    author_email="team@contextbox.dev",
    description="A tool for capturing and organizing digital context",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/contextbox/contextbox",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Desktop Environment",
        "Topic :: System :: Systems Administration",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "contextbox=contextbox.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "contextbox": ["*.yml", "*.yaml", "*.json", "*.txt", "*.md"],
    },
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.812",
        ],
        "yaml": ["PyYAML>=5.4.0"],
    },
    keywords="context capture digital privacy desktop automation",
    project_urls={
        "Bug Reports": "https://github.com/contextbox/contextbox/issues",
        "Source": "https://github.com/contextbox/contextbox",
        "Documentation": "https://contextbox.readthedocs.io/",
    },
)