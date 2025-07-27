#!/usr/bin/env python3
"""
Setup configuration for Project Athena: Multimodal Ethics Framework
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements from requirements.txt
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Package metadata
PACKAGE_NAME = "athena"
VERSION = "1.0.0"
AUTHOR = "Michael Jaramillo"
AUTHOR_EMAIL = "jmichaeloficial@gmail.com"
DESCRIPTION = "Multimodal Ethics Framework for Meta Superintelligence Labs"
URL = "https://github.com/meta-ai/project-athena-multimodal-ethics"
LICENSE = "MIT"

# Classifiers for PyPI
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Scientific/Engineering",
    "Topic :: Multimedia",
    "Topic :: Multimedia :: Graphics",
    "Topic :: Multimedia :: Sound/Audio",
    "Topic :: Multimedia :: Video",
    "Topic :: Security",
]

# Keywords for discovery
KEYWORDS = [
    "ai", "ethics", "multimodal", "meta", "llama", "computer-vision", 
    "nlp", "audio-processing", "video-analysis", "rlhf", "constitutional-ai",
    "responsible-ai", "ai-safety", "content-moderation", "pytorch"
]

# Entry points for command-line tools
ENTRY_POINTS = {
    "console_scripts": [
        "athena=athena.cli.main:main",
        "athena-evaluate=athena.cli.evaluate:main",
        "athena-monitor=athena.cli.monitor:main",
        "athena-train=athena.cli.train:main",
    ],
}

# Package data to include
PACKAGE_DATA = {
    "athena": [
        "configs/*.yaml",
        "models/*.json",
        "data/*.csv",
        "templates/*.html",
        "static/*",
    ],
}

# Extra requirements for different use cases
EXTRAS_REQUIRE = {
    "dev": [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "pytest-asyncio>=0.21.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "mypy>=1.4.0",
        "pre-commit>=3.3.0",
        "sphinx>=7.0.0",
        "sphinx-rtd-theme>=1.3.0",
    ],
    "gpu": [
        "torch[cuda]>=2.0.0",
        "torchvision[cuda]>=0.15.0",
        "torchaudio[cuda]>=2.0.0",
    ],
    "production": [
        "gunicorn>=21.0.0",
        "prometheus-client>=0.17.0",
        "sentry-sdk>=1.28.0",
        "redis>=4.5.0",
    ],
    "research": [
        "jupyter>=1.0.0",
        "notebook>=7.0.0",
        "ipywidgets>=8.0.0",
        "plotly>=5.15.0",
        "dash>=2.11.0",
    ],
    "meta-integration": [
        # Placeholder for Meta-specific packages
        # These would be internal Meta packages
    ],
}

# Long description from README
LONG_DESCRIPTION = read_readme()

# Setup configuration
setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    license=LICENSE,
    packages=find_packages(),
    classifiers=CLASSIFIERS,
    keywords=" ".join(KEYWORDS),
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require=EXTRAS_REQUIRE,
    entry_points=ENTRY_POINTS,
    package_data=PACKAGE_DATA,
    include_package_data=True,
    zip_safe=False,
    project_urls={
        "Bug Reports": f"{URL}/issues",
        "Source": URL,
        "Documentation": f"{URL}/docs",
        "Funding": "https://github.com/sponsors/meta-ai",
        "LinkedIn": "https://www.linkedin.com/in/michael-jaramillo-b61815278",
    },
    # Additional metadata
    platforms=["any"],
    maintainer=AUTHOR,
    maintainer_email=AUTHOR_EMAIL,
    download_url=f"{URL}/archive/v{VERSION}.tar.gz",
)