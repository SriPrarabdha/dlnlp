"""Compatibility metadata for editable installs with pre-PEP 660 pip.

Modern installers use ``pyproject.toml``. Older pip releases fall back to
``setup.py develop`` only when this file is present, so the small amount of
metadata below is intentionally duplicated for that fallback.
"""

from setuptools import find_packages, setup

setup(
    name="focused-longdoc",
    version="0.1.0",
    description="Controlled long-document classification experiments",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        "torch>=2.2",
        "datasets>=3.0",
        "tokenizers>=0.19",
        "numpy>=1.26",
        "scikit-learn>=1.4",
        "PyYAML>=6.0",
        "tqdm>=4.66",
        "wandb>=0.17",
        "python-dotenv>=1.0",
        "psutil>=5.9",
    ],
    extras_require={"dev": ["pytest>=8.0", "ruff>=0.6"]},
    entry_points={
        "console_scripts": [
            "longdoc-data=longdoc.data.download:main",
            "longdoc-run=longdoc.cli:main",
            "run_experiment=longdoc.cli:main",
        ]
    },
)
