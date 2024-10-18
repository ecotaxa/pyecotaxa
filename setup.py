from setuptools import find_packages, setup

import versioneer

with open("README.md", "r") as fp:
    LONG_DESCRIPTION = fp.read()

setup(
    name="pyecotaxa",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Simon-Martin Schroeder",
    author_email="sms@informatik.uni-kiel.de",
    description="Query EcoTaxa and process its output",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/ecotaxa/pyecotaxa",
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "pandas",
        "numpy",
        "python-dotenv",
        "requests",
        "tqdm",
        "werkzeug",
        "semantic_version",
        "atomicwrites",
    ],
    python_requires=">=3.6",
    extras_require={
        "tests": [
            # Pytest
            "pytest",
            "pytest-cov",
            "ruff",
        ],
        "docs": [
            "sphinx >= 1.4",
            "sphinx_rtd_theme",
            "sphinx-autodoc-typehints>=1.10.0",
        ],
        "dev": ["black", "versioneer"],
    },
    entry_points={},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
    ],
)
