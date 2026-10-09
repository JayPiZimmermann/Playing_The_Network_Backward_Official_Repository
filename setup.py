from setuptools import setup, find_packages

setup(
    name="bwgame",
    version="0.1.0",
    description="Backward-game framework for neural-network attribution",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    # Tested with Python 3.12.
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0,<3",
        "torchvision>=0.15,<1",
        "numpy>=1.24,<2",
        "scipy~=1.10",
        "pandas~=2.0",
        "quantus~=0.6.0",
        "scikit-image~=0.20",
        "matplotlib~=3.7",
        "pillow>=9.0,<13",
        "opencv-python-headless>=4.8,<5",
    ],
    extras_require={
        "xai": ["captum~=0.7", "grad-cam~=1.5", "zennit~=1.0", "lxt~=2.0"],
        # Everything the test-suite imports.
        "dev": ["pytest>=7.0,<10", "zennit~=1.0", "lxt~=2.0", "captum~=0.7", "grad-cam~=1.5",
                "scikit-learn>=1.3"],
    },
)
