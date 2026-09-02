from setuptools import setup, find_packages

setup(
    name="latense",
    version="0.1.0",
    description="LaTense: Dynamic Latent Space Steering via Geometric Alignment Modulation",
    author="Anonymous",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "latense": ["assets/vectors/*.pt"],
    },
    install_requires=[
        "torch",
        "transformers>=4.46.0",
        "tqdm",
        "datasets",
        "numpy",
        "accelerate",
    ],
    python_requires=">=3.8",
)
