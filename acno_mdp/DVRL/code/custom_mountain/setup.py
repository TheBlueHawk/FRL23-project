from setuptools import find_packages, setup

setup(
    name="custom_mountain",
    packages=find_packages(),
    include_package_data=False,
    version="0.0.1",
    install_requires=["gym", "numpy", "pandas", "joblib"],
)
