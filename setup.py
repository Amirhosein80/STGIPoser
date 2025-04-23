from setuptools import find_packages, setup

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="STGIPoser",
    version="0.1",
    author="Amirhossein Feiz",
    author_email="amir1380feiz@gmail.com",
    description="Spatio-Temporal Graph Inertia Pose estimation with 6 IMU sensor",
    install_requires=requirements,
    packages=find_packages(),
)
