from setuptools import setup

setup(
    name="metaaf",
    version="0.0.1",
    description="metaaf",
    author="Jonah Casebeer, Nicholas J. Bryan, Paris Smaragdis",
    author_email="no-reply@adobe.com",
    url="https://github.com/adobe-research/metaaf",
    packages=["metaaf"],
    license="University of Illinois Open Source License and Adobe Research License",
    install_requires=["tqdm==4.62.*", "wandb==0.12.*", "SoundFile==0.10.*"],
)
