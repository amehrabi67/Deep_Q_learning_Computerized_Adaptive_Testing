from setuptools import setup, find_packages

setup(
    name="cat-agent",
    version="0.1.0",
    description="Reinforcement Learning Agents for Computerized Adaptive Testing (CAT)",
    author="Author",
    author_email="author@purdue.edu",
    url="https://github.com/author/cat-agent-project",  # Update with your repo URL
    packages=find_packages(),  # Automatically finds cat_agent/
    install_requires=[
        "numpy>=1.23.5",
        "pandas>=1.5.3",
        "stable-baselines3>=2.0.0",
        "gymnasium>=0.28.1",
        "torch>=2.0.0",
        "scipy>=1.10.1",
        "matplotlib>=3.7.1",
        "tensorboard>=2.12.0"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="MIT",
)
