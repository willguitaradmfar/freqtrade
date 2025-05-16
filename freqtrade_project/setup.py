#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="freqtrade_project",
    version="0.1.0",
    description="Modular FreqTrade project for cryptocurrency trading strategies",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/freqtrade_project",
    packages=find_packages(),
    install_requires=[
        "freqtrade",
        "pandas",
        "numpy",
        "ta-lib",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.8",
) 