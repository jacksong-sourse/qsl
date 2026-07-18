"""
QSL - Quantum Search Language
=============================

基于"前提-工具-问题-主函数"框架的量子搜索领域专用语言。
支持经典模拟器和 IBM 量子计算机两种运行模式。

安装:
    pip install -e .              # 基础安装 (经典模拟器)
    pip install -e ".[ibm]"      # 含 IBM Quantum 支持
    pip install -e ".[dev]"      # 含测试工具
"""

from setuptools import setup, find_packages
import os

# 读取 README (如果存在)
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
long_description = ""
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="qsl-quantum",
    version="0.4.0",
    author="Song Ziming",
    author_email="15011462616@163.com",
    description="Quantum Search Language - DSL for Grover search with simulator and IBM Quantum backends",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitee.com/song_jack/qsl",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifier=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20",
    ],
    extras_require={
        "ibm": [
            "qiskit>=1.0.0",
            "qiskit-aer>=0.14.0",
            "qiskit-ibm-runtime>=0.20.0",
        ],
        "aws": [
            "amazon-braket-sdk>=1.0.0",
            "amazon-braket-default-simulator>=1.0.0",
        ],
        "algorithms": [
            "scipy>=1.8.0",
        ],
        "qml": [
            "torch>=2.0.0",
            "scikit-learn>=1.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
        "full": [
            "qiskit>=1.0.0",
            "qiskit-aer>=0.14.0",
            "qiskit-ibm-runtime>=0.20.0",
            "amazon-braket-sdk>=1.0.0",
            "amazon-braket-default-simulator>=1.0.0",
            "scipy>=1.8.0",
            "torch>=2.0.0",
            "scikit-learn>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "qsl = qsl.__main__:main",
        ],
    },
    include_package_data=True,
)
