# MSPath: A Multi-Scale Vision-Language Model with Clinical Information Prompting for Pathology Report Generation]{A Multi-Scale Vision-Language Model with Clinical Information Prompting for Pathology Report Generation🔬

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch 2.0](https://img.shields.io/badge/pytorch-2.0-red)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 项目简介

MSPath是一个基于**多倍率病理切片(WSI)分析**和**临床信息提示学习**的结直肠癌(COAD)病理诊断模型，实现：
- 全自动肿瘤区域检测与分级
- 结合临床数据(分期、分级)的多模态诊断
- 病理报告结构化生成
- 支持TCGA/HMU等多中心数据集的跨中心验证

**核心创新**：提出了"病理特征+临床语义"双驱动的诊断范式，解决传统模型仅依赖视觉特征的局限性。

## 目录
- [项目简介](#项目简介)
- [环境配置](#环境配置)
- [快速开始](#快速开始)
- [数据准备](#数据准备)
  - [TCGA数据集处理](#tcga数据集处理)
  - [本地WSI文件准备](#本地wsi文件准备)
- [模型训练与评估](#模型训练与评估)
- [项目结构](#项目结构)
- [结果展示](#结果展示)
- [引用方式](#引用方式)
- [许可证](#许可证)
- [联系方式](#联系方式)

## 环境配置

### 1. 依赖安装
```bash
# 克隆仓库
git clone https://github.com/XuDa020612/MSPath.git
cd MSPath

# 创建环境（推荐conda）
conda create -n mspath python=3.9
conda activate mspath

# 安装依赖
pip install -r requirements.txt
