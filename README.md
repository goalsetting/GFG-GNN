# Deciphering Collaborative Fraud in Digital Transaction Platforms

This repository contains the implementation for the manuscript:

**"Deciphering Collaborative Fraud in Digital Transaction Platforms: Jointly Modeling User Relationships and Community Signals"**,  
submitted to *Decision Support Systems (DSS)*.

---

## 🔧 Dependency: QueryOPT

The implementation of the $(\alpha, \beta)$-core algorithm relies on the following project:

- https://github.com/boge-liu/alpha-beta-core

---

## 📊 Dataset

The dataset used in this project is publicly available from the competition:

- https://tianchi.aliyun.com/dataset/dataDetail?dataId=123862

⚠️ Please download the **final round dataset**.

> Note: Due to licensing and size constraints, datasets are **not included** in this repository.  
> Users are required to download and preprocess the data independently.

---

## ⚙️ Installation & Setup

### 1. Preprocess the dataset
```bash
python dataset/get_data.py
```

### 2. Install SWIG
```bash
sudo apt-get install swig
```

Official website: https://www.swig.org/

### 3. Build pyabcore (QueryOPT backend)
```bash
sudo apt-get -y install libboost-all-dev
cd ./queryopt
./build.sh
cd ..
```

---

## 🚀 Running the Project

```bash
python main.py
```

---

## 📌 Notes

- Ensure all dependencies (e.g., Boost, SWIG) are correctly installed before building.
- The preprocessing step is required before running the main script.
- The project assumes a Linux-based environment (tested on Ubuntu).

---
