# 🇮🇳 Project Samarth — Intelligent Q&A System over data.gov.in (Prototype)

**Built for:** *Build for Bharat / Bharat Digital Fellowship — 2026 Cohort*  
**Author:** Vijayan Naidu  
**Screenshots:** 
![Flowchart](https://github.com/vj220803/Bharat_Digital_Gov/blob/main/F1.PNG)
![Flowchart](https://github.com/vj220803/Bharat_Digital_Gov/blob/main/F2.PNG)
![Flowchart](https://github.com/vj220803/Bharat_Digital_Gov/blob/main/F3.PNG)

---

## 🎯 Overview

**Project Samarth** is an intelligent Q&A system that combines multiple government datasets  
(**IMD Rainfall + Agriculture Crop Production**) into a unified analytical interface.

Users can simply ask questions in natural language like:

- *“Top 5 crops in Himachal Pradesh”*  
- *“Compare rainfall in Kerala and Karnataka for last 10 years”*  
- *“Show rainfall trend in Tamil Nadu for last 15 years”*

The system gives **accurate**, **traceable**, and **dataset-backed** answers using:

✅ DuckDB + Parquet-based OLAP processing  
✅ Deterministic NL → SQL templates  
✅ Fully offline & secure computation  
✅ Government-source dataset citations for every response  

This aligns directly with the **Bharat Digital Fellowship's goals**:  
building privacy-first, high-accuracy, locally-deployable government-tech tools.

---

## 🌐 About Bharat Digital Fellowship (Context)

The Bharat Digital Fellowship encourages building **production-ready, citizen-centric digital systems** for India’s public infrastructure.

This project demonstrates:

- **Data Sovereignty:** No external APIs  
- **Reliability:** Every answer backed by government datasets  
- **Accountability:** Automatic citations  
- **Scalability:** Can integrate soil, temperature, and more datasets  

---

## 🚨 Problem Statement

The primary challenge:

> **Government datasets are not designed to work together.**  
> Rainfall, crop production, soil, temperature — each uses different **formats**, **years**, **codes**, and **schemas**.

The project must:

✅ Fetch & standardize datasets  
✅ Clean inconsistent/irregular CSVs  
✅ Merge them into one logical system  
✅ Allow natural–language queries  
✅ Ensure provenance & accuracy  

**Project Samarth** solves this through a unified ETL + DuckDB analytics engine.

---

## 📚 Current Datasets Used

### ✅ **1. IMD Rainfall Dataset (1901–2017)**
- Columns: state, year, monthly rainfall, annual  
- Granularity: **State/Subdivision**  
- Format: CSV → cleaned → Parquet  
- Use: trend analysis / comparisons  

### ✅ **2. Himachal Pradesh Crop Production Dataset (2019–20)**
- Columns: state, district, crop, production metric tonnes  
- Transformed to **long format**  
- Year standardized to **2022** for demonstration  
- Use: ranking crops, district insights  

---

## ⚠️ Dataset Limitations (Important)

### ❌ Rainfall ends at **2017**  
### ❌ Crop dataset is for **2022**  

➡️ Therefore **same-year joins produce NA values**  
➡️ Prototype handles crop and rainfall queries **independently**  

This is normal and expected.  
Will be fixed when **multi-year crop series** is added in the next version.

---

## 🏗️ System Architecture (ETL + Query Pipeline)
      ┌────────────────────────────┐
      │        Download CSVs        │
      └─────────────┬──────────────┘
                    ▼
      ┌────────────────────────────┐
      │   Validate (peek first lines)│
      └─────────────┬──────────────┘
                    ▼
      ┌────────────────────────────┐
      │ Robust CSV Parser (sep, enc)│
      └─────────────┬──────────────┘
                    ▼
      ┌────────────────────────────┐
      │ Normalise Columns (snake_case)│
      └─────────────┬──────────────┘
                    ▼
      ┌────────────────────────────┐
      │   Standardize: state/year   │
      └─────────────┬──────────────┘
                    ▼
      ┌────────────────────────────┐
      │ Convert Crop Data → Long    │
      └─────────────┬──────────────┘
                    ▼
      ┌────────────────────────────┐
      │ Store Clean Parquet Files   │
      └─────────────┬──────────────┘
                    ▼
      ┌────────────────────────────┐
      │ Load via DuckDB parquet_scan │
      └─────────────┬──────────────┘
                    ▼
      ┌────────────────────────────┐
      │ NL → SQL intent detection   │
      └─────────────┬──────────────┘
                    ▼
      ┌────────────────────────────┐
      │ Run SQL Queries + Cite Data │
      └────────────────────────────┘

---

## 📦 Repository Structure

├── app.py # Streamlit Q&A interface
├── requirements.txt # Python dependencies
├── imd_rainfall.parquet
│---crop_production.parquet
├── assets/
│ └── flowchart.png
├── README.md # This file
└── notebooks/
└── Project_Samarth.ipynb


---

## ⚙️ Installation & Running

### ✅ **1. Install dependencies**
```bash
pip install -r requirements.txt

### ✅ **1. Run the Streanlit App**
```bash
streamlit run app.py


## 🖥️ Supported Queries
### ✅ **Crop-only queries**
1. Top 5 crops in Himachal Pradesh
2. What are the most produced crops in HP?
3. Which district produces the most Wheat in Himachal Pradesh?

### ✅ **Rainfall-only queries**
1. Trend of rainfall in Kerala for last 20 years
2. Compare rainfall in Himachal Pradesh and Punjab for last 5 years
3. Highest rainfall states in India

### ✅ **Future Advancements**
1. Multi-year crop dataset integration
To enable:
Crop vs rainfall correlations
Climate impact forecasting

2. Temperature dataset merge
(Needed for climate risk alerts)

3. Soil Health Card dataset
To evaluate:

4. Soil fertility

5. Crop yield potential

4. API version of the model
With POST /query endpoint.

📚 **Citations**
1. IMD Rainfall Dataset: https://data.gov.in
2. Crop Production Dataset: https://data.gov.in
3. All source links & checksums stored in data_catalog.csv

📞 **Contact**
Vijayan Naidu
venkatesh45naidu@gmail.com / LinkedIn: 
