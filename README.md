# Personalization Engine for Content Optimization



## Overview

This project builds an end-to-end personalization framework to improve content engagement using user behavior and content features.



## Problem Statement

Content platforms need to deliver relevant content to users. This project explores whether incorporating user segmentation improves engagement prediction.



## Approach



### 1. Data Exploration

- Analyzed user behavior and content metadata from Microsoft MIND dataset



### 2. User Segmentation

- Applied KMeans clustering using:

&#x20; - Click-through rate (CTR)

&#x20; - Engagement frequency

- Identified distinct behavioral segments



### 3. Modeling

- Built two models:

&#x20; - Baseline (content only)

&#x20; - Enhanced (content + user segment)



### 4. Results



| Model | AUC |

|------|-----|

| Baseline | 0.57 |

| Enhanced | 0.69 |



👉 \~21% improvement using segmentation



## Key Insights

- User behavior significantly improves engagement prediction

- Different segments prefer different content categories

- Personalization should be segment-driven



## Business Impact

- Improved recommendation relevance

- Better user engagement strategies

- Scalable personalization framework



## Tech Stack

- Python (Pandas, Scikit-learn)

- Jupyter Notebook

- Streamlit (demo app)


## How to Run

1. Clone the repository
git clone https://github.com/romy0806/personalization-engine-content-optimization.git
cd personalization-engine-content-optimization

2. Install dependencies
pip install -r requirements.txt

3. Add dataset (IMPORTANT)
Download the MIND dataset (small version) and place it in:
data/MINDsmall_train/
data/MINDsmall_dev/

Note: Dataset is not included in this repo due to size limitations.

4. Run the notebook
Start Jupyter Notebook:
jupyter notebook

Then open:
notebooks/01_exploration.ipynb

5. (Optional) Run the Streamlit app
streamlit run app/streamlit_app.py

## Repository Structure
personalization-engine/
├── data/ # Dataset (not included)
├── notebooks/
│ └── 01_exploration.ipynb
├── app/
│ └── streamlit_app.py
├── requirements.txt
└── README.md

