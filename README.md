\# Personalization Engine for Content Optimization



\## Overview

This project builds an end-to-end personalization framework to improve content engagement using user behavior and content features.



\## Problem Statement

Content platforms need to deliver relevant content to users. This project explores whether incorporating user segmentation improves engagement prediction.



\## Approach



\### 1. Data Exploration

\- Analyzed user behavior and content metadata from Microsoft MIND dataset



\### 2. User Segmentation

\- Applied KMeans clustering using:

&#x20; - Click-through rate (CTR)

&#x20; - Engagement frequency

\- Identified distinct behavioral segments



\### 3. Modeling

\- Built two models:

&#x20; - Baseline (content only)

&#x20; - Enhanced (content + user segment)



\### 4. Results



| Model | AUC |

|------|-----|

| Baseline | 0.57 |

| Enhanced | 0.69 |



👉 \~21% improvement using segmentation



\## Key Insights

\- User behavior significantly improves engagement prediction

\- Different segments prefer different content categories

\- Personalization should be segment-driven



\## Business Impact

\- Improved recommendation relevance

\- Better user engagement strategies

\- Scalable personalization framework



\## Tech Stack

\- Python (Pandas, Scikit-learn)

\- Jupyter Notebook

\- Streamlit (demo app)



\## How to Run

