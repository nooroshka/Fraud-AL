<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.5em">Active Learning for Credit Card Fraud Detection in Imbalanced Transaction Data</h1>

<p align='center' style="text-align:center;font-size:1em;">
    <strong>Noor Shahin</strong>&nbsp;,&nbsp;
    <strong>Kobi Amit</strong>&nbsp;,&nbsp;
    <strong>Noam Alter</strong>&nbsp;,&nbsp;
    <strong>Elad Polak</strong>
    <br/> 
    Technion - Israel Institute of Technology<br/> 
    Data Analysis and Presentation Lab (096260)
<br>
<br>
    <a href="httpsm://github.com/nooroshka/Fraud-AL">GitHub Repository</a> |
    <a href="https://github.com/nooroshka/Fraud-AL/blob/main/Cognitive_Science_Society_Conference_Submission_Template__COGSCI___3___Copy_(7).pdf">Final Report</a> |
    <a href="#video-link-tbd">Video Presentation (TBD)</a>

</p>

<br>
<br>

<p align="center">
  <img src="httpsstatic://zG9jLTAyLWNsdWMtMDAwMDE3MDRjYzhkZmRiOTZiY2U1MjA3YjI2Yzg3ZTgtMTQzNzY0NDcwMzc1LTUxNTMwNDM4NjA5MDkyNDYwOTQ0/learning_frauds_found.jpg?draw=true" alt="Learning Curve - Frauds Found" width="800">
</p>

# Overview

Credit card fraud detection is a critical task, but it's plagued by extreme class imbalance. In our dataset of over 284,000 transactions, only 492 (0.17%) are fraudulent. This creates a massive problem for machine learning: models need labeled data, but labeling is expensive, and randomly selecting data for labeling is incredibly inefficient—you almost exclusively label legitimate transactions.

This project tackles the problem of **label efficiency**. Instead of random sampling, we use **Active Learning (AL)** to intelligently select which transactions an expert should label next.

We introduce two novel hybrid AL strategies:
1.  [cite_start]**FRaUD++ Hybrid:** Combines a fraud-aware score (uncertainty, focal prior, boundary proximity) with Query-by-Committee (QBC) disagreement [cite: 5251-5278].
2.  **Graph-FRaUD (GraphHybrid):** Our most advanced method. [cite_start]It enhances the `FRaUD++ Hybrid` score by building a k-NN graph of high-risk candidates to identify fraud "hubs" (dense clusters) and "bridges" (boundary-spanning links) [cite: 5279-5374].

This graph-based approach allows the model to reason about the *structure* of fraud, balancing the hunt for known fraud patterns with the search for new ones. The results are dramatic: with the same 5,000-label budget, our `GraphHybrid` strategy finds **+296%** more fraudulent transactions than random sampling and **+61%** more than a standard QBC approach.

# Abstract

Credit card fraud detection poses a highly imbalanced classification challenge in which fraudulent transactions represent less than 0.2% of all activity. Because expert annotation is expensive, improving label efficiency is essential. We evaluate nine active learning (AL) strategies (including random, uncertainty-based, committee-based, cost-balanced, fraud-aware, and two hybrid methods) on the public Kaggle credit card fraud dataset (284,807 transactions, 492 frauds). [cite_start]Our main methodological contributions are two novel domain-tailored hybrid strategies: *FRaUD++ Hybrid*, which mixes fraud-aware scoring with Query-by-Committee (QBC) disagreement [cite: 5251-5278][cite_start], and *GraphHybrid*, which augments FRaUD++ with graph-based hub and bridge signals before combining with QBC [cite: 5279-5374]. Across three random seeds and multiple labeling budgets, both hybrid methods achieve substantially higher fraud discovery than simple AL baselines. At 5000 labels, *GraphHybrid* recovers roughly **61% more frauds than QBC** (218 frauds), a standard committee-based baseline, and up to **296% more than random sampling** (89 frauds), while attaining strong final metrics (median AUPRC 0.826; median AUROC 0.961). These findings demonstrate that integrating fraud-aware priors, rarity cues, structural context, and committee disagreement yields markedly more label-efficient fraud detection under extreme imbalance.

# How to Run

### 1. Setup
Install the required packages.
```bash
pip install -r requirements.txt
