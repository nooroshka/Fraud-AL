<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.5em">
Active Learning for Credit Card Fraud Detection in Highly Imbalanced Transaction Data
</h1>

<p align='center' style="text-align:center;font-size:1em;">
    <a href="https://github.com/nooroshka">Noor Shahin</a>&nbsp;,&nbsp;
    <a href="https://github.com/KobiAmit">Kobi Amit</a>&nbsp;,&nbsp;
    <a>Noam Alter</a>&nbsp;,&nbsp;
    <a>Elad Polak</a>
    <br/> 
    Technion – Israel Institute of Technology  
    <br/>
    Data Analysis & Presentation Lab (096260)  
    <br/><br/>
    <a href="https://github.com/nooroshka/Fraud-AL">GitHub Repository</a>
</p>

<br>

<p align="center">
  <img src="assets/fraud_icon.png" alt="Fraud Detection Logo" width="260">
</p>

<br>

# Overview

Credit card fraud detection is a **rare-event classification** problem where fraudulent transactions represent **less than 0.2 percent** of real-world data. Labeling fraud requires expert investigation, making annotation **expensive and slow**.  

This project builds a **full active learning (AL) framework** that aims to **maximize fraud discovery while minimizing labeling cost**.

We implement and evaluate **nine active-learning acquisition strategies**, including:

- Classical baselines (Random, Entropy, Margin, QBC)
- Cost-balanced uncertainty sampling
- Fraud-aware strategies (FRaUD & FRaUD++)
- Two novel hybrid methods:
  - **FRaUD++ Hybrid** – mixes fraud-aware scoring with QBC disagreement  
  - **GraphHybrid** – augments FRaUD++ with structural signals from kNN graphs and mixes with QBC  

Across three seeds and budgets up to **5000 labels**, our hybrid methods detect up to:

- **61 percent more frauds** than QBC  
- **296 percent more frauds** than Random  
- **Median AUROC:** 0.961  
- **Median AUPRC:** 0.826  

The results demonstrate that combining **fraud-aware priors, uncertainty, local structure, and diversity** yields dramatically improved label-efficiency under extreme imbalance.

<br>

# Abstract

This project evaluates a suite of classical, imbalance-aware, and structure-aware active learning strategies on the highly imbalanced **Kaggle Credit Card Fraud** dataset (284,807 transactions, 492 frauds).

We propose two new acquisition strategies:

1. **FRaUD++ Hybrid**  
   – Combines fraud-aware scoring (uncertainty, focal prior, rarity, boundary proximity) with committee-based disagreement.

2. **GraphHybrid**  
   – Builds a kNN graph over high-risk candidates and integrates **hub** and **bridge** structural signals before mixing with QBC.

Both hybrid methods substantially outperform classical uncertainty-based baselines and committee-based QBC across budgets and random seeds.

Evaluation includes:  
AUPRC, AUROC, recall@0.1 percent FPR, expected profit, frauds found, and capacity-constrained analysis.

<br>

# Repository Structure

