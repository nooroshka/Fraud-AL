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
    <a href="https://github.com/nooroshka/Fraud-AL/blob/main/Report_AL.pdf">Final Report</a> |
    <a href="#video-link-tbd">Video Presentation (TBD)</a>
</p>

<br>
<br>

<p align="center">
  <img src="https://github.com/nooroshka/Fraud-AL/raw/main/meme.png" alt="Project Overview" width="800">
</p>

# Overview

Credit card fraud detection is a critical task, but it's plagued by extreme class imbalance. In our dataset of over 284,000 transactions, only 492 (0.17%) are fraudulent. This creates a massive problem for machine learning: models need labeled data, but labeling is expensive, and randomly selecting data for labeling is incredibly inefficient, you almost exclusively label legitimate transactions.

This project tackles the problem of **label efficiency**. Instead of random sampling, we use **Active Learning (AL)** to intelligently select which transactions an expert should label next.

We introduce two novel hybrid AL strategies:
1.  **FRaUD++ Hybrid:** Combines a fraud-aware score (uncertainty, focal prior, boundary proximity) with Query-by-Committee (QBC) disagreement.
2.  **GraphHybrid (Graph-FRaUD):** Our most advanced method. It enhances the `FRaUD++ Hybrid` score by building a k-NN graph of high-risk candidates to identify fraud "hubs" (dense clusters) and "bridges" (boundary-spanning links).

This graph-based approach allows the model to reason about the *structure* of fraud, balancing the hunt for known fraud patterns with the search for new ones. The results are dramatic: with the same 5,000-label budget, our `GraphHybrid` strategy finds **+296%** more fraudulent transactions than random sampling and **+61%** more than a standard QBC approach.

# Abstract

Credit card fraud detection poses a highly imbalanced classification challenge in which fraudulent transactions represent less than 0.2% of all activity. Because expert annotation is expensive, improving label efficiency is essential. We evaluate nine active learning (AL) strategies (including random, uncertainty-based, committee-based, cost-balanced, fraud-aware, and two hybrid methods) on the public Kaggle credit card fraud dataset (284,807 transactions, 492 frauds). Our main methodological contributions are two novel domain-tailored hybrid strategies: *FRaUD++ Hybrid*, which mixes fraud-aware scoring with Query-by-Committee (QBC) disagreement, and *GraphHybrid*, which augments FRaUD++ with graph-based hub and bridge signals before combining with QBC. Across three random seeds and multiple labeling budgets, both hybrid methods achieve substantially higher fraud discovery than simple AL baselines. At 5000 labels, *GraphHybrid* recovers roughly **61% more frauds than QBC**, a standard committee-based baseline, and up to **296% more than random sampling**, while attaining strong final metrics (median AUPRC 0.826; median AUROC 0.961 ). These findings demonstrate that integrating fraud-aware priors, rarity cues, structural context, and committee disagreement yields markedly more label-efficient fraud detection under extreme imbalance.

# How to Run

1.  **Install dependencies:**
    ```bash
    pip install numpy pandas scikit-learn lightgbm matplotlib seaborn pyyaml
    ```
2.  **Run an experiment:**
    Edit `config.yaml` to choose your desired `strategy_name` and other parameters.
    ```bash
    python learner.py --config config.yaml
    ```
    Raw results will be saved in the `results/` folder.

3.  **Generate report plots:**
    After running all your experiments, aggregate the results:
    ```bash
    python report_visualizations.py
    ```
    This reads the `results/` folder and saves the final comparison plots and tables to `results_report/`.

# Repository Structure

* **`learner.py`**: Main script to run a full Active Learning experiment.
* **`strategies.py`**: Contains the core logic for all AL sampling strategies, including our novel `GraphHybrid` method.
* **`report_visualizations.py`**: Utility script to aggregate results and generate all plots/tables for the final report.
* **`config.yaml`**: Main configuration file to set all experimental parameters (strategy, budget, etc.).
* **`utils.py`**: Helper functions for metrics, diversity sampling, and plotting.
* **`config.py`**: Loads the `.yaml` config into Python objects.
* **`creditcard.csv`**: The Kaggle dataset used for the project.
* **`results/` (Generated)**: Raw output folder for per-run metrics and plots.
* **`results_report/` (Generated)**: Final output folder for aggregated report-ready figures and tables.
* **`Cognitive_Science_Society...pdf`**: The final project report.

# Acknowledgment

This project was developed as the final submission for the **Data Analysis and Presentation Lab (096260)** at the Technion - Israel Institute of Technology.
