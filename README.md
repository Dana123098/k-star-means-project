# K*-Means Clustering Project

## Overview

This project implements the **K*-Means clustering algorithm** based on the paper:

**“K*-Means: A Parameter-free Clustering Algorithm”**
by Louis Mahon and Mirella Lapata.

The goal of this project is to automatically determine the optimal number of clusters without manually specifying the parameter `k`.

---

## Motivation

The classical **K-Means** algorithm requires the number of clusters `k` to be defined in advance.

In real-world scenarios, this value is often unknown.

Standard approaches such as:

* Elbow method
* Silhouette score

require human interpretation or are not fully reliable.

---

## Idea of K*-Means

K*-Means solves this problem using the **Minimum Description Length (MDL)** principle.

The algorithm selects the number of clusters that minimizes the total description cost:

```text
MDL = L_data + L_model
```

In this project, the following formula is used:

```text
MDL = n*d*log(SSE/(n*d)) + k*d*log(n) + n*log(k)
```

Where:

* `n` – number of data points
* `d` – number of features
* `k` – number of clusters
* `SSE` – sum of squared errors

The idea is to balance:

* data fitting quality (low SSE)
* model complexity (small k)

---

## Project Structure

```text
k-star-means-project/
│
├── src/
│   ├── kmeans.py          # implementation of K-Means
│   ├── kstar_means.py     # K*-Means with MDL
│   ├── mdl.py             # MDL cost function
│   └── plots.py           # visualization
│
├── experiments/
│   └── run_synthetic.py   # experiment script
│
├── results/               # generated plots and CSV
│
├── EXAMPLE.md             # detailed experiment analysis
├── README.md              # project description
└── requirements.txt       # dependencies
```

---

## Installation

Create environment and install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the project

Run the experiment:

```bash
python experiments/run_synthetic.py
```

The script will:

* generate synthetic data
* run K-Means for different k
* compute SSE, silhouette and MDL
* select optimal k using MDL
* save results and plots

---

## Results

The algorithm automatically finds the optimal number of clusters.

Example result:

```text
Best k: 4
```

Generated outputs:

* SSE vs k plot
* Silhouette vs k plot
* MDL vs k plot
* Final clustering visualization
* CSV file with all results

---

## Comparison of methods

| Method     | Requires manual choice | Objective                    |
| ---------- | ---------------------- | ---------------------------- |
| SSE        | Yes                    | Minimize error               |
| Silhouette | Yes                    | Maximize separation          |
| **MDL**    | No                     | Balance error and complexity |

MDL is the main criterion used in this project.

---

## Conclusion

The implemented K*-Means algorithm successfully determines the correct number of clusters without manual tuning.

The MDL principle provides a more robust and theoretically grounded alternative to classical methods.

---

## Authors

* Student 1 – Algorithm implementation
* Student 2 – Analysis and visualization

---

## Notes

This project was created as part of a university assignment in machine learning.
