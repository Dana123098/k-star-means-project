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
│   ├── run_synthetic.py   # main experiment
│   └── compare_methods.py # comparison experiment
│
├── results/               # generated plots and CSV
│
├── EXAMPLE.md             # detailed experiment analysis
├── README.md              # project description
└── requirements.txt       # dependencies
```

---

## Installation

Install all required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the project

Run the main experiment:

```bash
python experiments/run_synthetic.py
```

This script will:

* generate a synthetic dataset
* run K-Means for different values of k
* compute SSE, silhouette score and MDL
* automatically select the best k using MDL
* save results and plots

---

## Output

After running the script, the following files will be created in the `results/` directory:

* `synthetic_results.csv` – table with k, SSE, silhouette and MDL
* `sse_vs_k.png` – SSE plot
* `silhouette_vs_k.png` – silhouette plot
* `mdl_vs_k.png` – MDL plot
* `clusters_kstar.png` – final clustering visualization

---

## Method comparison

To compare different methods of selecting k, run:

```bash
python experiments/compare_methods.py
```

This will generate:

* `method_comparison.csv` – comparison of:

  * manual k
  * silhouette-based selection
  * MDL-based K*-Means

---

## Results

The algorithm automatically finds the optimal number of clusters.

Example result:

```text
Best k: 4
```

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
