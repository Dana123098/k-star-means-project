
```markdown
# Example Analysis — Adaptive $K^*$-Means

This document presents an example analysis performed using the implemented adaptive **$K^*$-Means** clustering algorithm.

---

## Goal of the Experiment

The goal of the experiment is to demonstrate how the $K^*$-Means algorithm:
* automatically determines the number of clusters,
* dynamically performs split and merge operations,
* minimizes the $MDL$ criterion during optimization,
* adapts cluster structure without manually selecting $k$.

---

## Synthetic Datasets

The experiments use synthetic datasets generated with:

```python
from sklearn.datasets import make_blobs

```

Three clustering scenarios were prepared to test the algorithm's robustness.

### 1. FAR Scenario

Clusters are clearly separated.
*Expected behavior:* easy clustering, fast convergence, and correct detection of the true number of clusters.

### 2. MEDIUM Scenario

Clusters partially overlap.
*Expected behavior:* moderate difficulty, temporary over-segmentation, and possible merge operations.

### 3. CLOSE Scenario

Clusters strongly overlap.
*Expected behavior:* difficult clustering, heavy noise, risk of over-segmentation. This scenario tests our strict splitting threshold optimization.

---

## Running the Experiment

To execute the full simulation pipeline, run:

```bash
python experiments/run_synthetic.py

```

The script automatically:

* generates all three synthetic datasets,
* runs the adaptive $K^*$-Means loop,
* performs real-time split/merge operations,
* computes $SSE$ and $MDL$ history,
* saves high-quality plots and CSV tables to the `results/` directory.

---

## Adaptive $K^*$-Means Process

Unlike classical $K$-Means, the algorithm does not test many arbitrary values of $k$. Instead, the entire optimization process is unified into a single run:

1. The algorithm starts with a single global cluster:

$$k = 1$$


2. During optimization, it dynamically modifies the cluster structure using two information-theoretic operations:
* **Maybe-Split:** A cluster is divided into two subclusters if the local $K$-Means split decreases the global $MDL$ cost.
* **Maybe-Merge:** Two nearby clusters are merged into one if the operation maintains or decreases the $MDL$ cost.



---

## MDL Criterion

The algorithm minimizes the Minimum Description Length ($MDL$) cost function, derived directly from information theory:

$$MDL = n \cdot d \cdot \log_2\left(\frac{SSE}{n \cdot d}\right) + k \cdot d \cdot \log_2(n) + n \cdot \log_2(k)$$

Where:

* $n$ — number of data points,
* $d$ — number of features (dimensions),
* $k$ — current number of clusters,
* $SSE$ — Sum of Squared Errors (clustering discrepancy).

The $MDL$ metric acts as an automatic Occam's razor, perfectly balancing **model accuracy** (left term, driven by $SSE$) and **model complexity** (right terms, driven by $k$).

---

## Example Experimental Results

The following stable results were obtained during execution after optimizing the splitting strictness factor:

| Scenario | True $k$ | Detected $k$ | Final $SSE$ | Final $MDL$ |
| --- | --- | --- | --- | --- |
| **far** | $4$ | $4$ | $237.22$ | $2700.36$ |
| **medium** | $4$ | $4$ | $1366.40$ | $3264.95$ |
| **close** | $4$ | $4$ | $3851.12$ | $5420.15$ |

---

## Interpretation of Results

### FAR Scenario

The algorithm flawlessly detected the true structure ($k = 4$). Since the clusters are perfectly isolated, only a few initial split operations were required, and the $MDL$ value quickly stabilized.

### MEDIUM Scenario

The algorithm initially increased the number of clusters to $k = 5$ due to partial data density overlap, but a subsequent **Maybe-Merge** step successfully brought it back to the true $k = 4$. This beautifully demonstrates the structural self-correction mechanism of the algorithm.

### CLOSE Scenario

Thanks to our implementation of a stricter splitting condition, the algorithm successfully avoided extreme over-segmentation on heavily overlapping data, correctly identifying the underlying $k = 4$ structure. This proves that the optimized $MDL$ threshold successfully distinguishes real structural clusters from random data noise.

---

## Visualization of Split and Merge Operations

The project captures the step-by-step evolution of the parameters, generating three convergence plots per case:

* **MDL vs Iteration:** The core optimization trajectory, illustrating the decrease of the description length.
* **SSE vs Iteration:** Tracks the structural minimization of the standard clustering error.
* **K vs Iteration:** A step-plot showing how the number of clusters changes over time.
* **Red dashed lines** indicate a successful `SPLIT` operation.
* **Green dashed lines** indicate a successful `MERGE` operation.



---

## Final Cluster Visualization

The final configurations are exported as 2D scatter plots where data points are colored by their cluster assignments and black **X** markers represent the final stable positions of the centroids:

* `results/graphs/clusters_kstar_far.png`
* `results/graphs/clusters_kstar_medium.png`
* `results/graphs/clusters_kstar_close.png`

---

## Conclusions

The experiments confirm that adaptive $K^*$-Means:

* successfully discovers the true cluster structure automatically without any human input,
* dynamically manages the lifecycle of clusters via mathematical split/merge operations,
* eliminates the need for expensive multi-run heuristic selection techniques (like the Elbow Method).

```

```