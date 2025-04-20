
# 🍷 Wine Dataset Clustering Analysis

This project applies unsupervised clustering algorithms to the classic **Wine Recognition Dataset** using various preprocessing techniques. The goal is to determine the best combination of algorithm and data transformation to maximize clustering performance.

---

## 📊 Dataset Description

| Property             | Value                                       |
|----------------------|---------------------------------------------|
| **Number of Samples** | 178                                         |
| **Number of Features**| 13 numerical attributes + target class      |
| **Classes**           | 3 (class_0, class_1, class_2)               |

### 📌 Features
- Alcohol  
- Malic acid  
- Ash  
- Alcalinity of ash  
- Magnesium  
- Total phenols  
- Flavanoids  
- Nonflavanoid phenols  
- Proanthocyanins  
- Color intensity  
- Hue  
- OD280/OD315 of diluted wines  
- Proline

---

## ⚙️ Clustering Methods and Preprocessing Techniques

We evaluated three clustering algorithms:
- **K-Means Clustering**
- **Hierarchical Clustering**
- **MeanShift Clustering**

Across multiple preprocessing strategies:
- No Processing
- Normalization
- Log Transformation
- PCA (Principal Component Analysis)
- Transformation + Normalization (T+N)
- Transformation + Normalization + PCA (T+N+PCA)

For KMeans and Hierarchical clustering, we tested cluster counts: **3, 4, and 5**.

---

## 📈 Evaluation Metrics

| Metric                | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| **Silhouette Score**   | Measures how similar a point is to its own cluster vs. other clusters       |
| **Calinski-Harabasz**  | Ratio of between-cluster dispersion to within-cluster dispersion            |
| **Davies-Bouldin**     | Measures intra-cluster similarity and inter-cluster differences (lower is better) |

---

## 🔍 Results Summary

### 🧪 K-Means Clustering

| Preprocessing       | Clusters | Silhouette | Calinski-Harabasz | Davies-Bouldin |
|---------------------|----------|------------|--------------------|----------------|
| No Processing        | 3        | 0.571      | 561.82             | 0.534          |
| Using PCA            | 3        | 0.581      | 569.40             | 0.521          |
| T+N+PCA              | 3        | 0.338      | 98.35              | 1.198          |

---

### 🧬 Hierarchical Clustering

| Preprocessing       | Clusters | Silhouette | Calinski-Harabasz | Davies-Bouldin |
|---------------------|----------|------------|--------------------|----------------|
| No Processing        | 3        | 0.564      | 552.85             | 0.536          |
| Using PCA            | 3        | **0.603**  | 462.66             | 0.522          |
| T+N+PCA              | 3        | 0.329      | 94.45              | 1.205          |

---

### 🔄 MeanShift Clustering

| Preprocessing       | Silhouette | Calinski-Harabasz | Davies-Bouldin |
|---------------------|------------|--------------------|----------------|
| No Processing        | 0.582      | 350.84             | 0.450          |
| Using PCA            | **0.585**  | 269.44             | **0.388**      |
| T+N+PCA              | 0.203      | 45.11              | 2.233          |

---

## 🏆 Best Configuration

| Algorithm              | Preprocessing | Clusters | Silhouette Score |
|------------------------|---------------|----------|------------------|
| **Hierarchical Clustering** | **Using PCA**    | **3**      | **0.6031**        |

---

## 🧠 Key Takeaways

- **PCA** significantly enhances clustering performance across all algorithms.
- **Hierarchical Clustering + PCA** delivers the best results with a **Silhouette Score of 0.6031**.
- **Normalization alone** and **Transformation + Normalization** tend to degrade performance.
- **MeanShift Clustering** also performs well but is more computationally expensive and does not allow easy tuning of the number of clusters.
- The optimal number of clusters across methods was consistently **3**, aligning with the actual number of classes in the dataset.

---

## 📌 Conclusions

✅ The best-performing combination for the Wine dataset is:

> **Hierarchical Clustering + PCA + 3 Clusters**

This combination provides clear separation among clusters, effective dimensionality reduction, and excellent silhouette and Davies-Bouldin scores.

---

## 📂 Project Structure

```
.
├── wine_clustering.ipynb        # Main Jupyter notebook
├── README.md                    # Project summary and documentation
├── data/                        # Wine dataset
└── output/                      # Visualizations and export files
```

---

## 📎 References

- [UCI Wine Dataset](https://archive.ics.uci.edu/ml/datasets/Wine)
- scikit-learn documentation
