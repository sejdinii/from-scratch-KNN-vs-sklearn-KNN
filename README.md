# KNN Classifier: From Scratch vs scikit-learn

A K-Nearest Neighbours classifier written from scratch in Python and compared with scikit-learn's `KNeighborsClassifier` under identical conditions, with the statistical analysis done in R.
MSc Data Science coursework, Lancaster University Leipzig (January 2026).

## What was built

- **`customKNN_Model`**: fit / predict API with Euclidean and Manhattan distance and uniform or distance-weighted voting.
- **Input checks**: rejects a non-positive k, mismatched training data and labels, prediction before training, and a k larger than the training set.
- **Unit tests (pytest, 6 passing)**: distance functions against known values, training storage, prediction length, uniform voting and distance-weighted voting.

## Experiment

Both implementations ran on the same splits, scaling and seeds:

- **Datasets:** Iris, Wine, Digits, Breast Cancer Wisconsin, Nursery
- **k:** 1, 5, 10, 20, 50, 100 (values larger than the training set were skipped)
- **Distance:** Euclidean, Manhattan
- **Voting:** uniform, distance-weighted
- **Training split:** 20%, 50%, 80%

That gives 336 configurations per implementation, 672 in total. Each run recorded accuracy, precision, recall, F1, fit time, prediction time and memory.

## Results

| Implementation | Accuracy | F1 | Predict time | Memory |
|---|---|---|---|---|
| From scratch | 0.921 | 0.869 | 20.42 s | 212.8 KB |
| scikit-learn | 0.915 | 0.862 | 0.04 s | 212.8 KB |

- **Same predictions:** both implementations draw identical decision boundaries at every k. Wilcoxon signed-rank tests flag the accuracy gap as statistically significant, but it is 0.6 percentage points.
- **Different speed:** scikit-learn predicts about 484 times faster. The from-scratch model compares every test point with every training point, while scikit-learn uses optimised neighbour search.
- **Regression analysis:** the implementation is not a significant predictor of accuracy, precision, recall or F1; k and training size are. For prediction time it is the biggest factor (adjusted R² = 0.89), followed by dataset size.

## Run it

```bash
pip install numpy pandas scikit-learn pytest
cd "Python Files"
python experiments.py              # runs all configurations, writes the results CSVs
python decisionBoundaries_data.py  # data for the decision-boundary plots
pytest test_knn.py -v
```

The report (`39991938_Report.Rmd` / `.pdf`) contains the R analysis: summary statistics, Wilcoxon tests, multiple regression and plots.
