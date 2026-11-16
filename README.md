# Iris Classification with a Simple Neural Network (PyTorch)

A minimal feed-forward neural network built with **PyTorch** that classifies the
[Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) into its three species.

## Model

| Layer  | Input | Output | Activation |
|--------|-------|--------|------------|
| `fc1`  | 4     | 8      | ReLU       |
| `fc2`  | 8     | 9      | ReLU       |
| `out`  | 9     | 3      | —          |

- **Inputs (4):** sepal length, sepal width, petal length, petal width
- **Outputs (3):** `0 = setosa`, `1 = versicolor`, `2 = virginica`

## Training setup

- **Data:** Iris CSV loaded from a public GitHub gist with pandas
- **Split:** 80% train / 20% test (`train_test_split`, `random_state=41`)
- **Loss:** `CrossEntropyLoss`
- **Optimizer:** Adam, learning rate `0.01`
- **Epochs:** 100 (full-batch)
- **Seed:** `torch.manual_seed(41)` for reproducibility

## Notebook walkthrough (`NN.ipynb`)

1. Define the `Model` class (`nn.Module`)
2. Load the Iris data and encode species labels as integers
3. Split into train/test sets and convert to tensors
4. Train for 100 epochs, printing the loss every 10 epochs
5. Plot the training loss curve
6. Evaluate on the test set (loss and number of correct predictions)

## Getting started

```bash
pip install -r requirements.txt
jupyter notebook NN.ipynb
```

## Requirements

- Python 3.8+
- torch, pandas, matplotlib, scikit-learn, jupyter

---

## توضیح فارسی

این پروژه یک شبکه عصبی ساده (Feed-forward) با **PyTorch** است که گل‌های دیتاست Iris را
بر اساس چهار ویژگی (طول و عرض کاسبرگ و گلبرگ) به سه گونه طبقه‌بندی می‌کند.
مدل دو لایه پنهان (۸ و ۹ نورون با تابع فعال‌سازی ReLU) دارد، با تابع هزینه
CrossEntropy و بهینه‌ساز Adam به مدت ۱۰۰ epoch آموزش داده می‌شود و در پایان دقت آن
روی داده‌های تست (۲۰٪) سنجیده می‌شود.
