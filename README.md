# 📊 Machine Learning Assignment 1 - Data Analysis & Regression

##  Overview
This Assignment involves **data preprocessing, visualization, and regression modeling** on a dataset of cars. The goal is to **analyze relationships** between features such as **horsepower and MPG** and apply **linear & quadratic regression** techniques to understand these relationships.  

The project includes:
- **Handling missing values**
- **Data visualization (histograms, scatter plots, and boxplots)**
- **Identifying data distributions & skewness**
- **Applying Linear Regression (Closed-form solution & Gradient Descent)**
- **Applying Quadratic Regression**

## 📂 Files
- `Assignment1ML.py` - Contains the full **Python implementation** of the analysis.
- `Assignment1Report.pdf` - Detailed explanation of each part with **visualizations and conclusions**.

---

## 🚀 Steps Performed

### 📌 Part 1: Loading & Inspecting the Dataset
- A CSV file containing **cars data** is loaded using `pandas`.
- The number of **features (columns) and examples (rows)** is displayed.

### 📌 Part 2: Handling Missing Data
- Missing values are **identified** for each column.
- The **'horsepower'** column had 6 missing values.
- The **'origin'** column had 2 missing values.

### 📌 Part 3: Imputing Missing Data
- **'horsepower'** missing values were filled using the **median**.
- **'origin'** missing values were filled using the **mode** (most common value).
- After imputation, there are **zero missing values** in the dataset.

### 📌 Part 4: Boxplot - MPG by Country of Origin
- A **boxplot** is created using `seaborn` to analyze **fuel efficiency (MPG) by country**.
- Conclusion: **Asian cars** tend to be the most fuel-efficient, followed by European cars, while **American cars** have the lowest average MPG.

### 📌 Part 5: Histograms for Feature Distributions
- **Histograms with KDE curves** were created for:
  - 🚗 **Acceleration**
  - 🚙 **Horsepower**
  - ⛽ **MPG**
- Conclusion: **Acceleration** is the feature **closest to a Gaussian (Normal) distribution**.

### 📌 Part 6: Skewness Analysis
- The **skewness** of **acceleration, horsepower, and MPG** is calculated.
- **Acceleration has the lowest skewness**, meaning its distribution is the **most symmetrical**.

### 📌 Part 7: Scatter Plot - MPG vs. Horsepower
- A **scatter plot** was created to visualize the relationship between **MPG and horsepower**.
- Conclusion: As **horsepower increases, MPG decreases** (Negative correlation).

### 📌 Part 8: Linear Regression (Closed-Form Solution)
- A **linear regression line** was fitted to the data using the **closed-form solution**.
- **Formula used**:  
  \[
  W = (X^T X)^{-1} X^T y
  \]
- The parameters **W0** (intercept) and **W1** (slope) were found and plotted.

### 📌 Part 9: Quadratic Regression
- A **quadratic regression model** was applied to improve the fit.
- **Formula used**:  
  \[
  y = W_0 + W_1 X + W_2 X^2
  \]
- The quadratic curve provides a **better fit** to the data than linear regression.

### 📌 Part 10: Gradient Descent for Linear Regression
- **Z-score normalization** was applied to `horsepower` before running **gradient descent**.
- The algorithm iteratively updated the parameters **W0 and W1** until convergence.
- The resulting **regression line closely matched the closed-form solution**.

---

## 📊 Visualizations
This project generates multiple **plots** to analyze relationships between variables:
- **Boxplot**: MPG by Country of Origin
- **Histograms**: Acceleration, Horsepower, MPG
- **Scatter Plot**: MPG vs. Horsepower
- **Linear Regression Line** (Closed-Form Solution)
- **Quadratic Regression Curve**
- **Gradient Descent Regression Line**

---

## 🛠 Technologies Used
- **Python** 🐍
- **pandas** 📊
- **numpy** 🔢
- **seaborn** 🎨
- **matplotlib** 📈

---

## 📎 How to Run the Code?

### 1️⃣ Install Required Libraries
Before running the script, ensure you have the necessary Python libraries installed.  
If they are not installed, use the following command:

```sh
pip install pandas matplotlib seaborn numpy
```
### 2️⃣ Run the Script
Ensure you have the cars.csv dataset in the same directory as the script. Then, execute the following command in your terminal or command prompt:
```sh
python Assignment1ML.py
```

