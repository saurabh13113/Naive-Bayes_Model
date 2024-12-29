# 🤖 Bayesian Networks: Predicting Salaries with Naive Bayes

## 📖 Overview
This project implements a **Naive Bayesian model** to predict salaries based on 1994 US Census data attributes such as gender, education, work type, and more. The assignment also explores **fairness** in machine learning predictions.

![image](https://github.com/user-attachments/assets/14ff7db0-d6a9-4e4b-9fea-a1a10e2e02c0)


## 🎯 Objectives

### 🔍 Tasks:
1. **Implement Variable Elimination Algorithm:**
   - Normalize
   - Restrict
   - Sum Out
   - Multiply
   - Compute probabilities with evidence using `ve()`.
2. **Build a Naive Bayes Network:**
   - Define variables and conditional probability tables from census data.
   - Use `adult-train.csv` for training.
3. **Predict Salaries:**
   - Predict salaries in `adult-test.csv` based on evidence.
   - Evaluate predictions for fairness using metrics like **demographic parity** and **separation**.

---

## 📚 Key Concepts

### Naive Bayes Network
- Models relationships between a target variable (`Salary`) and other attributes (`Work`, `Education`, etc.).
- `Salary` acts as the **parent node** in the Bayesian network.
- Conditional probabilities are computed using the training dataset.

#### Example:
- **P(Salary<50K):**  
  `(Count(Salary<50K) / Total Individuals)`
- **P(Work=Private | Salary<50K):**  
  `(Count(Work=Private AND Salary<50K) / Count(Salary<50K))`

---

## 🚀 How to Run

### Prerequisites
- **Python 3**: Ensure Python 3 is installed.

### Commands
To train and test the Naive Bayes model:
```bash
python3 naive_bayes_solution.py
```


markdown
Copy code
# 🤖 Bayesian Networks: Predicting Salaries with Naive Bayes

## 📖 Overview
This project implements a **Naive Bayesian model** to predict salaries based on 1994 US Census data attributes such as gender, education, work type, and more. The assignment also explores **fairness** in machine learning predictions.

---

## 🎯 Objectives

### 🔍 Tasks:
1. **Implement Variable Elimination Algorithm:**
   - Normalize
   - Restrict
   - Sum Out
   - Multiply
   - Compute probabilities with evidence using `ve()`.
2. **Build a Naive Bayes Network:**
   - Define variables and conditional probability tables from census data.
   - Use `adult-train.csv` for training.
3. **Predict Salaries:**
   - Predict salaries in `adult-test.csv` based on evidence.
   - Evaluate predictions for fairness using metrics like **demographic parity** and **separation**.

![image](https://github.com/user-attachments/assets/20cd06a0-a293-4a35-9dca-33566b0a7ca4)


## 📚 Key Concepts

### Naive Bayes Network
- Models relationships between a target variable (`Salary`) and other attributes (`Work`, `Education`, etc.).
- `Salary` acts as the **parent node** in the Bayesian network.
- Conditional probabilities are computed using the training dataset.

#### Example:
- **P(Salary<50K):**  
  `(Count(Salary<50K) / Total Individuals)`
- **P(Work=Private | Salary<50K):**  
  `(Count(Work=Private AND Salary<50K) / Count(Salary<50K))`

---

## 🚀 How to Run

### Prerequisites
- **Python 3**: Ensure Python 3 is installed.

### Commands
To train and test the Naive Bayes model:
```bash
python3 naive_bayes_solution.py
```

### Input Files
adult-train.csv (training dataset)
adult-test.csv (test dataset)

### Output
Predicted probabilities and analysis of fairness metrics.

## 🛠️ Implementation Details
Evaluate predictions for:
Demographic Parity: Equal probabilities across genders.
Separation: Predictions independent of gender given evidence.
Sufficiency: Predictions equal actual outcomes independent of gender.

📂 Project Structure
bash
Copy code
📁 bayesian-networks
├── naive_bayes_solution.py  # Main implementation file
├── bnetbase.py              # Core Bayes Net utilities (do not modify)
├── adult-train.csv          # Training data
├── adult-test.csv           # Test data

## 🔍 Fairness Metrics Analysis
What percentage of women in E1 > E2?
What percentage of men in E1 > E2?
What percentage of women with P(Salary >=$50K | E1) > 0.5 actually earn >=$50K?
What percentage of men with P(Salary >=$50K | E1) > 0.5 actually earn >=$50K?
What percentage of women overall have P(Salary >=$50K | E1) > 0.5?
What percentage of men overall have P(Salary >=$50K | E1) > 0.5?
