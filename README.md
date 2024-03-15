# creditworthiness_ml_model
The Peer Loan Kart dataset is loaded and I will train it on various pytorch neural networks or lightGBM

# Data Analysis

The following plots were generated in Tableau Public. Orange is paid loans and blue is deliquent loans.

## Loan paid back or not (Target Variable)

![Loan Paid (Target)](https://github.com/bpkucsb/creditworthiness_ml_model/assets/13769127/b39a9a72-3768-48a5-a408-962b585279a9)

## Information on Applicant

### Logarithm of Annual Income Amount

![Log Annual Income](https://github.com/bpkucsb/creditworthiness_ml_model/assets/13769127/1ad87f3c-cd0c-4ec3-b06c-9e445c4824cb)

### FICO Score

![Fico Score](https://github.com/bpkucsb/creditworthiness_ml_model/assets/13769127/6363bb3d-ed03-4fc6-b4c3-5a457b09c5a2)

### Delinquencies (30 Days Overdue) in Past 2 Years

![Delinquent Past 2 Years](https://github.com/bpkucsb/creditworthiness_ml_model/assets/13769127/577139cf-d788-4b40-864e-41e9d72e1422)

### Number of Derogatory Records in Last 6 Months

![Number of Derogatory Records](https://github.com/bpkucsb/creditworthiness_ml_model/assets/13769127/2079b887-784d-41b5-a19c-8f56bea5fd03)

## Information on Credit Line

### Does Loan Meet Peer Loan Kart's credit underwriting criteria

![Meets Underwriting Criteria](https://github.com/bpkucsb/creditworthiness_ml_model/assets/13769127/c9c0ad89-cc6b-4a64-bf3e-5302f44dff4a)

### Purpose of the Loan

![Loan Purpose](https://github.com/bpkucsb/creditworthiness_ml_model/assets/13769127/31019a3a-e34e-4f22-86f4-86efcf0feb88)

### Interest Rate of Loan

![Interest Rate](https://github.com/bpkucsb/creditworthiness_ml_model/assets/13769127/a7da02eb-753a-458f-aa6f-f8fd2a469962)

### Monthly Instalments Amount 

![Monthly Installments Amount](https://github.com/bpkucsb/creditworthiness_ml_model/assets/13769127/37e56366-eeb1-480a-88ec-2787fcc5b47e)

### Number of Days Credit Line has been Open

![Days Credit Line Open](https://github.com/bpkucsb/creditworthiness_ml_model/assets/13769127/9d6fdf11-57f6-4495-8dce-efb2cc9044c1)

### Number of Credit Inquiries in the last 6 Months

![Number of Credit Inquires in Last 6 Months](https://github.com/bpkucsb/creditworthiness_ml_model/assets/13769127/7724fbb7-5068-4b0d-becb-ea97e310a130)

## Information on Credit Utilization

### Debt to Income Ratio

![Debt To Income Ratio](https://github.com/bpkucsb/creditworthiness_ml_model/assets/13769127/7707c36c-776d-4b76-8441-b021387091e6)

### Percentage of Credit Utilization

![Percentage Credit Utilization](https://github.com/bpkucsb/creditworthiness_ml_model/assets/13769127/e5c160a4-6d2a-4622-a833-c5fad87890e9)

### Revolving Balance

![Revolving Balance](https://github.com/bpkucsb/creditworthiness_ml_model/assets/13769127/4acae362-2ce7-4ada-a2af-6a3b38f86494)

# Model Exploration and Evaluation

The feature "does the loan meet peer loan kart's underwriting criteria" had a disproportionately high importance in XGBoost. Because it is not entirely clear at which point of the underwriting process this feature becomes available we will not use it as input for now.

## Logistic Regression

Regularization parameter found to be C=0.1

<img width="629" alt="Screen Shot 2024-03-14 at 6 08 57 PM" src="https://github.com/bpkucsb/creditworthiness_ml_model/assets/13769127/79c2bbdd-6d5e-4839-90fe-accef17bb9d1">

<img width="771" alt="Screen Shot 2024-03-14 at 6 09 11 PM" src="https://github.com/bpkucsb/creditworthiness_ml_model/assets/13769127/0b176732-1328-4347-9a8e-4e7d5c8b872d">

## Random Forest

200 decision  trees. Could have done more but for processing speed. Regularization paramater maximum tree depth set to 6.

<img width="625" alt="Screen Shot 2024-03-14 at 6 15 04 PM" src="https://github.com/bpkucsb/creditworthiness_ml_model/assets/13769127/4b40dcb0-4527-4dd4-a5d2-a309ab01df4e">

<img width="759" alt="Screen Shot 2024-03-14 at 6 15 33 PM" src="https://github.com/bpkucsb/creditworthiness_ml_model/assets/13769127/34553118-df7d-44fc-9860-fc02883dfd5e">

## XG Boost

50 rounds of boosting resulting in 50 trees. Learning rate is 0.1. Regularization parameters are gamma=0.1, maximum tree depth 3, alpha=0.1 and lambda=0.005

<img width="620" alt="Screen Shot 2024-03-14 at 6 58 04 PM" src="https://github.com/bpkucsb/creditworthiness_ml_model/assets/13769127/5f616d75-e106-45ae-9152-db97a8faa9e0">

<img width="776" alt="Screen Shot 2024-03-14 at 7 00 39 PM" src="https://github.com/bpkucsb/creditworthiness_ml_model/assets/13769127/7ed26f59-5270-4dce-bfbc-123d875738d5">

## Neural Network

Two layer deep neural network with 8 nodes in each layer. Dropout rate is 10% during training.

<img width="624" alt="Screen Shot 2024-03-14 at 7 01 53 PM" src="https://github.com/bpkucsb/creditworthiness_ml_model/assets/13769127/50a6f426-f7fb-4475-9367-0acb666f915a">

