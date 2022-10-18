# Model_Assertion
Model assertion framework that monitor model drift and decay using existing ML models error metrics

### Project Description
This project aims to build a universal framework that takes in different error metrics from existing rolling ML models as input and monitor/identify model decay & drift with appropriate follow-up action. 

Model Drift/Decay is a depreciation of predictive power due to changes in the external environment. One example could be - when traders use machine learning models to predict price of a finance asset, decay/drift could happen when stock split.

### High-level Breakdown
The project is currently broken down into two parts:

Part A: 
  1. Data Generation
      a. Using daily finance data (via Yahoo Finance), we generate a ML-based trading strategy using XGBRegressor. With this model predicting the t+1 price of the financial asset, we can come up with our some error metric - data we use to validate our main Model Assertion Framework.
      
  2. Model Assertion
      a. Using a wide variety of robust and ML logic, we want to create as many scenarios that can "catch" when the model is drifting using the error metrics provided. 
      b. Robust exmaple - Monitor standard deviation of the MAE score and reports when MAE breach over x standard deviation
      c. ML-based exmaple - Cluster analysis and report when error metrics reports to generate seperated cluster
      


### How this project can be used
At this current stage, we aim to develop this framework using Financial Trading ML strategy as our input. This is because Trading strategy is the most straightforward in terms of validating predictive power and model drift/decay. However, we aim to build this framework to a sophisticated level where all industry and use cases can optimize and help make informed decision for everyone. 


