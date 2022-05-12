# TransactionPrediction
 
 Purpose:
 Machine Learning Model XGBoost Classifier that predicts whether debtors will default based on 11 key customer indicators. 
 Built for PayHippo, a SME loan startup based in Nigeria. Nigeria has a rapidly growing economy and the highest population in Africa. 
 The financial markets in Nigeria are just developing so credit scores are not available and ensuring users are trustworth is important in 
 their business model. 
 
 Code:
  1. Read in CSV of two seperate transaction and customer data pools.
  2. Merge data by matching the customer IDs from both data sources
  3. Data Cleaning: 
      a. Convert non-integer data points to integers using category encoding 
      b. Remove data not useful for machine learning model (Ex. IDs)
      c. Use Correlation Heatmap to detect which features are redundant
      d. Remove NaN and Infinity Datapoints
      e. Remove outliers
  4. Split data between Training and Testing data (Test data size is 30% of data) 
  5. Create synthetic data using Adaptive Synthetic (ADASYN) oversampling algorithm due to limited data points
  6. Utilize Bayesian Optimaztion algorithm to hyper-tune parameters of XGBClassifier
  7. Feed sanitized training data into XGBoost Classifier which returns mean AUC score and model
  8. Run predictions on testing data using generated model
  9. Compare predicted values with testing data using Confusion Matrix to analyze performance of classifaction algorithm
  10. Visuals:
       a. Plot Correlation HeatMap
       b. Plot Confusion Matrix
       c. Plot Feature Importance
