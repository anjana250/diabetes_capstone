### Predictive Modeling of Diabetes Risk Factors

**Anjana Cox**

#### Executive summary

#### Rationale
Health and Social Impact
Public Health: Diabetes is a major public health concern, affecting millions of people worldwide. Early prediction and intervention can prevent or delay the onset of diabetes, reducing healthcare costs and improving quality of life.

Preventive Care: By identifying individuals at risk of developing diabetes, healthcare providers can offer targeted preventive measures, such as lifestyle changes or medical interventions, to reduce the likelihood of disease progression.

Personalized Medicine: Machine learning models can help tailor personalized healthcare plans based on an individual's unique risk factors, leading to more effective and efficient treatment strategies.


#### Research Question
"Can we accurately predict the risk of diabetes and prediabetes in individuals using machine learning models based on a set of health indicators and lifestyle factors?"

Specific Objectives:
Prediction Accuracy: How accurately can different machine learning models predict the presence of diabetes or prediabetes using features such as blood pressure, cholesterol levels, BMI, smoking status, physical activity, and dietary habits?

Model Comparison: Which machine learning model (Logistic Regression, SVM, KNN, Random Forest, Decision Tree) performs best in terms of accuracy, recall, precision, and F1-score for predicting diabetes risk?

Feature Importance: Which health indicators and lifestyle factors are most significant in predicting diabetes risk, as identified by the machine learning models?

#### Data Sources
[The dataset can be found here.](https://www.archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators). 

This dataset pertains to patient health indicators and includes both demographic and health-related features. Each entry is uniquely identified by a patient ID (ID). The primary target variable, Diabetes_binary, indicates the presence of diabetes or prediabetes. Various binary features capture health conditions and behaviors, such as high blood pressure (HighBP), high cholesterol (HighChol), smoking history (Smoker), history of stroke (Stroke), and coronary heart disease or myocardial infarction (HeartDiseaseorAttack). Additional features track physical activity (PhysActivity), dietary habits (Fruits and Veggies), heavy alcohol consumption (HvyAlcoholConsump), and access to healthcare (AnyHealthcare, NoDocbcCost). General health status is assessed using a scale (GenHlth), and mental and physical health issues are quantified by the number of days affected in the past month (MentHlth, PhysHlth). The dataset also includes information on difficulties with walking (DiffWalk) and the sex of the patient (Sex). All these features are essential for analyzing and predicting diabetes risk, with no missing values reported.

#### Methodology

1. Data Preprocessing
Data Cleaning: The dataset did not have any missing values but did contain some duplicates. Those were removed. It was observed that there was a huge disparity in the target column for the minority class (individuals with diabetes or prediabetes). To compensate for this, we did Data Resampling. This addressed the impalance by upsampling the minority class to match the number of samples in majority class.


2. Feature Engineering
Feature Selection: All of the features in the dataset were included in the analysis. Preprossing was done for all of the columns except the binary. 

3. Machine Learning Models
A baseline accuracy was obtained and the following models were then implemented.
 1. Logistic Regression: A linear model for binary classification
 2. Support Vector Machine (SVM): A model that finds the optimal hyperplane for separating classes.
 3. K-Nearest Neighbors (KNN): A non-parametric model that classifies based on the majority class of the nearest neighbors.
 4. Random Forest: An ensemble model using multiple decision trees to improve classification performance.
 5. Decision Tree: A tree-based model that splits data based on feature values to make predictions.
 6. Neural Network: A deep learning model designed to capture complex patterns and interactions in the data. It consists of multiple layers (dense and dropout layers) and is optimized using hyperparameter tuning to improve performance.

4. Model Evaluation and Hyperparameter Tuning
Grid Search with Cross-Validation: Tuning hyperparameters for each model to optimize performance. Using 5-fold/ 3-fold cross-validation ensures that the model generalizes well to unseen data. 
Evaluation Metrics: Using accuracy, precision, recall, and F1-score to assess and compare the performance of each model. While all of these metrics are very important, recall was priortitized because it is crucial to minimize false negatives since this can result in a patient not receiving necessary treatments.

5. Feature Importance
Logistic Regression Coefficients: Analyzing the magnitude and direction of coefficients to determine feature importance.
Random Forest Feature Importance: Using feature importance scores from the Random Forest model to identify significant predictors.
Visualization and Reporting
Confusion Matrix: Visualizing the performance of each model in terms of true positives, true negatives, false positives, and false negatives.
Bar Plots: Comparing the performance metrics (accuracy, precision, recall, F1-score) of all models using bar plots.
Feature Importance Plots: Visualizing the importance of different features as determined by the models.


#### Results

![comparison_metrics](https://github.com/user-attachments/assets/6a41c0fc-d8ec-4032-b2d7-eddaba83f0d1)


Random Forest did the best on all the metrics except for Recall. KNN did the best for recall. Since Random Forest seems to have performed the best overall, the recommendations will be made using this model.


### Random Forest Performance Analysis
**1. Performance Metrics:**

Random Forest vs. KNN Recall: KNN achieved the highest recall, indicating that it was more successful in identifying all relevant cases of diabetes (i.e., true positives). This is important for applications where missing a positive case (false negative) can have serious consequences.
Overall Performance: Despite KNN’s higher recall, Random Forest outperformed KNN on other metrics like precision, accuracy, and the ROC AUC score. This suggests that while Random Forest might be slightly less sensitive in identifying every positive case, it generally provides a more balanced and reliable performance across different metrics.

**2. Reasons for Random Forest's Strong Performance:**

Ensemble Learning: Random Forest is an ensemble method that combines the predictions of multiple decision trees. This approach helps to reduce overfitting and improves generalization by averaging the predictions from several models.
Feature Importance: Random Forest can evaluate feature importance, which helps to understand which features are most predictive of the outcome. This is particularly useful for understanding the factors contributing to diabetes risk.
Robustness to Noise: Random Forest is less sensitive to noise in the data compared to single decision trees. The aggregation of multiple trees helps in reducing the effect of outliers and noise.
Handling of Non-Linearity: Random Forest can capture complex, non-linear relationships between features and the target variable, which might be missed by simpler models like Logistic Regression.

**3. Observations from Feature Importance**

Factors Increasing Likelihood of Diabetes:

1. Cholesterol Check: Regular cholesterol checks may indicate an awareness or management of health conditions that could be associated with diabetes. If individuals have checked their cholesterol, it might be due to existing health issues, including diabetes.
2. High Blood Pressure and High Cholesterol: These conditions are well-known risk factors for diabetes. They often co-occur with diabetes and contribute to its development.
3. General Health: Poor general health is a broad indicator that can encompass multiple health issues, including diabetes. Individuals reporting poor health may be more likely to have diabetes.

Factors Decreasing Likelihood of Diabetes:

1. Eating Fruits and Vegetables: A diet rich in fruits and vegetables is associated with lower risks of chronic diseases, including diabetes. These foods are high in fiber and nutrients that can help regulate blood sugar levels.
2. Physical Activity: Regular physical activity is crucial for maintaining a healthy weight and improving insulin sensitivity, both of which reduce the risk of diabetes.

### Conclusion and Recommendations

**4. Conclusion:**

Risk Factors: Individuals with high blood pressure, high cholesterol, or poor general health are at a higher risk for diabetes. Monitoring and managing these conditions is crucial for early detection and prevention.
Preventive Measures: Encouraging a diet rich in fruits and vegetables and promoting regular physical activity can help reduce the risk of diabetes. These lifestyle changes are beneficial for individuals at risk and the general population.
Recommendations:

Early Intervention: For individuals with known risk factors such as high blood pressure or cholesterol, regular screenings and early interventions are recommended to prevent the onset of diabetes.
Lifestyle Modifications: Implement programs or initiatives focused on improving diet and increasing physical activity. This could include public health campaigns, community fitness programs, and nutritional counseling.
Regular Monitoring: Encourage regular health check-ups and screenings for those with risk factors, to track their health status and intervene promptly if needed.

#### Next steps

1. Advanced Modeling Techniques : Ensemble Methods: Explore advanced ensemble techniques such as Gradient Boosting (e.g., XGBoost, LightGBM) and AdaBoost, which often provide superior performance by combining the strengths of multiple models.

2. Model Stacking: Combine predictions from multiple models (e.g., Logistic Regression, SVM, Random Forest) to create a meta-model that can improve overall predictive performance.

3. Model Evaluation and Validation: ROC and AUC: Use ROC (Receiver Operating Characteristic) curves and AUC (Area Under the Curve) scores to evaluate and compare model performance more comprehensively.

4. Ethical Considerations: Bias and Fairness: Evaluate your models for potential biases and ensure fairness across different demographic groups. Consider techniques to mitigate any identified biases.

#### Outline of project

[Full Jupyter Notebook located here.](https://github.com/anjana250/capstone/blob/main/Diabetes_Capstone.ipynb)

##### Contact and Further Information

[You can reach me on LinkedIn!](https://www.linkedin.com/in/anjana-cox-593b407a/)
