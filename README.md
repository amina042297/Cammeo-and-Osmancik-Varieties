# Using Machine Learning to Differentiate Cammeo and Osmancik Varieties

Among  the certified rice grown in TURKEY,  the  Osmancik species, which has a large planting area since 1997 and the Cammeo species grown since 2014 have been selected for the study.  When  looking  at  the  general  characteristics  of  Osmancik species, they have a wide, long, glassy and dull appearance.  When looking at the general characteristics of the Cammeo species, they have wide and long, glassy and dull in appearance.  A total of 3810 rice grain's images were taken for the two species, processed and feature inferences were made. 7 morphological features were obtained for each grain of rice.


#### Table 1. Variables Description
![image](https://github.com/user-attachments/assets/2c7d5ec4-5536-4166-8dd1-347150de338b)




An exploratory data analysis was conducted, including statistical summaries of the features.

#### Table 2. Statistical Summary of variables
![image](https://github.com/user-attachments/assets/e50f8c44-092e-4389-9007-0321929e219e)


Also pairplot comparisons separately for each class was performed to understand the distributions and relationships of the features within each class of rice.

#### Table 3. Pairplots for Cammeo

<img width="293" alt="image" src="https://github.com/user-attachments/assets/230fad76-ca65-4599-8e4d-b47825fc4b0f">

#### Table 4. Pairplots for Osmancik

<img width="293" alt="image" src="https://github.com/user-attachments/assets/95800301-217c-450e-9abf-8f41b1f6e57f">


The dataset was splitted:
Training set size: (3048, 7)
Testing set size: (762, 7)

Model Training and Evaluation
Multiple machine learning models were employed to classify the rice varieties based on their features. The dataset was divided into training (80%) and testing (20%) sets using a random seed for reproducibility. The models trained included:

K-Nearest Neighbors (KNN): Several KNN models with different values of k (number of neighbors) were evaluated to determine the optimal configuration based on accuracy. A feature importance analysis was also conducted by iteratively removing each feature and assessing the impact on model accuracy.

#### Table 5. Accuracy vs k in KNN classifier

<img width="200" alt="image" src="https://github.com/user-attachments/assets/8736118b-1e99-4fbf-9ff9-5e1ee15183d7">

Scenario: Just Area is missing, Accuracy: 0.92
Scenario: Just Perimeter is missing, Accuracy: 0.88
Scenario: Just Major_Axis_Length is missing, Accuracy: 0.88
Scenario: Just Minor_Axis_Length is missing, Accuracy: 0.89
Scenario: Just Eccentricity is missing, Accuracy: 0.89
Scenario: Just Convex_Area is missing, Accuracy: 0.91
Scenario: Just Extent is missing, Accuracy: 0.89

As a result Area was dropped

Logistic Regression: This model was applied to evaluate its performance in terms of accuracy and the ability to linearly separate the classes.

Support Vector Machine (SVM) with a Linear Kernel: Used to determine if a linear decision boundary is sufficient for class separation.

Gaussian Naive Bayes: Applied due to its efficacy with high-dimensional data

Decision Tree and Random Forest: These models were chosen for their ability to handle nonlinear relationships. 

The Random Forest model was further tuned by experimenting with different numbers of trees and depths to minimize the classification error.


#### Table 6. Random Forest: Error rates for different values of N and d

<img width="468" alt="image" src="https://github.com/user-attachments/assets/d7f79e2b-ed46-430c-80ed-730de09e688d">



### Results. 

The following table summarizes the performance metrics of each classification models used to differentiate between two rice varieties, Cammeo and Osmancik. The results include the number of true positives (TP), false positives (FP), true negatives (TN), false negatives (FN), along with the true positive rate (TPR), true negative rate (TNR), and overall accuracy of each model.
•	Random Forest achieved the highest accuracy (93%) with excellent rates for correctly identifying both varieties.
•	Linear SVM also performed very well, with a slightly lower accuracy (92.9%) but the best ability to distinguish non-Cammeo varieties.
•	Logistic Regression was effective, with an accuracy of 92.7%.
•	KNN and Naive Bayes were close in performance, with accuracies just over 91%.
•	Decision Tree lagged behind the other models at 89.5% accuracy.
Random Forest and Linear SVM were the top performers, indicating more sophisticated models were better suited for this classification task


#### Table 7. Results of the models
 
![image](https://github.com/user-attachments/assets/c029c66d-17cf-4c1c-959c-7936b3474cd7)



Dataset: [here](https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik)

- [Summary](https://github.com/amina042297/Cammeo-and-Osmancik-Varieties/blob/main/Summary.docx)
- [Code](https://github.com/amina042297/Cammeo-and-Osmancik-Varieties/blob/main/rice-project.py)
