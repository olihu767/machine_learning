# Machine_learning - Credit Risk Application 

With the use of supervised machine learning, we use several models to predict credit risk. 
Some of the main features that helped us determine which model to use were the following:

*Naive Random Oversampling 
*SMOTE Oversampling
*Undersampling	
*Combination Sampling




## Resources
Data Source: LoanStats_2019Q1.csv
Software: Python 3.7.6, Jupyter Lab 1.2.6
Libraries: Warnings, Numpy, Pandas, Pathlib, Collections, Sklearn, Imblearn


## Analysis

We have used 4 models to test out the predictions of low and high risk credit applications. The following table below shows our results.

<table style = "width:20%>
<tr>
<th><Model></th>	
<th><Accuracy></th> 
<th><Precision - high risk></th>
<th><Recall - high risk></th>
<th><F1 - high risk></th>
<th><Precision - low risk></th>
<th><Recall - low risk></th>
<th><F1 - low risk></th>

<tr>
<td>< Naive Random Oversampling ></td>
<td>< 0.7168></td>
<td>< 0.01></td>
<td>< 0.72></td>
<td>< 0.03></td>
<td>< 1.00></td> 
<td>< 0.71></td> 
<td>< 0.83></td> 
</tr>

<tr>
<td>SMOTE Oversampling</td>
<td>< 0.6700></td>
<td>< 0.01></td>
<td>< 0.57></td>
<td>< 0.03></td>
<td>< 1.00></td>
<td>< 0.77></td>
<td>< 0.87></td>
</tr>

<tr>
<td>< Undersampling ></td>
<td>< 0.6432></td>
<td>< 0.90></td>
<td>< 0.02></td>
<td>< 1.00></td>
<td>< 0.39></td>
<td>< 0.56></td>
</tr>

<tr>
<td>< Combination Sampling></td>
<td>< 0.01></td>
<td>< 0.65></td>
<td>< 0.03></td>
<td>< 0.74></td>
<td>< 0.85></td>
</table>

Looking at this table, we see that Naive Random Oversampling has the highest accuracy score. One thing to note, 
we used the same solver for our Logistic Regression (solver = 'newton-cg') across all four models. This allows us to stay consistent to better measure the results.

All four models had 0.01 for their precision scores for high credit risk and 1.00 for precision low credit risk. 
Undersampling had the highest recall score, at 0.90, for high risk but the lowest for low risk sitting at 0.39. All F1 for high risk scores were low, 
and for low risk they were roughly the same, with the exception of undersampling with an F1 score for low risk at 0.56.

In regards to which model to use, both Oversampling models seem to be quite close, however, 
if we took a look at their confusion matrix, it would seem that Naive Random Oversampling does 
slightly better than SMOTE Oversampling. We recommend using the Naive Random Oversampling because 
it has a higher accuracy score and lower predictions of False Negatives. With credit applications, 
money can be made from approving credit applications where individuals are less likely to default on their loans. 
False Negatives are considered Type II errors and should be avoided if possible. Type I errors are more preferred 
because the applications that are not low risk but are predicted as so, are able to go through another analysis to determine 
if they are approved or not. On the opposite side, if applications are low risk but deemed otherwise, money is lost and potential customers as 
they will look for other financial institutions to borrow credit.