Meilleurs paramètres trouvés :
```python
{'classifier': SVC(C=1), 'classifier__C': 1, 'classifier__kernel': 'rbf'}
```
User
Rapport de classification :
              precision    recall  f1-score   support

       False       0.88      0.98      0.93      1638
        True       0.84      0.48      0.61       412

    accuracy                           0.88      2050
   macro avg       0.86      0.73      0.77      2050
weighted avg       0.87      0.88      0.86      2050

Meilleurs paramètres trouvés :
```python
{'classifier': SVC(C=100, kernel='poly'), 'classifier__C': 100, 'classifier__kernel': 'poly'}
```
Meilleur score ROC AUC :
0.7118174595818993

Rapport de classification :
              precision    recall  f1-score   support

       False       0.89      0.94      0.91      1638
        True       0.69      0.51      0.59       412

    accuracy                           0.86      2050
   macro avg       0.79      0.73      0.75      2050
weighted avg       0.85      0.86      0.85      2050

# 50% Balanced
Meilleurs paramètres trouvés :
```python
{'classifier': SVC(C=1), 'classifier__C': 1, 'classifier__kernel': 'rbf'}
```
Meilleur score ROC AUC :
0.7923418803418804

Rapport de classification :
              precision    recall  f1-score   support

       False       0.79      0.82      0.80       404
        True       0.81      0.79      0.80       408

    accuracy                           0.80       812
   macro avg       0.80      0.80      0.80       812
weighted avg       0.80      0.80      0.80       812

# Autre essai
Meilleurs paramètres trouvés :
{'classifier': SVC(), 'classifier__C': 100, 'classifier__kernel': 'rbf'}
Meilleur score ROC AUC :
0.715438932092475

Rapport de classification :
              precision    recall  f1-score   support

       False       0.87      0.95      0.91     18828
        True       0.73      0.49      0.58      5364

    accuracy                           0.85     24192
   macro avg       0.80      0.72      0.74     24192
weighted avg       0.84      0.85      0.83     24192

# Autre essai avec balancing des targets
Meilleurs paramètres trouvés :
{'classifier': SVC(), 'classifier__C': 10, 'classifier__kernel': 'rbf'}
Meilleur score ROC AUC :
0.788307532891904

Rapport de classification :
              precision    recall  f1-score   support

       False       0.78      0.80      0.79      5269
        True       0.80      0.78      0.79      5324

    accuracy                           0.79     10593
   macro avg       0.79      0.79      0.79     10593
weighted avg       0.79      0.79      0.79     10593