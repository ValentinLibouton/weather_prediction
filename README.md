# Models
1. `ML_SGDClassifier.joblib`
- Meilleurs paramètres trouvés :<br>
{'classifier__alpha': 0.001, 'classifier__loss': 'log_loss', 'classifier__max_iter': 1000, 'classifier__penalty': 'l2'}
- Best score: 0.8429646485528453
- Rapport de classification :<br>

    |              | precision | recall | f1-score | support |
    |--------------|-----------|--------|----------|---------|
    | False        | 0.86      | 0.95   | 0.90     | 18828   |
    | True         | 0.72      | 0.48   | 0.58     | 5364    |
    | accuracy     |           |        | 0.84     | 24192   |
    | macro avg    | 0.79      | 0.71   | 0.74     | 24192   |
    | weighted avg | 0.83      | 0.84   | 0.83     | 24192   |
- Précision du modèle sur l'ensemble des données de test : 0.8257275132275133

2. 