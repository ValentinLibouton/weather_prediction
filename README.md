# Models
1. ## `ML_SGDClassifier.joblib`
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

2. ## `best_model_in_deep_learning.h5` - une epoch
- Meilleurs paramètres trouvés :<br>
{'model__activation': 'relu', 'model__dropout_rate': 0.3, 'model__kernel_regularizer': None, 'model__learning_rate': 0.001}
- Best score: 0.8442659139633178

    |              | precision | recall | f1-score | support |
    |--------------|-----------|--------|----------|---------|
    | False        | 0.87      | 0.95   | 0.90     | 18828   |
    | True         | 0.72      | 0.48   | 0.58     | 5364    |
    | accuracy     |           |        | 0.84     | 24192   |
    | macro avg    | 0.79      | 0.72   | 0.74     | 24192   |
    | weighted avg | 0.83      | 0.84   | 0.83     | 24192   |
- Précision sur les données de test : 0.8444940476190477

3. ## `best_model_in_deep_learning.h5` - 100 epochs
- Meilleurs paramètres trouvés :<br>
{'model__activation': 'relu', 'model__dropout_rate': 0.3, 'model__kernel_regularizer': None, 'model__learning_rate': 0.001}
  - Best score: 0.8448859333992005

    |              | precision | recall | f1-score | support |
    |--------------|-----------|--------|----------|---------|
    | False        | 0.86      | 0.96   | 0.90     | 18828   |
    | True         | 0.74      | 0.45   | 0.56     | 5364    |
    | accuracy     |           |        | 0.84     | 24192   |
    | macro avg    | 0.80      | 0.70   | 0.73     | 24192   |
    | weighted avg | 0.83      | 0.84   | 0.83     | 24192   |
- Précision sur les données de test : 0.843584656084656

4. ## `ML_SVClassifier.joblib`
- Meilleurs paramètres trouvés :<br>
{'classifier': SVC(), 'classifier__C': 10, 'classifier__kernel': 'rbf'}
- Best score: 0.8494750705633111

    |              | precision | recall | f1-score | support |
    |--------------|-----------|--------|----------|---------|
    | False        | 0.86      | 0.96   | 0.91     | 18828   |
    | True         | 0.77      | 0.45   | 0.57     | 5364    |
    | accuracy     |           |        | 0.85     | 24192   |
    | macro avg    | 0.81      | 0.71   | 0.74     | 24192   |
    | weighted avg | 0.84      | 0.85   | 0.83     | 24192   |
- Précision sur les données de test : 0.836102843915344

5. ## `best_model_in_deep_learning_balanced.h5` - 100 epochs + balanced
- Meilleurs paramètres trouvés :<br>
{'model__activation': 'relu', 'model__dropout_rate': 0.3, 'model__kernel_regularizer': None, 'model__learning_rate': 0.01}
- Best score: 0.7865973353385926

    |              | precision | recall | f1-score | support |
    |--------------|-----------|--------|----------|--|
    | False        | 0.77      | 0.83   | 0.80     | 18958 |
    | True         | 0.81      | 0.75   | 0.78     | 18834 |
    | accuracy     |           |        | 0.79     | 37792 |
    | macro avg    | 0.79      | 0.79   | 0.79     | 37792 |
    | weighted avg | 0.79      | 0.79   | 0.79     | 37792 |
- Précision sur les données de test : 0.7903259949195597

6. ## `ML_SGDClassifier_balanced.joblib` - balanced
- Meilleurs paramètres trouvés :<br>
{'classifier__alpha': 0.001, 'classifier__loss': 'hinge', 'classifier__max_iter': 1000, 'classifier__penalty': 'l2'}
- Best score: 0.7781747080124555

    |              | precision | recall | f1-score | support |
    |--------------|-----------|--------|----------|--|
    | False        | 0.77      | 0.79   | 0.78     | 18958 |
    | True         | 0.78      | 0.76   | 0.77     | 18834 |
    | accuracy     |           |        | 0.78     | 37792 |
    | macro avg    | 0.78      | 0.78   | 0.78     | 37792 |
    | weighted avg | 0.78      | 0.78   | 0.78     | 37792 |
- Précision sur les données de test : 0.7492061812023709

7. ## `best_model_in_deep_learning_change_layers.h5`
- Meilleurs paramètres trouvés :<br>
{'model__activation': 'relu', 'model__batch_size': 16, 'model__dropout_rate': 0.3, 'model__kernel_regularizer': None, 'model__learning_rate': 0.001}
- Best score: 0.7868949890136718

    |              | precision | recall | f1-score | support |
    |--------------|-----------|--------|----------|--|
    | False        | 0.81      | 0.76   | 0.78     | 18958 |
    | True         | 0.77      | 0.82   | 0.79     | 18834 |
    | accuracy     |           |        | 0.79     | 37792 |
    | macro avg    | 0.79      | 0.79   | 0.79     | 37792 |
    | weighted avg | 0.79      | 0.79   | 0.79     | 37792 |
- Précision sur les données de test : 0.7881297629127858
- Temps d'exécution: 22h42 pour 810 fits
```python
GridSearchCV(cv=5,
             estimator=Pipeline(steps=[('scaler', StandardScaler()),
                                       ('model',
                                        <keras.wrappers.scikit_learn.KerasClassifier object at 0x7f0e19762820>)]),
             param_grid={'model__activation': ['relu', 'sigmoid', 'tanh'],
                         'model__batch_size': [16, 32, 64],
                         'model__dropout_rate': [0.3, 0.4, 0.5],
                         'model__kernel_regularizer': [None,
                                                       <keras.regularizers.L2 object at 0x7f0e19762520>],
                         'model__learning_rate': [0.001, 0.01, 0.1]},
             verbose=2)
```