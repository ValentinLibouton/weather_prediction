# ML
## Sans ROC AUC
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
   macro avg       0.86      0.73      0.77      2050<br>
weighted avg       0.87      0.88      0.86      2050

## Avec ROC AUC
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
   macro avg       0.79      0.73      0.75      2050<br>
weighted avg       0.85      0.86      0.85      2050

## ROC AUC + 50% Balanced
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
   macro avg       0.80      0.80      0.80       812<br>
weighted avg       0.80      0.80      0.80       812

## Autre essai
Meilleurs paramètres trouvés :
{'classifier': SVC(), 'classifier__C': 100, 'classifier__kernel': 'rbf'}
Meilleur score ROC AUC :
0.715438932092475

Rapport de classification :
              precision    recall  f1-score   support

       False       0.87      0.95      0.91     18828
        True       0.73      0.49      0.58      5364

    accuracy                           0.85     24192
   macro avg       0.80      0.72      0.74     24192<br>
weighted avg       0.84      0.85      0.83     24192

## Autre essai avec balancing des targets
Meilleurs paramètres trouvés :
{'classifier': SVC(), 'classifier__C': 10, 'classifier__kernel': 'rbf'}
Meilleur score ROC AUC :
0.788307532891904

Rapport de classification :
              precision    recall  f1-score   support

       False       0.78      0.80      0.79      5269
        True       0.80      0.78      0.79      5324

    accuracy                           0.79     10593
   macro avg       0.79      0.79      0.79     10593<br>
weighted avg       0.79      0.79      0.79     10593

# Résultats en Deep Learning
## DL 1
```python
# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création du modèle
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))

# Définition des rappels ReduceLROnPlateau et EarlyStopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Compilation du modèle avec un learning rate initial
initial_learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Entraînement du modèle avec les rappels ReduceLROnPlateau et EarlyStopping
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[reduce_lr, early_stopping])

# Évaluation du modèle sur les données de test
loss, accuracy = model.evaluate(X_test, y_test)
print("Perte sur les données de test :", loss)
print("Précision sur les données de test :", accuracy)
```
Perte sur les données de test : 0.38624513149261475<br>
Précision sur les données de test : 0.8316798806190491

## DL 2
```python
# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création du modèle
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compilation du modèle
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# Entraînement du modèle
model.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_test, y_test))

# Évaluation du modèle sur les données de test
loss, accuracy = model.evaluate(X_test, y_test)
print("Perte sur les données de test :", loss)
print("Précision sur les données de test :", accuracy)
```
Perte sur les données de test : 0.41624686121940613<br>
Précision sur les données de test : 0.8084903955459595

## DL 3
```python
print("Précision sur les données de test :", accuracy)
#%%
# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création du modèle
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))

# Définition des rappels ReduceLROnPlateau et EarlyStopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Compilation du modèle avec un learning rate initial
initial_learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Entraînement du modèle avec les rappels ReduceLROnPlateau et EarlyStopping
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[reduce_lr, early_stopping])

# Évaluation du modèle sur les données de test
loss, accuracy = model.evaluate(X_test, y_test)
print("Perte sur les données de test :", loss)
print("Précision sur les données de test :", accuracy)
```
Perte sur les données de test : 0.5290846824645996<br>
Précision sur les données de test : 0.7782738208770752

## DL 4
```python
# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création du modèle
model = Sequential()
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01), input_dim=X_train.shape[1]))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))

# Définition des rappels ReduceLROnPlateau et EarlyStopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Compilation du modèle avec un learning rate initial
initial_learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Entraînement du modèle avec les rappels ReduceLROnPlateau et EarlyStopping
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[reduce_lr, early_stopping])

# Évaluation du modèle sur les données de test
loss, accuracy = model.evaluate(X_test, y_test)
print("Perte sur les données de test :", loss)
print("Précision sur les données de test :", accuracy)
```
Perte sur les données de test : 0.5290907621383667<br>
Précision sur les données de test : 0.7782738208770752

## DL 5
```python
# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création du modèle
model = Sequential()
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01), input_dim=X_train.shape[1]))
model.add(Dropout(0.4))
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))

# Définition des rappels ReduceLROnPlateau et EarlyStopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Compilation du modèle avec un learning rate initial
initial_learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Entraînement du modèle avec les rappels ReduceLROnPlateau et EarlyStopping
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[reduce_lr, early_stopping])

# Évaluation du modèle sur les données de test
loss, accuracy = model.evaluate(X_test, y_test)
print("Perte sur les données de test :", loss)
print("Précision sur les données de test :", accuracy)
```
Perte sur les données de test : 0.5291100144386292<br>
Précision sur les données de test : 0.7782738208770752

## DL 6
```python
# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création du modèle
model = Sequential()
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01), input_dim=X_train.shape[1]))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.4))
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))

# Définition des rappels ReduceLROnPlateau et EarlyStopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Compilation du modèle avec un learning rate initial
initial_learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Entraînement du modèle avec les rappels ReduceLROnPlateau et EarlyStopping
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[reduce_lr, early_stopping])

# Évaluation du modèle sur les données de test
loss, accuracy = model.evaluate(X_test, y_test)
print("Perte sur les données de test :", loss)
print("Précision sur les données de test :", accuracy)
```
Perte sur les données de test : 0.5291064381599426<br>
Précision sur les données de test : 0.7782738208770752

## DL 7
```python
# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création du modèle
model = Sequential()
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01), input_dim=X_train.shape[1]))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.4))
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))

# Définition des rappels ReduceLROnPlateau et EarlyStopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Compilation du modèle avec un learning rate initial
initial_learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Entraînement du modèle avec les rappels ReduceLROnPlateau et EarlyStopping
history = model.fit(X_train, y_train, epochs=100, batch_size=128, validation_data=(X_test, y_test), callbacks=[reduce_lr, early_stopping])

# Évaluation du modèle sur les données de test
loss, accuracy = model.evaluate(X_test, y_test)
print("Perte sur les données de test :", loss)
print("Précision sur les données de test :", accuracy)
```
Perte sur les données de test : 0.5291208028793335<br>
Précision sur les données de test : 0.7782738208770752