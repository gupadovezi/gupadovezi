# Instalar pacotes necessários (caso esteja em ambiente novo)
# !pip install ISLP scipy statsmodels

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.datasets import get_rdataset
from sklearn.model_selection import (
    train_test_split, ShuffleSplit, KFold, GridSearchCV, cross_validate
)
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder
from ISLP import load_data, confusion_table
from ISLP.models import ModelSpec as MS

# Carregar o conjunto de dados Carseats
Carseats = load_data('Carseats')

# Criar variável binária para Sales > 8
High = np.where(Carseats.Sales > 8, "Yes", "No")

# Codificar a variável High como 0 e 1
le = LabelEncoder()
High_encoded = le.fit_transform(High)  # "Yes" -> 1, "No" -> 0

# Definir especificação do modelo sem intercepto
model = MS(Carseats.columns.drop('Sales'), intercept=False)
D = model.fit_transform(Carseats)
feature_names = list(D.columns)
X = np.asarray(D)

# Treinar árvore de decisão com entropia e profundidade máxima 3
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
clf.fit(X, High_encoded)

# Avaliar o modelo com acurácia e log loss
train_acc = accuracy_score(High_encoded, clf.predict(X))
resid_dev = log_loss(High_encoded, clf.predict_proba(X))

print("Training Accuracy:", train_acc)
print("Residual Deviance (Log Loss):", resid_dev)

# Visualizar a árvore de decisão
fig, ax = plt.subplots(figsize=(12, 12))
plot_tree(clf, feature_names=feature_names, class_names=le.classes_, filled=True, ax=ax)
plt.show()

# Exibir a árvore em texto
print(export_text(clf, feature_names=feature_names, show_weights=True))

# Validação cruzada com 200 amostras de teste
validation = ShuffleSplit(n_splits=1, test_size=200, random_state=0)
results = cross_validate(clf, D, High_encoded, cv=validation)
print("Validation Accuracy (ShuffleSplit):", results['test_score'][0])

# Separar dados em treino e teste (50%)
X_train, X_test, y_train, y_test = train_test_split(
    X, High_encoded, test_size=0.5, random_state=0
)

# Treinar modelo completo para poda posterior
clf = DecisionTreeClassifier(criterion='entropy', random_state=0)
clf.fit(X_train, y_train)

# Caminho de poda (cost complexity pruning)
ccp_path = clf.cost_complexity_pruning_path(X_train, y_train)

# Grid search para encontrar melhor alpha
kfold = KFold(n_splits=10, shuffle=True, random_state=1)
grid = GridSearchCV(
    estimator=clf,
    param_grid={'ccp_alpha': ccp_path.ccp_alphas},
    cv=kfold,
    scoring='accuracy'
)
grid.fit(X_train, y_train)

print("Best cross-validated accuracy:", grid.best_score_)
print("Best alpha:", grid.best_params_['ccp_alpha'])

# Visualizar a árvore com melhor alpha
best_model = grid.best_estimator_

fig, ax = plt.subplots(figsize=(12, 12))
plot_tree(best_model, feature_names=feature_names, class_names=le.classes_, filled=True, ax=ax)
plt.show()

# Número de folhas da árvore final
print("Number of leaves in pruned tree:", best_model.tree_.n_leaves)

# Acurácia no conjunto de teste
test_acc = accuracy_score(y_test, best_model.predict(X_test))
print("Test Accuracy (Pruned Tree):", test_acc)

# Matriz de confusão
confusion = confusion_table(best_model.predict(X_test), y_test)
print("Confusion Table:\n", confusion)

# (Opcional) Plotar curva alpha vs. acurácia
alphas = grid.cv_results_['param_ccp_alpha'].data
scores = grid.cv_results_['mean_test_score']

plt.figure(figsize=(8, 5))
plt.plot(alphas, scores, marker='o')
plt.xlabel('ccp_alpha')
plt.ylabel('Cross-validated Accuracy')
plt.title('Alpha vs Accuracy (Pruning Path)')
plt.grid(True)
plt.show()
