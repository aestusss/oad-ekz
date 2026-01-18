# Импортируем готовую модель из библиотеки sklearn
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import numpy as np

# Создаем данные
X = np.random.randn(200, 2)
clf = IsolationForest(n_estimators=100, contamination=0.1).fit(X)

# Создаем сетку для отрисовки границ
xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Визуализация
plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu)  # Цветовая карта оценок
plt.scatter(X[:, 0], X[:, 1], c="white", edgecolors="k")  # Точки данных
plt.title("Граница доверия модели")
plt.colorbar(label="Anomaly Score")
plt.show()
