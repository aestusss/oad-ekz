# Импортируем готовую модель из библиотеки sklearn
from sklearn.ensemble import IsolationForest
import numpy as np

# Подготовим данные (нормальные и аномальные)
normal_data = np.random.randn(100, 2)
outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
data = np.concatenate([normal_data, outliers], axis=0)

# Создаем модель
# n_estimators - число деревьев, которое будет построено
# contamination - ожидаемый процент аномалий в данных
# оба параметра можно не указывать
model = IsolationForest(n_estimators=100, contamination=0.1)

# Обучаем модель на наших данных и предсказываем аномалии
predictions = model.fit_predict(data)

# Получаем оценки аномальностиЖ:
scores = model.decision_function(data)
