
import pandas as pd
from catboost import datasets


def load_titanic():
    df_train, df_test = datasets.titanic()  # допущение: возвращает (train, test) как DataFrame
    df = pd.concat([df_train, df_test], ignore_index=True)
    # Нормализуем имена столбцов под задачу
    cols = {c.lower(): c for c in df.columns}
    # частые варианты имен у источников
    mapping = {
        'pclass': ['Pclass', 'pclass'],
        'sex': ['Sex', 'sex'],
        'age': ['Age', 'age'],
    }

    def pick(name):
        for k in mapping[name]:
            if k in df.columns: return k
        for k in mapping[name]:
            if k in cols: return cols[k]
        raise KeyError(f"column '{name}' not found")

    return df[[pick('pclass'), pick('sex'), pick('age')]].rename(
        columns=lambda c: {'Pclass': 'Pclass', 'pclass': 'Pclass', 'Sex': 'Sex', 'sex': 'Sex', 'Age': 'Age',
                           'age': 'Age'}.get(c, c)
    )



df = load_titanic()
df.to_csv("data/titanic.csv", index=False)
print(df.head())