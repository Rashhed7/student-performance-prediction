import pandas as pd
import numpy as np

np.random.seed(42)

data = []

for i in range(400):  # 400 students
    studytime = np.random.randint(1, 5)
    failures = np.random.randint(0, 4)
    absences = np.random.randint(0, 30)
    G1 = np.random.randint(5, 20)
    G2 = np.random.randint(5, 20)

    # Logic for final grade (realistic pattern)
    score = (G1 + G2)/2 + studytime*2 - failures*3 - absences*0.1

    final_grade = 1 if score >= 10 else 0

    data.append([studytime, failures, absences, G1, G2, final_grade])

df = pd.DataFrame(data, columns=[
    'studytime','failures','absences','G1','G2','final_grade'
])

df.to_csv("dataset.csv", index=False)

print("Dataset created successfully!")