import json
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression

from movie_recs.serve import get_db_connection

if __name__ == "__main__":
    conn = get_db_connection()
    data = conn.execute("SELECT id, rating from movies;").fetchall()
    embeddings = json.load(open("./bert-plot-features.json"))  # or s3

    X = np.array([embeddings[str(i)] for i, _ in data])
    y = np.array([int(d[1] >= 7) for d in data])

    model = LogisticRegression()
    model.fit(X, y)
    print(model)
    print("Score", model.score(X, y))

    with open('./base-logistic-regression.pkl', 'wb') as f:
        pickle.dump(model, f)