from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

# Cargar la matriz de co-ocurrencia desde el archivo .pkl
with open('../models/co_ocurrence_matrix.pkl', 'rb') as f:
    co_ocurrence_matrix = pickle.load(f)

# Cargar la matriz de similitudes desde el archivo .pkl
with open('../models/product_similarities.pkl', 'rb') as f:
    product_similarities = pickle.load(f)

@app.get("/recommend/{stockcode}")
def recommend(stockcode: str, n: int = 10):
    product_idx = co_ocurrence_matrix.columns.get_loc(stockcode)
    sim_scores = list(enumerate(product_similarities[product_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    similar_products = [co_ocurrence_matrix.columns[i[0]] for i in sim_scores]
    return {"stockcode": stockcode, "recommendations": similar_products}
