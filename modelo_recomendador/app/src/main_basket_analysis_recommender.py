from typing import Dict, List
from fastapi import Body, FastAPI, HTTPException
import pickle
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

app = FastAPI()

# Cargar las reglas de asociación desde el archivo .pkl
with open('../../models/association_rules.pkl', 'rb') as f:
    rules = pickle.load(f)

def recommend_association_rules(
    stockcode: str,
    rules: pd.DataFrame,
    n: int = 10
) -> Dict[str, List[str]]:
    """
    Recomienda productos basados en reglas de asociación.

    Args:
        stockcode (str): El código del producto para el cual se desean recomendaciones.
        rules (pd.DataFrame): DataFrame con las reglas de asociación.
        n (int, opcional): Número de recomendaciones a devolver. Por defecto es 10.

    Returns:
        Dict[str, List[str]]: Un diccionario con el stockcode y una lista de recomendaciones únicas.
    """
    product_rules = rules[rules['antecedents'].apply(lambda x: stockcode in x)]
    product_rules = product_rules.sort_values(by='lift', ascending=False).head(n)
    recommended_products = []
    
    for rule in product_rules['consequents']:
        recommended_products.extend(list(rule))
    
    # Eliminar productos duplicados
    unique_recommendations = list(set(recommended_products))
    
    return {"stockcode": stockcode, "recommendations": unique_recommendations}

class StockcodesModel(BaseModel):
    """
    Modelo para representar el stockcode y el número de recomendaciones.

    Attributes:
        stockcode (str): El valor del stockcode.
        recommendations (int): El número de recomendaciones.
    """
    stockcode: str = Field(..., description="El valor del stockcode")
    recommendations: int = Field(..., description="Número de recomendaciones")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"stockcode": "23355", "recommendations": 10},
                {"stockcode": "22064", "recommendations": 10}
            ]
        }
    )

@app.post("/recommend/")
def recommend(stockcodes: List[StockcodesModel]):
    """
    Endpoint para obtener recomendaciones para múltiples stockcodes.

    Args:
        stockcodes (List[StockcodesModel]): Lista de modelos de stockcode para los cuales obtener recomendaciones.

    Returns:
        List[Dict[str, List[str]]]: Lista de diccionarios con el stockcode y las recomendaciones únicas.
    """
    recommendations_list = []
    
    for item in stockcodes:
        stockcode = item.stockcode
        n = item.recommendations
        recommendations = recommend_association_rules(stockcode, rules, n)
        
        if recommendations["recommendations"]:
            recommendations_list.append(recommendations)
    
    return recommendations_list
