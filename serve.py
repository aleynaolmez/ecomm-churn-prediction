import os
import pandas as pd
import numpy as np
import torch
import gradio as gr

from model import ChurnMLP

ARTIFACT_PATH = "churn_artifact.pt"

def load_artifact():
    if not os.path.exists(ARTIFACT_PATH):
        raise FileNotFoundError(
            f"Artifact not found: {ARTIFACT_PATH}\n"
            f"Run: python train.py (it will create {ARTIFACT_PATH})"
        )
    artifact = torch.load(ARTIFACT_PATH, map_location="cpu")
    model = ChurnMLP(input_dim=len(artifact["num_cols"]) + len(artifact["cat_onehot_cols"]))
    model.load_state_dict(artifact["model_state"])
    model.eval()
    return artifact, model

artifact, model = load_artifact()

def _standardize_num(num_df: pd.DataFrame) -> pd.DataFrame:
    mean = pd.Series(artifact["num_mean"])
    std = pd.Series(artifact["num_std"]).replace(0, 1.0)
    return (num_df - mean) / std

def _one_hot_cat(cat_df: pd.DataFrame) -> pd.DataFrame:
    x = pd.get_dummies(cat_df, drop_first=False)
    # align to training one-hot columns
    cols = artifact["cat_onehot_cols"]
    x = x.reindex(columns=cols, fill_value=0)
    return x

def predict(age, country, gender, cancellations_count, purchase_frequency,
            unit_price, quantity, preferred_category, category, product_name,
            days_since_signup, days_since_last_purchase):
    # build numeric features, including engineered total_order_value
    total_order_value = float(unit_price) * float(quantity)

    num = pd.DataFrame([{
        "age": float(age),
        "cancellations_count": float(cancellations_count),
        "purchase_frequency": float(purchase_frequency),
        "unit_price": float(unit_price),
        "quantity": float(quantity),
        "total_order_value": float(total_order_value),
        "days_since_signup": float(days_since_signup),
        "days_since_last_purchase": float(days_since_last_purchase),
    }], columns=artifact["num_cols"])

    cat = pd.DataFrame([{
        "country": str(country),
        "gender": str(gender),
        "preferred_category": str(preferred_category),
        "category": str(category),
        "product_name": str(product_name),
    }], columns=artifact["cat_cols"])

    num_s = _standardize_num(num)
    cat_oh = _one_hot_cat(cat)

    X = np.concatenate([num_s.values.astype(np.float32), cat_oh.values.astype(np.float32)], axis=1)
    X_t = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        prob = torch.sigmoid(model(X_t)).item()

    decision = "CHURN (iptal etme olası)" if prob >= artifact["threshold"] else "NOT CHURN (aktif kalma olası)"
    return {
        "churn_probability": round(prob, 4),
        "decision": decision
    }

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="age", value=30),
        gr.Textbox(label="country", value="USA"),
        gr.Dropdown(["Female", "Male", "Unknown"], label="gender", value="Female"),
        gr.Number(label="cancellations_count", value=0),
        gr.Number(label="purchase_frequency", value=10),
        gr.Number(label="unit_price", value=50),
        gr.Number(label="quantity", value=1),
        gr.Textbox(label="preferred_category", value="Electronics"),
        gr.Textbox(label="category", value="Electronics"),
        gr.Textbox(label="product_name", value="Headphones"),
        gr.Number(label="days_since_signup", value=200),
        gr.Number(label="days_since_last_purchase", value=30),
    ],
    outputs=gr.JSON(label="prediction"),
    title="E-Commerce Churn Prediction (Tabular DL)",
    description="Dataset'teki subscription_status üzerinden churn etiketi üretilmiştir (cancelled=1, active=0)."
)

if __name__ == "__main__":
    demo.launch(share=True)
