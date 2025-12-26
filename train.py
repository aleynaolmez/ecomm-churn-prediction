import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from model import ChurnMLP

CSV_PATH = "E Commerce Customer Insights and Churn Dataset (1).csv"
ARTIFACT_PATH = "churn_artifact.pt"

NUM_COLS = [
    "age",
    "cancellations_count",
    "purchase_frequency",
    "unit_price",
    "quantity",
    "total_order_value",
    "days_since_signup",
    "days_since_last_purchase",
]

CAT_COLS = [
    "country",
    "gender",
    "preferred_category",
    "category",
    "product_name",
]


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["signup_date", "last_purchase_date", "order_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = parse_dates(df)

    # label: churn=1 if cancelled, churn=0 if active
    df = df[df["subscription_status"].isin(["active", "cancelled"])].copy()
    df["churn"] = (df["subscription_status"] == "cancelled").astype(np.float32)

    # engineered numeric features
    df["total_order_value"] = df["unit_price"] * df["quantity"]

    # Use order_date as reference point for "recency" / "tenure"
    df["days_since_signup"] = (df["order_date"] - df["signup_date"]).dt.days
    df["days_since_last_purchase"] = (df["order_date"] - df["last_purchase_date"]).dt.days

    # Fill missing numeric date-diffs with median (robust)
    for col in ["days_since_signup", "days_since_last_purchase"]:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Keep only required columns
    keep = NUM_COLS + CAT_COLS + ["churn"]
    df = df[keep].copy()

    # Basic cleaning for categorical
    for c in CAT_COLS:
        df[c] = df[c].fillna("Unknown")

    return df


def one_hot_align(train_df: pd.DataFrame, test_df: pd.DataFrame):
    X_train_cat = pd.get_dummies(train_df[CAT_COLS], drop_first=False)
    X_test_cat = pd.get_dummies(test_df[CAT_COLS], drop_first=False)

    # align columns (test might miss some categories)
    X_train_cat, X_test_cat = X_train_cat.align(
        X_test_cat, join="left", axis=1, fill_value=0
    )
    return X_train_cat, X_test_cat


def standardize(train_num: pd.DataFrame, test_num: pd.DataFrame):
    mean = train_num.mean(axis=0)
    std = train_num.std(axis=0).replace(0, 1.0)
    train_s = (train_num - mean) / std
    test_s = (test_num - mean) / std
    return train_s, test_s, mean.to_dict(), std.to_dict()


def compute_pos_weight(y_train_t: torch.Tensor) -> torch.Tensor:
    """
    y_train_t: (N,1) float 0/1
    pos_weight = (#neg / #pos)
    """
    pos = y_train_t.sum()
    neg = y_train_t.numel() - pos
    return neg / (pos + 1e-8)


def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"CSV not found: {CSV_PATH}\nPut the dataset next to train.py or update CSV_PATH."
        )

    raw = pd.read_csv(CSV_PATH)
    df = build_features(raw)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["churn"]
    )

    # numeric + categorical
    X_train_num = train_df[NUM_COLS].astype(np.float32)
    X_test_num = test_df[NUM_COLS].astype(np.float32)

    X_train_num, X_test_num, num_mean, num_std = standardize(X_train_num, X_test_num)

    X_train_cat, X_test_cat = one_hot_align(train_df, test_df)
    cat_columns = X_train_cat.columns.tolist()

    X_train = np.concatenate(
        [X_train_num.values, X_train_cat.values.astype(np.float32)], axis=1
    )
    X_test = np.concatenate(
        [X_test_num.values, X_test_cat.values.astype(np.float32)], axis=1
    )

    # Labels: keep (N,1) shape
    y_train = train_df["churn"].values.astype(np.float32).reshape(-1, 1)
    y_test = test_df["churn"].values.astype(np.float32).reshape(-1, 1)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    model = ChurnMLP(input_dim=X_train_t.shape[1])

    #  imbalance fix
    pos_weight = compute_pos_weight(y_train_t).to(torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    pos_count = float(y_train_t.sum().item())
    neg_count = float(y_train_t.numel() - y_train_t.sum().item())
    print(f"Train positives (churn=1): {pos_count:.0f}")
    print(f"Train negatives (churn=0): {neg_count:.0f}")
    print(f"pos_weight (neg/pos): {float(pos_weight.item()):.4f}\n")

    loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)

    epochs = 40
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for xb, yb in loader:
            #  SHAPE FIX (en kritik kısım)
            logits = model(xb).view(-1, 1)   # (B,1) garanti
            yb = yb.view(-1, 1)              # (B,1) garanti

            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item() * xb.size(0)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:02d} | Loss: {total/len(loader.dataset):.4f}")

    # evaluation
    model.eval()
    with torch.no_grad():
        logits_test = model(X_test_t).view(-1, 1)        # (N,1) garanti
        probs = torch.sigmoid(logits_test).cpu().numpy() # (N,1)

    preds = (probs >= 0.5).astype(int)

    # sklearn wants 1D
    y_test_1d = y_test.reshape(-1)
    preds_1d = preds.reshape(-1)

    acc = accuracy_score(y_test_1d, preds_1d)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test_1d, preds_1d, average="binary", zero_division=0
    )

    print("\n=== Test Metrics ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1       : {f1:.4f}")

    artifact = {
        "model_state": model.state_dict(),
        "num_cols": NUM_COLS,
        "cat_cols": CAT_COLS,
        "cat_onehot_cols": cat_columns,
        "num_mean": num_mean,
        "num_std": num_std,
        "threshold": 0.5,
        "pos_weight": float(pos_weight.item()),
    }
    torch.save(artifact, ARTIFACT_PATH)
    print(f"\nSaved -> {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()

