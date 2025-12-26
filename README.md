# E-Commerce Churn Prediction (PyTorch + Gradio)

## Files
- model.py: MLP model
- train.py: training + evaluation + artifact saving
- serve.py: Gradio UI for inference

## Dataset
Place the CSV next to `train.py` with this exact name:
`E Commerce Customer Insights and Churn Dataset (1).csv`

## Run
```bash
pip install torch pandas numpy scikit-learn gradio
python train.py
python serve.py
```

## Labeling
`churn = 1` if `subscription_status == "cancelled"`
`churn = 0` if `subscription_status == "active"`
Rows with `paused` are excluded for a clean binary problem.
