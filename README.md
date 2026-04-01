# 🏦 Bank Customer Churn Prediction

This project predicts whether a bank customer is likely to churn or stay.

## Files
- `app.py` - Streamlit app
- `main.py` - FastAPI API
- `churn_model.pkl` - trained model
- `requirements.txt` - dependencies
- `Churn_Modelling.csv` - dataset
- `churn modeling.ipynb` - notebook

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Run API

```bash
uvicorn main:app --reload
```

## Deploy on Streamlit
- Push all files to GitHub
- Open Streamlit Community Cloud
- Select the repository
- Choose `app.py`
- Deploy

## Author
Saniya Shekhar Dhule
