# Nexariza ML Model Serving System

**Live API:** https://nexariza-mlops-deployment-week5-production.up.railway.app
**Interactive docs:** https://nexariza-mlops-deployment-week5-production.up.railway.app/docs

This is my Week 5 project. I took the churn predictor I built in Week 2 and actually turned
it into a deployable service instead of just a notebook demo. It is wrapped in a FastAPI REST
API, containerized with Docker, tracked with MLflow, and has a monitoring dashboard on top.
It is deployed live on Railway right now at the link above.

## What it actually does

1. Serves the Week 2 churn model as a REST API with a `/predict` endpoint
2. Auto generates interactive API documentation through Swagger UI at `/docs`
3. Logs every prediction to a file so usage can be monitored over time
4. Tracks the model training run (params and metrics) in MLflow
5. Has a Streamlit dashboard that shows API health, prediction stats, and the MLflow run history

## Being upfront about a limitation

I did not have Docker available in the environment I built this in, so I could not build and
run the container myself before deploying it. The Dockerfile followed the standard pattern for
a Python FastAPI service, and it turned out to work correctly the first time Railway built it,
but I want to be honest that I only had the plain Python version tested and confirmed working
before that point, not the actual container.

## Project structure

```
.
├── main.py                    (the FastAPI app)
├── monitoring_dashboard.py    (Streamlit dashboard for monitoring the API)
├── Dockerfile
├── requirements.txt
├── models/
│   ├── nexariza_churn_model.pkl
│   ├── model_columns.pkl
│   └── recommendation_map.pkl
└── mlflow_tracking/
    ├── train_and_log.py       (retrains the model and logs the run to MLflow)
    └── mlflow.db              (the MLflow tracking database with the logged run)
```

## Running it locally

Start the API:

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Then visit `http://localhost:8000/docs` for the interactive Swagger documentation, where you
can try out the `/predict` endpoint directly in the browser.

In a separate terminal, run the monitoring dashboard:

```bash
streamlit run monitoring_dashboard.py
```

To retrain the model and log a fresh run to MLflow:

```bash
cd mlflow_tracking
python train_and_log.py
```

## API endpoints

- `GET /` - basic service info
- `GET /health` - health check, confirms the model is loaded
- `POST /predict` - takes client details, returns churn probability, risk level, and top
  recommendations
- `GET /stats` - basic monitoring stats pulled from the prediction log

## Testing it

I tested this locally before calling it done. Sent a high risk client profile (short tenure,
low engagement, late payment, no account manager) and got back a 96% churn probability
correctly flagged as High risk, with recommendations that matched the actual risk factors.
Sent a low risk profile (long tenure, high engagement, good NPS) and got back under 1%
probability, correctly flagged as Low risk. The `/stats` endpoint correctly aggregated both
predictions afterward, and the Swagger docs page loaded and worked as expected.

## Deploying it

I could not create an account or deploy this myself, since that needs a real login tied to a
real person. I deployed it myself once the code was ready, using Railway. Connected my GitHub
repo directly, Railway found the Dockerfile automatically, built the image, and it went live
without needing any extra config. The Dockerfile worked on the first try, which was a relief
since I could not test the actual container build myself beforehand, only the plain Python
version.

Hugging Face Spaces was my original plan, but their Docker Spaces now require a paid PRO plan,
so I switched to Railway instead, which was in the brief as an accepted alternative and has a
usable free tier.

## What I learned

Turning a notebook model into an API is a genuinely different skill than building the model
itself. Most of the actual work here was in input validation, since a REST API needs to
reject bad input gracefully instead of just crashing, which is why there is a fair amount of
Pydantic validation in main.py that was not needed anywhere in the Week 2 notebook.

MLflow ran into a version issue right away, since the newer version does not allow the old
plain file based tracking store anymore and wants a database backend instead. Switching to a
local SQLite database fixed it, and honestly a database backend is a more realistic setup than
loose files anyway.

I also had to switch from `mlflow.sklearn.log_model` to `mlflow.xgboost.log_model`, since
sklearn's model saving does a safety check on the model's internal types and XGBoost's model
object hit that check. MLflow has separate logging functions for different model libraries,
and it matters which one you use.

Not being able to test the actual Docker build was the one gap I could not close myself.
Everything up to the container boundary is tested and working, but the containerized version
and the live deployment both need pieces I do not have access to in this environment.

## Tech stack

Python, FastAPI, Uvicorn, Docker, MLflow, XGBoost, Streamlit
