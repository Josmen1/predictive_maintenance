import os
import io
import sys
import pandas as pd
import certifi

ca = certifi.where()

from dotenv import load_dotenv

load_dotenv()
mongo_db_url = os.getenv("MONGO_DB_URL")
import pymongo

client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import Response
from uvicorn import run as app_run
from starlette.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from predictive_maintenance.logging.logger import get_logger
from predictive_maintenance.exception.exception import PredictiveMaintenanceException
from predictive_maintenance.pipeline.training_pipeline import TrainingPipeline
from predictive_maintenance.utils.main_utils.general_utils import load_object
from predictive_maintenance.constants.training_pipeline import (
    DATA_INGESTION_DATABASE_NAME,
    COLLECTION_NAME,
)
from predictive_maintenance.utils.ml_utils.model.predictor import ModelPredictor

database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[COLLECTION_NAME]


logger = get_logger(__name__)
app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="./templates")


@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def train_route():
    try:
        training_pipeline = TrainingPipeline()
        training_pipeline.run_pipeline()
        return Response("Training successful !!")
    except Exception as e:
        raise PredictiveMaintenanceException(e, sys) from e


EXPECTED_COLS = (
    ["unit_number", "time_in_cycles", "ops_1", "ops_2", "ops_3"]
    + [f"sensor_{i}" for i in range(1, 22)]  # sensor_1 .. sensor_21
    + ["subset"]  # keep if your preprocessor expects it; otherwise you can drop it
    + ["split"]  # keep if your preprocessor expects it; otherwise you can drop it
)

ID_COLS = ["unit_number", "time_in_cycles", "subset", "split"]

# Optional fallback if your preprocessor doesn't expose feature_names_in_
# Adjust if your training schema differs
FALLBACK_FEATURE_COLS = ["ops_1", "ops_2", "ops_3"] + [
    f"sensor_{i}" for i in range(1, 22)
]  # sensor_1..sensor_21


@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        # --- Robust read: TSV vs CSV ---
        head_bytes = await file.read(4096)
        head = head_bytes.decode("utf-8", errors="ignore")
        sep = "\t" if ("\t" in head and "," not in head) else ","
        await file.seek(0)
        df = pd.read_csv(file.file, sep=sep)

        # Normalize headers (trim spaces)
        df.columns = [c.strip() for c in df.columns]

        # Ensure ID columns exist (these are only for display/output)
        missing_ids = [c for c in ID_COLS if c not in df.columns]
        if missing_ids:
            raise HTTPException(
                status_code=400, detail=f"Missing required ID columns: {missing_ids}"
            )

        # Load artifacts
        preprocessor = load_object(file_path="final_model/preprocessor.joblib")
        model = load_object(file_path="final_model/model.joblib")
        predictor = ModelPredictor(preprocessor=preprocessor, model=model)

        # Figure out the exact feature columns the preprocessor expects
        if hasattr(preprocessor, "feature_names_in_"):
            expected_features = list(preprocessor.feature_names_in_)
        else:
            # Fallback to your known training schema
            expected_features = FALLBACK_FEATURE_COLS

        # Sanity check features present
        missing_feats = [c for c in expected_features if c not in df.columns]
        if missing_feats:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Your input file is missing features required by the trained pipeline: "
                    f"{missing_feats}"
                ),
            )

        # Split IDs (for output) and Features (for model)
        df_ids = df[ID_COLS].copy()
        X = df[expected_features].copy()  # ONLY features go to the pipeline/model

        # --- Predict ---
        # IMPORTANT: do not pass ID columns into the predictor/model
        # Preprocessor gets all columns including IDs as it needs them
        X_transformed = predictor.transform(df)
        # We then select only the expected features after transformation
        X = X_transformed.drop(columns=ID_COLS, errors="ignore", axis=1)
        y_pred = predictor.predict(X)

        # Build final output: ID columns + predictions (no features)
        out = df_ids.copy()
        out["Predictions"] = y_pred

        # Save and render
        os.makedirs("prediction_output", exist_ok=True)
        out.to_csv("prediction_output/predictions.csv", index=False)

        table_html = out.to_html(classes="table table-striped", index=False)
        return templates.TemplateResponse(
            "table.html", {"request": request, "table": table_html}
        )

    except PredictiveMaintenanceException:
        raise
    except HTTPException:
        raise
    except Exception as e:
        # Wrap unexpected errors into your domain exception for consistency
        raise PredictiveMaintenanceException(e, sys) from e
