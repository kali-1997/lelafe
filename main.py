# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import io
from func import s  

app = FastAPI()
ancient_remains_data = []

# Route to upload CSV
@app.post("/upload-ancient-remains")
async def upload_ancient_remains(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")
    contents = await file.read()

    try:
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")

    expected_columns = {"id", "region", "age", "seed"}
    if not expected_columns.issubset(df.columns):
        raise HTTPException(status_code=400, detail=f"CSV must include columns: {expected_columns}")

    ancient_remains_data.clear()
    ancient_remains_data.extend(df.to_dict(orient="records"))

    return JSONResponse(content={"message": "File uploaded successfully", "num_records": len(df)})

# Route to generate DNA
@app.get("/generate-dna")
async def generate_dna(id: str):
    record = next((r for r in ancient_remains_data if r["id"] == id), None)
    if not record:
        raise HTTPException(status_code=404, detail="ID not found")

    dna = s(id=record["id"], region=record["region"], age=int(record["age"]), dna_seed=record["seed"])

    return {"id": id, "generated_dna": dna}
