from fastapi import FastAPI, UploadFile, File, HTTPException, Body  # <-- Add Body here
from fastapi.responses import JSONResponse
import pandas as pd
import io
from func import s  # Your DNA generation function
from pydantic import BaseModel
import httpx
from google import genai

# Initialize FastAPI app
app = FastAPI()

# Store the data
ancient_remains_data = []


# Upload DNA_Dataa
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

    # Filter
    ancient_remains_data.clear()
    ancient_remains_data.extend(df.to_dict(orient="records"))

    return JSONResponse(content={"message": "File uploaded successfully", "num_records": len(df)})


# DnA_Generation
@app.get("/generate-dna")
async def generate_dna(id: str, start: int = 0, length: int = 1000):
    record = next((r for r in ancient_remains_data if r["id"] == id), None)
    if not record:
        raise HTTPException(status_code=404, detail="ID not found")

    # Generate DNA sequence using S function
    dna = s(id=record["id"], region=record["region"], age=int(record["age"]), dna_seed=record["seed"])
    partial_dna = dna[start:start + length]

    return {
        "id": id,
        "start": start,
        "length": len(partial_dna),
        "partial_dna": partial_dna,
        "total_length": len(dna)
    }


# Utility function to compute k-mers from DNA sequence
def get_kmers(sequence: str, k: int) -> set:
    return set(sequence[i:i+k] for i in range(len(sequence) - k + 1))


# Calculating similarity score basd on k-mer overlaps
def fast_similarity_score(seq1: str, seq2: str, k: int = 5) -> float:
    kmers1 = get_kmers(seq1, k)
    kmers2 = get_kmers(seq2, k)
    intersection = kmers1.intersection(kmers2)
    union = kmers1.union(kmers2)
    if not union:
        return 0.0
    return round(len(intersection) / len(union), 4)


# Comparison of DNA
@app.get("/compare-dna")
async def compare_dna(id1: str, id2: str, length: int = 2000, k: int = 5):
    rec1 = next((r for r in ancient_remains_data if r["id"] == id1), None)
    rec2 = next((r for r in ancient_remains_data if r["id"] == id2), None)

    if not rec1 or not rec2:
        raise HTTPException(status_code=404, detail="One or both IDs not found")

    dna1 = s(id=rec1["id"], region=rec1["region"], age=int(rec1["age"]), dna_seed=rec1["seed"])[:length]
    dna2 = s(id=rec2["id"], region=rec2["region"], age=int(rec2["age"]), dna_seed=rec2["seed"])[:length]

    similarity = fast_similarity_score(dna1, dna2, k)

    return {
        "sample_1": id1,
        "sample_2": id2,
        "similarity_score": similarity,
        "k_mer_size": k,
        "sequence_length_compared": length
    }


# For queries

# Gemini client with your API key
client = genai.Client(api_key="AIzaSyAXTiVxDGp6LZY3sADkAEAA600okKrr3_o")

@app.post("/chat/")
async def chat_with_gemini(message: str = Body(..., embed=False)):
    lower_msg = message.lower()

    # Keyword-based hardcoded responses
    if "dna generation" in lower_msg:
        return {
            "response": (
                "DNA sequences are generated based on ancient remains data — "
                "including region, age, and a seed value. The seed ensures reproducibility. "
                "The 'generate-dna' endpoint allows you to fetch a portion of this generated sequence."
            )
        }

    elif "dna comparison" in lower_msg:
        return {
            "response": (
                "DNA comparison uses k-mers, which are substrings of length 'k'. "
                "The similarity score is the ratio of overlapping k-mers between two sequences. "
                "This helps measure how genetically close the two samples are."
            )
        }

    # Otherwise, fallback to Gemini API for a general response
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=message,
        )
        return {"response": response.text}
    except Exception as e:
        return {"error": f"Gemini API error: {str(e)}"}