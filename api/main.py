from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
import shutil
import os
import json

# import your ExpenseExtractor class here
from pipeline.ocr_model import ExpenseExtractor

# Initialize FastAPI app
app = FastAPI(title="Expense Extraction API", version="1.0")

# Initialize the extractor once
extractor = ExpenseExtractor()

class Item(BaseModel):
    bill_no: str
    expence_name: str
    amount: float
    category: str
    mode: str

class ExtractionResponse(BaseModel):
    extracted_data: List[Item]


@app.post("/extract", response_model=ExtractionResponse)
async def extract_expense(file: UploadFile = File(...)):
    """
    Upload an image (receipt, bill, etc.) and extract expense details.
    """
    temp_file = f"temp_{file.filename}"
    try:
        # Save uploaded file temporarily
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run extraction
        try:
            extracted = extractor.extract_expense(temp_file)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="The provided image does not appear to be a bill or receipt.")
        except Exception as e:
            # This can happen if the image is not a bill and the model returns a non-JSON response.
            if "Failed to parse" in str(e):
                raise HTTPException(status_code=400, detail="The provided image does not appear to be a bill or receipt.")
            else:
                raise e

        return {"extracted_data": extracted}

    except Exception as e:
        raise e
    finally:
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)