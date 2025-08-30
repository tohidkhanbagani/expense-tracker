
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import shutil
import os
import json
from google.api_core import exceptions

# Import updated classes
from pipeline.ocr_model import ExpenseExtractor
from pipeline.financial_insights import FinancialInsightsAnalyzer
from database.supabase_client import SupabaseClient

# import your ExpenseExtractor class here
from pipeline.ocr_model import ExpenseExtractor

# Initialize FastAPI app
app = FastAPI(title="Expense Extraction API", version="1.0")

# Initialize the extractor once
extractor = ExpenseExtractor()
# Initialize components

insights_analyzer = FinancialInsightsAnalyzer()
db_client = SupabaseClient()


class ExpenseItem(BaseModel):
    bill_no: Optional[str]  # <-- allow null
    expence_name: str
    amount: float
    category: str
    mode: str

class ExtractionResponse(BaseModel):
    extracted_data: List[ExpenseItem]

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
        except exceptions.InvalidArgument as e:
            raise HTTPException(status_code=400, detail=f"Invalid image provided: {e}")
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



@app.get("/insights/{user_id}")
async def get_user_insights(user_id: str, analysis_period: int = 30):
    """
    Get comprehensive financial insights for a user from Supabase data.
    """
    try:
        insights = insights_analyzer.analyze_comprehensive_insights(user_id, analysis_period)
        
        if "error" in insights:
            raise HTTPException(status_code=404, detail=insights["error"])
            
        return insights
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating insights: {str(e)}")

@app.get("/budget/{user_id}")
async def generate_user_budget(user_id: str, savings_target: float = 20.0):
    """
    Generate smart budget plan for user based on Supabase data.
    """
    try:
        budget = insights_analyzer.generate_smart_budget(user_id, savings_target)
        
        if "error" in budget:
            raise HTTPException(status_code=404, detail=budget["error"])
            
        return budget
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating budget: {str(e)}")

@app.get("/expenses/{user_id}")
async def get_user_expenses(user_id: str, days: int = 30):
    """
    Get user's expenses from Supabase.
    """
    try:
        expenses = db_client.fetch_user_expenses(user_id, days)
        summary = db_client.get_expense_summary(user_id, days)
        
        return {
            "expenses": expenses,
            "summary": summary,
            "user_id": user_id,
            "period_days": days
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching expenses: {str(e)}")

@app.get("/profile/{user_id}")
async def get_user_profile(user_id: str):
    """
    Get user's profile from Supabase.
    """
    try:
        profile = db_client.fetch_user_profile(user_id)
        
        if not profile:
            raise HTTPException(status_code=404, detail="User profile not found")
            
        return profile
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching profile: {str(e)}")

@app.put("/profile/{user_id}")
async def update_user_profile(user_id: str, profile_data: Dict[str, Any]):
    """
    Update user's profile in Supabase.
    """
    try:
        result = db_client.upsert_user_profile(user_id, profile_data)
        
        if not result.get("success", False):
            raise HTTPException(status_code=500, detail=f"Failed to update profile: {result.get('error', 'Unknown error')}")
            
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating profile: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)