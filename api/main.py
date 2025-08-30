from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn 
import shutil
import os
import json
from google.api_core import exceptions
from pipeline.financial_insights import FinancialInsightsAnalyzer

# import your ExpenseExtractor class here
from pipeline.ocr_model import ExpenseExtractor

# Initialize FastAPI app
app = FastAPI(title="Expense Extraction API", version="1.0")

# Initialize the extractor once
extractor = ExpenseExtractor()

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



# Initialize the financial insights analyzer
insights_analyzer = FinancialInsightsAnalyzer()
@app.get("/insights/{user_id}")
async def get_user_insights(user_id: str, analysis_period: int = 30):
    """
    Get comprehensive financial insights for a user.
    Data is automatically fetched from database.
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
    Generate smart budget plan for user based on historical data.
    """
    try:
        budget = insights_analyzer.generate_smart_budget(user_id, savings_target)
        
        if "error" in budget:
            raise HTTPException(status_code=404, detail=budget["error"])
            
        return budget
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating budget: {str(e)}")

@app.get("/alerts/{user_id}")
async def get_spending_alerts(user_id: str):
    """
    Get spending anomalies and alerts for user.
    """
    try:
        alerts = insights_analyzer.detect_spending_anomalies(user_id)
        
        if "error" in alerts:
            raise HTTPException(status_code=404, detail=alerts["error"])
            
        return alerts
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting anomalies: {str(e)}")

@app.get("/report/{user_id}")
async def get_financial_report(user_id: str, report_type: str = "monthly"):
    """
    Generate comprehensive financial report for user.
    """
    if report_type not in ["weekly", "monthly", "quarterly"]:
        raise HTTPException(status_code=400, detail="Invalid report type")
        
    try:
        report = insights_analyzer.generate_financial_report(user_id, report_type)
        return report
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")





if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
