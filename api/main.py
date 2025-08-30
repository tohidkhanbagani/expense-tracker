# main.py (Complete Fixed Version)

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import shutil
import os
import json
from google.api_core import exceptions

# Import all classes
from pipeline.ocr_model import ExpenseExtractor
from pipeline.financial_insights import FinancialInsightsAnalyzer
from pipeline.nlp_chatbot import FinancialNLPChatbot
from database.supabase_client import SupabaseClient

# Initialize FastAPI app
app = FastAPI(title="Complete Expense & Financial Insights API", version="2.0")

# Initialize all components
extractor = ExpenseExtractor()
insights_analyzer = FinancialInsightsAnalyzer()
nlp_chatbot = FinancialNLPChatbot()
db_client = SupabaseClient()

# Pydantic Models
class ExpenseItem(BaseModel):
    bill_no: Optional[str]
    expence_name: str
    amount: float
    category: str
    mode: str

class ExtractionResponse(BaseModel):
    extracted_data: List[ExpenseItem]
    saved_to_database: bool
    user_id: str

class ChatQuery(BaseModel):
    query: str
    user_id: str
    conversation_id: Optional[str] = None

class ConversationHistory(BaseModel):
    conversation_history: List[Dict]
    current_query: str
    user_id: str

# ==========================
# OCR & EXTRACTION ENDPOINTS
# ==========================

@app.post("/extract/{user_id}", response_model=ExtractionResponse)
async def extract_and_save_expense(user_id: str, file: UploadFile = File(...)):
    """
    Extract expense data from uploaded file and save to Supabase.
    """
    temp_file = f"temp_{file.filename}"
    
    try:
        # Save uploaded file temporarily
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Extract data using OCR
        try:
            extracted_data = extractor.extract_expense(temp_file)
        except exceptions.InvalidArgument as e:
            raise HTTPException(status_code=400, detail=f"Invalid image provided: {e}")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="The provided image does not appear to be a bill or receipt.")
        except Exception as e:
            if "Failed to parse" in str(e):
                raise HTTPException(status_code=400, detail="The provided image does not appear to be a bill or receipt.")
            else:
                raise e
        
        # Save to Supabase
        save_result = db_client.insert_expenses(user_id, extracted_data)
        
        if not save_result.get("success", False):
            raise HTTPException(status_code=500, detail=f"Failed to save to database: {save_result.get('error', 'Unknown error')}")
        
        return {
            "extracted_data": extracted_data,
            "saved_to_database": True,
            "user_id": user_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
    finally:
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)

# Legacy endpoint (without saving to database)
@app.post("/extract", response_model=ExtractionResponse)
async def extract_expense_only(file: UploadFile = File(...)):
    """
    Extract expense data from uploaded file (without saving to database).
    """
    temp_file = f"temp_{file.filename}"
    
    try:
        # Save uploaded file temporarily
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Extract data using OCR
        try:
            extracted_data = extractor.extract_expense(temp_file)
        except exceptions.InvalidArgument as e:
            raise HTTPException(status_code=400, detail=f"Invalid image provided: {e}")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="The provided image does not appear to be a bill or receipt.")
        except Exception as e:
            if "Failed to parse" in str(e):
                raise HTTPException(status_code=400, detail="The provided image does not appear to be a bill or receipt.")
            else:
                raise e
        
        return {
            "extracted_data": extracted_data,
            "saved_to_database": False,
            "user_id": "not_specified"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
    finally:
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)

# =============================
# FINANCIAL INSIGHTS ENDPOINTS
# =============================

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

@app.get("/alerts/{user_id}")
async def get_spending_alerts(user_id: str):
    """
    Get spending anomalies and alerts for user from Supabase data.
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

# ========================
# NLP CHATBOT ENDPOINTS
# ========================

@app.post("/chat")
async def financial_chat(chat_request: ChatQuery):
    """
    Process natural language financial queries and return structured insights.
    """
    try:
        response = nlp_chatbot.process_query(
            chat_request.query, 
            chat_request.user_id
        )
        
        if "error" in response:
            raise HTTPException(status_code=400, detail=response["error"])
            
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@app.post("/chat/quick")
async def quick_financial_chat(chat_request: ChatQuery):
    """
    Get quick response for simple financial queries.
    """
    try:
        response = nlp_chatbot.generate_quick_response(
            chat_request.query,
            chat_request.user_id
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quick chat failed: {str(e)}")

@app.post("/chat/conversation")
async def multi_turn_chat(conversation_request: ConversationHistory):
    """
    Handle multi-turn conversations with context awareness.
    """
    try:
        response = nlp_chatbot.handle_multi_turn_conversation(
            conversation_request.conversation_history,
            conversation_request.current_query,
            conversation_request.user_id
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversation processing failed: {str(e)}")

@app.post("/chat/intent")
async def classify_user_intent(chat_request: ChatQuery):
    """
    Classify user intent from natural language query.
    """
    try:
        intent = nlp_chatbot.classify_intent(
            chat_request.query,
            chat_request.user_id
        )
        
        return intent
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Intent classification failed: {str(e)}")

# =======================
# DATABASE ENDPOINTS
# =======================

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

# =======================
# HEALTH & INFO ENDPOINTS
# =======================

@app.get("/")
async def root():
    """
    API Health check and information.
    """
    return {
        "message": "Complete Expense & Financial Insights API",
        "version": "2.0",
        "status": "healthy",
        "available_endpoints": {
            "extraction": ["/extract", "/extract/{user_id}"],
            "insights": ["/insights/{user_id}", "/budget/{user_id}", "/alerts/{user_id}", "/report/{user_id}"],
            "chat": ["/chat", "/chat/quick", "/chat/conversation", "/chat/intent"],
            "data": ["/expenses/{user_id}", "/profile/{user_id}"]
        }
    }

@app.get("/health")
async def health_check():
    """
    Simple health check endpoint.
    """
    return {"status": "healthy", "timestamp": "2025-08-30T22:14:00"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
