# financial_insights.py (Updated version)

from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain.schema import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import json
from datetime import datetime, timedelta
from database.supabase_client import SupabaseClient

load_dotenv()

class FinancialInsightsAnalyzer:
    """
    Advanced Financial Insights Analyzer using Supabase database.
    """
    
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        self.model_name = model_name
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Initialize Supabase client
        self.db_client = SupabaseClient()
        
        # Your existing financial_insights_prompt here...
        self.financial_insights_prompt = """
        You are an expert financial advisor AI with deep knowledge of personal finance, budgeting, and expense management. Your role is to analyze user expense data and provide precise, actionable financial insights.

        CORE RESPONSIBILITIES:
        - Analyze spending patterns with statistical precision
        - Identify financial risks and opportunities  
        - Provide personalized, evidence-based recommendations
        - Generate realistic budget suggestions
        - Assess financial health with scoring methodology

        OUTPUT REQUIREMENTS:
        - Return only valid JSON without explanations or markdown
        - Use exact key names as specified in each method
        - Ensure all numeric values are properly formatted
        - Categories must match predefined lists exactly
        - Recommendations must be specific and actionable

        ANALYSIS PRINCIPLES:
        - Base insights on actual data patterns, not assumptions
        - Consider seasonal variations and spending trends
        - Prioritize high-impact, achievable recommendations
        - Account for user's income-to-expense ratios
        - Focus on sustainable financial habits

        FINANCIAL CATEGORIES (use only these):
        ["Food", "Grocery", "Transport", "Fuel", "Travel", "Utilities", "Rent", "Health", "Pharmacy", "Education", "Entertainment", "Shopping", "Electronics", "Home", "Services", "Subscriptions", "Fees", "Taxes", "Office", "Misc"]
        """
    
    def fetch_user_expenses(self, user_id: str, days: int = 30) -> List[Dict]:
        """
        Fetch user expenses using Supabase client.
        """
        return self.db_client.fetch_user_expenses(user_id, days)
    
    def fetch_user_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Fetch user profile using Supabase client.
        """
        return self.db_client.fetch_user_profile(user_id)
    
    # Keep your existing methods (analyze_comprehensive_insights, generate_smart_budget, etc.)
    # but they now use the Supabase client methods above
    
    def analyze_comprehensive_insights(self, user_id: str, analysis_period: int = 30) -> Dict[str, Any]:
        """
        Generate comprehensive financial insights using Supabase data.
        """
        # Fetch data using Supabase client
        expenses = self.fetch_user_expenses(user_id, analysis_period)
        user_profile = self.fetch_user_profile(user_id)
        
        if not expenses:
            return {"error": "No expense data found for user", "user_id": user_id}
        
        # Your existing analysis logic...
        prompt = f"""
        {self.financial_insights_prompt}
        
        Analyze the following user's financial data and provide comprehensive insights:
        
        EXPENSE DATA ({len(expenses)} transactions over {analysis_period} days):
        {json.dumps(expenses, indent=2)}
        
        USER PROFILE:
        {json.dumps(user_profile, indent=2)}
        
        Provide analysis in this exact JSON format:
        {{
            "financial_summary": {{
                "total_spending": 0,
                "average_daily_spending": 0,
                "total_transactions": 0,
                "analysis_period_days": {analysis_period}
            }},
            "spending_breakdown": {{
                "category_analysis": {{}},
                "payment_mode_analysis": {{}}
            }},
            "financial_health_score": {{
                "overall_score": 0,
                "spending_discipline": 0,
                "budget_adherence": 0,
                "savings_rate": 0,
                "category_balance": 0
            }},
            "key_insights": [],
            "personalized_recommendations": {{
                "immediate_actions": [],
                "budget_optimizations": [],
                "savings_opportunities": []
            }},
            "spending_alerts": [],
            "projected_monthly_budget": {{}}
        }}
        """
        
        try:
            msg = HumanMessage(content=[{"type": "text", "text": prompt}])
            response = self.llm.invoke([msg])
            parser = JsonOutputParser()
            
            insights = parser.parse(response.content)
            insights["user_id"] = user_id
            insights["generated_at"] = datetime.now().isoformat()
            
            # Save insights to database
            self.db_client.save_financial_insights(user_id, insights)
            
            return insights
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}", "user_id": user_id}
    
    # Add other methods (generate_smart_budget, detect_spending_anomalies, etc.)
    # following the same pattern
