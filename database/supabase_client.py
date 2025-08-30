# database/supabase_client.py

import os
from supabase import create_client, Client
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json

load_dotenv()

class SupabaseClient:
    """
    Supabase database client for expense tracking application.
    """
    
    def __init__(self):
        # Get Supabase credentials from environment variables
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_ANON_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set in environment variables")
        
        # Create Supabase client
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
    
    # EXPENSE OPERATIONS
    def insert_expenses(self, user_id: str, expenses: List[Dict]) -> Dict[str, Any]:
        """
        Insert extracted expenses into Supabase.
        
        Args:
            user_id: User identifier
            expenses: List of expense dictionaries from OCR extraction
        
        Returns:
            Result of insert operation
        """
        try:
            # Prepare data for insertion
            expense_records = []
            for expense in expenses:
                record = {
                    "user_id": user_id,
                    "bill_no": expense.get("bill_no"),
                    "expence_name": expense.get("expence_name"),
                    "amount": float(expense.get("amount", 0)),
                    "category": expense.get("category"),
                    "mode": expense.get("mode"),
                    "created_date": datetime.now().isoformat()
                }
                expense_records.append(record)
            
            # Insert into Supabase
            result = self.supabase.table("expenses").insert(expense_records).execute()
            
            return {
                "success": True,
                "inserted_count": len(expense_records),
                "data": result.data
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def fetch_user_expenses(self, user_id: str, days: int = 30) -> List[Dict]:
        """
        Fetch user's expenses from Supabase for specified period.
        
        Args:
            user_id: User identifier
            days: Number of days to fetch data for
        
        Returns:
            List of expense records
        """
        try:
            # Calculate date threshold
            date_threshold = (datetime.now() - timedelta(days=days)).isoformat()
            
            # Query expenses from Supabase
            result = self.supabase.table("expenses").select("*").eq(
                "user_id", user_id
            ).gte(
                "created_date", date_threshold
            ).order("created_date", desc=True).execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            print(f"Error fetching expenses: {e}")
            return []
    
    def get_expense_summary(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Get expense summary for user.
        """
        try:
            expenses = self.fetch_user_expenses(user_id, days)
            
            if not expenses:
                return {"total": 0, "count": 0, "categories": {}}
            
            total = sum(expense["amount"] for expense in expenses)
            categories = {}
            
            for expense in expenses:
                category = expense["category"]
                if category not in categories:
                    categories[category] = {"amount": 0, "count": 0}
                categories[category]["amount"] += expense["amount"]
                categories[category]["count"] += 1
            
            return {
                "total": total,
                "count": len(expenses),
                "categories": categories,
                "period_days": days
            }
            
        except Exception as e:
            print(f"Error getting expense summary: {e}")
            return {"error": str(e)}
    
    # USER PROFILE OPERATIONS
    def fetch_user_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Fetch user's profile from Supabase.
        
        Args:
            user_id: User identifier
        
        Returns:
            User profile dictionary
        """
        try:
            result = self.supabase.table("user_profiles").select("*").eq(
                "user_id", user_id
            ).execute()
            
            if result.data and len(result.data) > 0:
                return result.data[0]
            else:
                return {}
                
        except Exception as e:
            print(f"Error fetching user profile: {e}")
            return {}
    
    def upsert_user_profile(self, user_id: str, profile_data: Dict) -> Dict[str, Any]:
        """
        Insert or update user profile.
        
        Args:
            user_id: User identifier
            profile_data: Profile data to upsert
        
        Returns:
            Result of upsert operation
        """
        try:
            profile_record = {
                "user_id": user_id,
                "updated_date": datetime.now().isoformat(),
                **profile_data
            }
            
            # Check if profile exists
            existing = self.fetch_user_profile(user_id)
            
            if existing:
                # Update existing profile
                result = self.supabase.table("user_profiles").update(
                    profile_record
                ).eq("user_id", user_id).execute()
            else:
                # Insert new profile
                profile_record["created_date"] = datetime.now().isoformat()
                result = self.supabase.table("user_profiles").insert(
                    profile_record
                ).execute()
            
            return {
                "success": True,
                "data": result.data
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    # INSIGHTS OPERATIONS
    def save_financial_insights(self, user_id: str, insights_data: Dict) -> Dict[str, Any]:
        """
        Save financial insights to database.
        
        Args:
            user_id: User identifier
            insights_data: Financial insights data
        
        Returns:
            Result of save operation
        """
        try:
            insights_record = {
                "user_id": user_id,
                "insights_data": json.dumps(insights_data),
                "generated_date": datetime.now().isoformat(),
                "insights_type": insights_data.get("response_metadata", {}).get("intent", "general")
            }
            
            result = self.supabase.table("financial_insights").insert(
                insights_record
            ).execute()
            
            return {
                "success": True,
                "data": result.data
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_latest_insights(self, user_id: str, insights_type: str = None) -> Dict[str, Any]:
        """
        Get latest financial insights for user.
        
        Args:
            user_id: User identifier
            insights_type: Type of insights to fetch
        
        Returns:
            Latest insights data
        """
        try:
            query = self.supabase.table("financial_insights").select("*").eq(
                "user_id", user_id
            )
            
            if insights_type:
                query = query.eq("insights_type", insights_type)
            
            result = query.order("generated_date", desc=True).limit(1).execute()
            
            if result.data and len(result.data) > 0:
                insights = result.data[0]
                insights["insights_data"] = json.loads(insights["insights_data"])
                return insights
            else:
                return {}
                
        except Exception as e:
            print(f"Error fetching insights: {e}")
            return {}
