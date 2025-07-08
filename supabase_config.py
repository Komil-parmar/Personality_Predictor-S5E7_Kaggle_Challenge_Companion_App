"""
Supabase configuration and database operations for the Personality Predictor app.
This module handles all database operations including storing PCA submissions and user feedback.
"""

import os
import json
import pandas as pd
from datetime import datetime
from supabase import create_client, Client
import streamlit as st


class SupabaseManager:
    """Handles all Supabase database operations"""
    
    def __init__(self):
        """Initialize Supabase client"""
        self.supabase_url = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL"))
        self.supabase_key = st.secrets.get("SUPABASE_ANON_KEY", os.getenv("SUPABASE_ANON_KEY"))
        
        if not self.supabase_url or not self.supabase_key:
            st.warning("⚠️ Supabase credentials not found. Using local JSON files as fallback.")
            self.use_fallback = True
            self.supabase = None
            return
        
        try:
            self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
            self.use_fallback = False
            # Test connection
            self.test_connection()
        except Exception as e:
            st.warning(f"⚠️ Failed to connect to Supabase: {str(e)}. Using local JSON files as fallback.")
            self.use_fallback = True
            self.supabase = None
    
    def test_connection(self):
        """Test the Supabase connection"""
        try:
            # Simple test query
            response = self.supabase.table("pca_submissions").select("count", count="exact").execute()
            st.success("✅ Connected to Supabase successfully!")
        except Exception as e:
            st.warning(f"⚠️ Supabase connection test failed: {str(e)}")
    
    def create_tables(self):
        """Create necessary tables if they don't exist"""
        try:
            # Create pca_submissions table
            pca_table_sql = """
            CREATE TABLE IF NOT EXISTS pca_submissions (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                display_name TEXT NOT NULL,
                linkedin_profile TEXT,
                actual_personality TEXT NOT NULL,
                predicted_personality TEXT NOT NULL,
                prediction_confidence FLOAT NOT NULL,
                user_characteristics JSONB NOT NULL,
                reliability_score FLOAT,
                distance_to_problematic FLOAT,
                feature_vector JSONB,
                model_confidence_scale INTEGER,
                user_confidence_scale INTEGER,
                confidence_difference INTEGER,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            """
            
            # Create user_feedback table
            feedback_table_sql = """
            CREATE TABLE IF NOT EXISTS user_feedback (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                display_name TEXT NOT NULL,
                linkedin_profile TEXT,
                feedback_text TEXT NOT NULL,
                rating INTEGER,
                user_characteristics JSONB,
                predicted_personality TEXT,
                actual_personality TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            """
            
            # Note: These would need to be run manually in Supabase SQL editor
            # or through the Supabase dashboard as the Python client doesn't support DDL
            st.info("📋 Please create the following tables in your Supabase dashboard:")
            st.code(pca_table_sql, language="sql")
            st.code(feedback_table_sql, language="sql")
            
        except Exception as e:
            st.error(f"❌ Error creating tables: {str(e)}")
    
    def save_pca_submission(self, submission_data):
        """Save a PCA submission to the database or local file"""
        if self.use_fallback:
            return self._save_pca_submission_local(submission_data)
        
        try:
            # Convert timestamp to ISO format if it's a number
            if isinstance(submission_data.get('timestamp'), (int, float)):
                submission_data['timestamp'] = datetime.fromtimestamp(submission_data['timestamp'] / 1000).isoformat()
            elif isinstance(submission_data.get('timestamp'), str):
                # If it's already a string, parse and reformat to ensure consistency
                try:
                    dt = datetime.fromisoformat(submission_data['timestamp'].replace('Z', '+00:00'))
                    submission_data['timestamp'] = dt.isoformat()
                except:
                    submission_data['timestamp'] = datetime.now().isoformat()
            else:
                submission_data['timestamp'] = datetime.now().isoformat()
            
            # Insert into database
            response = self.supabase.table("pca_submissions").insert(submission_data).execute()
            
            if response.data:
                st.success("✅ PCA submission saved to database!")
                return True
            else:
                st.error("❌ Failed to save PCA submission")
                return False
                
        except Exception as e:
            st.error(f"❌ Error saving PCA submission: {str(e)}")
            return False
    
    def _save_pca_submission_local(self, submission_data):
        """Save PCA submission to local JSON file (fallback)"""
        try:
            # Try to load existing PCA submissions
            try:
                existing_pca_data = pd.read_json('personality_pca_submissions.json', lines=True)
                pca_df = pd.concat([existing_pca_data, pd.DataFrame([submission_data])], ignore_index=True)
            except FileNotFoundError:
                pca_df = pd.DataFrame([submission_data])
            
            # Save to JSON file
            pca_df.to_json('personality_pca_submissions.json', orient='records', lines=True)
            st.success("✅ PCA submission saved to local file!")
            return True
            
        except Exception as e:
            st.error(f"❌ Error saving PCA submission to local file: {str(e)}")
            return False
    
    def get_pca_submissions(self, limit=None):
        """Retrieve PCA submissions from the database or local file"""
        if self.use_fallback:
            return self._get_pca_submissions_local()
        
        try:
            query = self.supabase.table("pca_submissions").select("*").order("timestamp", desc=True)
            
            if limit:
                query = query.limit(limit)
            
            response = query.execute()
            
            if response.data:
                return pd.DataFrame(response.data)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            st.error(f"❌ Error retrieving PCA submissions: {str(e)}")
            return pd.DataFrame()
    
    def _get_pca_submissions_local(self):
        """Get PCA submissions from local JSON file (fallback)"""
        try:
            return pd.read_json('personality_pca_submissions.json', lines=True)
        except FileNotFoundError:
            return pd.DataFrame()
        except Exception as e:
            st.error(f"❌ Error reading local PCA submissions: {str(e)}")
            return pd.DataFrame()
    
    def save_user_feedback(self, feedback_data):
        """Save user feedback to the database or local file"""
        if self.use_fallback:
            return self._save_user_feedback_local(feedback_data)
        
        try:
            # Ensure timestamp is in correct format
            if 'timestamp' not in feedback_data:
                feedback_data['timestamp'] = datetime.now().isoformat()
            
            # Insert into database
            response = self.supabase.table("user_feedback").insert(feedback_data).execute()
            
            if response.data:
                st.success("✅ User feedback saved to database!")
                return True
            else:
                st.error("❌ Failed to save user feedback")
                return False
                
        except Exception as e:
            st.error(f"❌ Error saving user feedback: {str(e)}")
            return False
    
    def _save_user_feedback_local(self, feedback_data):
        """Save user feedback to local JSON file (fallback)"""
        try:
            # Try to load existing feedback file
            try:
                existing_feedback = pd.read_json('user_feedback.json', lines=True)
                feedback_df = pd.concat([existing_feedback, pd.DataFrame([feedback_data])], ignore_index=True)
            except FileNotFoundError:
                feedback_df = pd.DataFrame([feedback_data])
            
            # Save to JSON file
            feedback_df.to_json('user_feedback.json', orient='records', lines=True)
            st.success("✅ User feedback saved to local file!")
            return True
            
        except Exception as e:
            st.error(f"❌ Error saving user feedback to local file: {str(e)}")
            return False
    
    def get_user_feedback(self, limit=None):
        """Retrieve user feedback from the database or local file"""
        if self.use_fallback:
            return self._get_user_feedback_local()
        
        try:
            query = self.supabase.table("user_feedback").select("*").order("timestamp", desc=True)
            
            if limit:
                query = query.limit(limit)
            
            response = query.execute()
            
            if response.data:
                return pd.DataFrame(response.data)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            st.error(f"❌ Error retrieving user feedback: {str(e)}")
            return pd.DataFrame()
    
    def _get_user_feedback_local(self):
        """Get user feedback from local JSON file (fallback)"""
        try:
            return pd.read_json('user_feedback.json', lines=True)
        except FileNotFoundError:
            return pd.DataFrame()
        except Exception as e:
            st.error(f"❌ Error reading local user feedback: {str(e)}")
            return pd.DataFrame()

    def migrate_local_data(self):
        """Migrate existing local JSON data to Supabase"""
        if self.use_fallback:
            st.info("📝 Supabase not configured. Cannot migrate data to database.")
            return
        
        try:
            # Migrate PCA submissions
            try:
                pca_data = pd.read_json('personality_pca_submissions.json', lines=True)
                st.info(f"📦 Found {len(pca_data)} PCA submissions to migrate...")
                
                for _, row in pca_data.iterrows():
                    submission_data = row.to_dict()
                    self.save_pca_submission(submission_data)
                
                st.success(f"✅ Migrated {len(pca_data)} PCA submissions to Supabase!")
                
            except FileNotFoundError:
                st.info("📝 No local PCA submissions found to migrate.")
            except Exception as e:
                st.warning(f"⚠️ Error migrating PCA submissions: {str(e)}")
            
            # Migrate user feedback
            try:
                with open('user_feedback.json', 'r') as f:
                    feedback_data = json.load(f)
                
                if feedback_data:
                    st.info(f"📦 Found {len(feedback_data)} feedback entries to migrate...")
                    
                    for feedback in feedback_data:
                        self.save_user_feedback(feedback)
                    
                    st.success(f"✅ Migrated {len(feedback_data)} feedback entries to Supabase!")
                else:
                    st.info("📝 No local feedback data found to migrate.")
                    
            except FileNotFoundError:
                st.info("📝 No local user feedback found to migrate.")
            except Exception as e:
                st.warning(f"⚠️ Error migrating user feedback: {str(e)}")
                
        except Exception as e:
            st.error(f"❌ Error during data migration: {str(e)}")


# Global instance
@st.cache_resource
def get_supabase_manager():
    """Get cached Supabase manager instance"""
    return SupabaseManager()
