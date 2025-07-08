"""
Setup script to migrate existing JSON data to Supabase.
Run this script once after setting up your Supabase database.
"""

import streamlit as st
from supabase_config import get_supabase_manager

def main():
    st.title("🚀 Supabase Setup & Migration")
    st.markdown("This page helps you set up Supabase and migrate existing data.")
    
    # Instructions
    st.markdown("""
    ## 📋 Setup Instructions
    
    ### 1. Create a Supabase Project
    1. Go to [Supabase](https://supabase.com) and create a new project
    2. Get your project URL and anon key from the API settings
    
    ### 2. Set Up Streamlit Secrets
    Create a `.streamlit/secrets.toml` file in your project root with:
    ```toml
    SUPABASE_URL = "your_supabase_url_here"
    SUPABASE_ANON_KEY = "your_supabase_anon_key_here"
    ```
    
    ### 3. Create Database Tables
    Run the SQL commands from `setup_database.sql` in your Supabase SQL editor.
    
    ### 4. Migrate Existing Data
    Click the button below to migrate your existing JSON data to Supabase.
    """)
    
    st.markdown("---")
    
    # Check if Supabase is configured
    try:
        db_manager = get_supabase_manager()
        st.success("✅ Supabase connection successful!")
        
        # Migration section
        st.markdown("## 🔄 Data Migration")
        
        if st.button("🚀 Migrate Existing Data", type="primary"):
            st.markdown("### Migration Progress")
            
            with st.spinner("Migrating data..."):
                db_manager.migrate_local_data()
            
            st.success("✅ Migration completed!")
            st.markdown("You can now use the main app with Supabase backend.")
            
        # Show current data counts
        st.markdown("## 📊 Current Data Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pca_data = db_manager.get_pca_submissions()
            st.metric("PCA Submissions", len(pca_data))
            
        with col2:
            feedback_data = db_manager.get_user_feedback()
            st.metric("User Feedback", len(feedback_data))
            
        # Show recent submissions
        if len(pca_data) > 0:
            st.markdown("### Recent PCA Submissions")
            st.dataframe(pca_data[['display_name', 'actual_personality', 'predicted_personality', 'timestamp']].head(10))
            
        if len(feedback_data) > 0:
            st.markdown("### Recent Feedback")
            st.dataframe(feedback_data[['display_name', 'feedback_text', 'timestamp']].head(10))
            
    except Exception as e:
        st.error(f"❌ Supabase setup error: {str(e)}")
        st.markdown("""
        ### Troubleshooting:
        1. Make sure you've created the `.streamlit/secrets.toml` file
        2. Verify your Supabase URL and API key are correct
        3. Ensure the database tables have been created
        4. Check your internet connection
        """)

if __name__ == "__main__":
    main()
