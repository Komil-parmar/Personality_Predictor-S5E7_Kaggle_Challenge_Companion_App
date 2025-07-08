# Supabase Setup Guide

This guide will help you set up Supabase for your Personality Predictor app to persist data across deployments.

## 🚀 Quick Setup

### 1. Create a Supabase Project
1. Go to [Supabase](https://supabase.com) and sign up/log in
2. Click "New Project"
3. Choose your organization and create a new project
4. Wait for the project to be set up (usually takes a few minutes)

### 2. Get Your API Keys
1. Go to your project dashboard
2. Click on "Settings" in the left sidebar
3. Click on "API" under Project Settings
4. Copy your:
   - Project URL
   - `anon` `public` key (this is safe to use in client-side code)

### 3. Set Up Streamlit Secrets

#### For Local Development:
Create a `.streamlit/secrets.toml` file in your project root:
```toml
SUPABASE_URL = "your_project_url_here"
SUPABASE_ANON_KEY = "your_anon_key_here"
```

#### For Streamlit Cloud Deployment:
1. Go to your Streamlit Cloud dashboard
2. Click on your app
3. Go to "Settings" > "Secrets"
4. Add the following secrets:
```toml
SUPABASE_URL = "your_project_url_here"
SUPABASE_ANON_KEY = "your_anon_key_here"
```

### 4. Create Database Tables
1. Go to your Supabase project dashboard
2. Click on "SQL Editor" in the left sidebar
3. Click "New Query"
4. Copy and paste the contents of `setup_database.sql`
5. Click "Run" to create the tables

### 5. Migrate Existing Data
1. Run your Streamlit app locally: `streamlit run setup_supabase.py`
2. Click "Migrate Existing Data" to transfer your local JSON data to Supabase
3. Verify the migration was successful

## 📊 Database Schema

### `pca_submissions` Table
Stores personality predictions and user data for the community map.

| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL | Primary key |
| timestamp | TIMESTAMP | When the submission was made |
| display_name | TEXT | User's display name |
| linkedin_profile | TEXT | LinkedIn profile URL (optional) |
| actual_personality | TEXT | User's actual personality type |
| predicted_personality | TEXT | Model's prediction |
| prediction_confidence | FLOAT | Model's confidence score |
| user_characteristics | JSONB | All user input characteristics |
| reliability_score | FLOAT | Reliability score based on distance |
| distance_to_problematic | FLOAT | Distance to problematic samples |
| feature_vector | JSONB | Numerical feature vector |
| model_confidence_scale | INTEGER | Model confidence (1-100) |
| user_confidence_scale | INTEGER | User confidence (1-100) |
| confidence_difference | INTEGER | Difference between confidences |

### `user_feedback` Table
Stores user feedback about predictions.

| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL | Primary key |
| timestamp | TIMESTAMP | When the feedback was submitted |
| display_name | TEXT | User's display name |
| linkedin_profile | TEXT | LinkedIn profile URL (optional) |
| feedback_text | TEXT | User's feedback message |
| rating | INTEGER | User's rating (if applicable) |
| user_characteristics | JSONB | User's input characteristics |
| predicted_personality | TEXT | Model's prediction |
| actual_personality | TEXT | User's actual personality |
| confidence_score | FLOAT | Model's confidence |
| sharing_preference | TEXT | User's sharing preference |
| user_name | TEXT | User's name (if provided) |
| additional_context | TEXT | Additional context from user |
| reliability_score | FLOAT | Reliability score |
| distance_to_problematic | FLOAT | Distance to problematic samples |

## 🔧 Configuration

### Environment Variables
The app looks for Supabase credentials in this order:
1. Streamlit secrets (`st.secrets`)
2. Environment variables (`os.getenv`)

### Required Environment Variables:
- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_ANON_KEY`: Your Supabase anon key

## 🛠️ Troubleshooting

### Common Issues:

1. **"Supabase credentials not found"**
   - Make sure you've set up your `.streamlit/secrets.toml` file
   - Verify the keys are exactly `SUPABASE_URL` and `SUPABASE_ANON_KEY`

2. **"Failed to connect to Supabase"**
   - Check that your URL and API key are correct
   - Ensure your internet connection is working
   - Verify your Supabase project is active

3. **"Table does not exist"**
   - Make sure you've run the SQL commands from `setup_database.sql`
   - Check that the tables were created successfully in your Supabase dashboard

4. **"Migration failed"**
   - Ensure your local JSON files exist and are valid
   - Check that the database tables are properly created
   - Verify your Supabase permissions

### Getting Help:
- Check the Supabase documentation: https://supabase.com/docs
- Review the error messages in the Streamlit app
- Ensure all dependencies are installed: `pip install -r requirements.txt`

## 📈 Benefits of Using Supabase

1. **Persistence**: Data survives app restarts and redeployments
2. **Scalability**: Can handle large amounts of data
3. **Real-time**: Built-in real-time subscriptions
4. **Security**: Built-in authentication and row-level security
5. **Analytics**: Query your data with SQL
6. **Backup**: Automatic backups and point-in-time recovery

## 🔒 Security Best Practices

1. **Never share your service role key** - only use the anon key in your app
2. **Use Row Level Security (RLS)** - already enabled in the setup
3. **Regularly rotate your API keys** if needed
4. **Monitor your database usage** in the Supabase dashboard

## 🎯 Next Steps

After setup is complete:
1. Test the app locally to ensure everything works
2. Deploy to Streamlit Cloud with the secrets configured
3. Monitor the database usage in your Supabase dashboard
4. Consider setting up automated backups for important data
