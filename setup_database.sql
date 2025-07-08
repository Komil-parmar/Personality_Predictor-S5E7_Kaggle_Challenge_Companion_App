-- SQL commands to create the required tables in Supabase
-- Run these commands in your Supabase SQL editor

-- Create pca_submissions table
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

-- Create user_feedback table
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
    confidence_score FLOAT,
    sharing_preference TEXT,
    user_name TEXT,
    additional_context TEXT,
    reliability_score FLOAT,
    distance_to_problematic FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable Row Level Security (RLS) for better security
ALTER TABLE pca_submissions ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_feedback ENABLE ROW LEVEL SECURITY;

-- Create policies to allow all operations (you can customize these as needed)
CREATE POLICY "Allow all operations on pca_submissions" 
    ON pca_submissions FOR ALL 
    USING (true);

CREATE POLICY "Allow all operations on user_feedback" 
    ON user_feedback FOR ALL 
    USING (true);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_pca_submissions_timestamp ON pca_submissions(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_pca_submissions_personality ON pca_submissions(actual_personality);
CREATE INDEX IF NOT EXISTS idx_user_feedback_timestamp ON user_feedback(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_user_feedback_personality ON user_feedback(predicted_personality);
