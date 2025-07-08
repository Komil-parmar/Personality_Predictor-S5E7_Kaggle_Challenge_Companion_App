import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Import Supabase configuration
from supabase_config import get_supabase_manager

# Page configuration
st.set_page_config(
    page_title=" Personality Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-description {
        font-size: 0.8rem;
        color: #666;
        font-style: italic;
        margin-top: -10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model_components():
    """Load the trained model and its components"""
    try:
        with open('xgboost_personality_model.pkl', 'rb') as f:
            components = pickle.load(f)
        return components
    except FileNotFoundError:
        st.error("❌ Model file not found! Please run the training notebook first.")
        st.stop()

@st.cache_data
def load_misclassified_data():
    """Load misclassified samples from CSV file"""
    try:
        misclassified_features = pd.read_csv('misclassified_features.csv')
        return misclassified_features.values
    except FileNotFoundError:
        st.warning("⚠️ Misclassified samples file not found. Distance analysis will be skipped.")
        return None

@st.cache_data
def load_model_statistics():
    """Load model statistics from JSON file"""
    import json
    try:
        with open('model_statistics.json', 'r') as f:
            stats = json.load(f)
        return stats
    except FileNotFoundError:
        st.warning("⚠️ Model statistics file not found. Statistics will be skipped.")
        return None

@st.cache_data
def load_enhanced_misclassified_data():
    """Load enhanced misclassified samples with metadata"""
    try:
        enhanced_misclassified = pd.read_csv('enhanced_misclassified_samples.csv')
        return enhanced_misclassified
    except FileNotFoundError:
        st.warning("⚠️ Enhanced misclassified samples file not found. Detailed comparison will be skipped.")
        return None

def calculate_distance_to_misclassified(user_input_array, misclassified_features):
    """Calculate the distance from user input to nearest misclassified sample"""
    if misclassified_features is None or len(misclassified_features) == 0:
        return None, None
    
    # Convert user input to array
    user_array = np.array(user_input_array).reshape(1, -1)
    
    # Calculate distances to all misclassified samples
    distances = euclidean_distances(user_array, misclassified_features)
    
    # Find minimum distance and its index
    min_distance = np.min(distances)
    min_index = np.argmin(distances)
    
    return min_distance, min_index

def decode_categorical_value(feature, encoded_value, label_encoders, categorical_columns):
    """Decode an encoded categorical value back to its original string"""
    if feature in categorical_columns and feature in label_encoders:
        try:
            # Convert to int if it's a float representation of an integer
            if isinstance(encoded_value, float) and encoded_value.is_integer():
                encoded_value = int(encoded_value)
            
            # Decode the value
            decoded = label_encoders[feature].inverse_transform([encoded_value])[0]
            return decoded
        except (ValueError, IndexError):
            # If decoding fails, return the original value
            return str(encoded_value)
    else:
        return encoded_value

def get_closest_misclassified_sample_info(user_input_dict, enhanced_misclassified_df, feature_columns, min_index):
    """Get detailed information about the closest misclassified sample"""
    if enhanced_misclassified_df is None or min_index is None or min_index >= len(enhanced_misclassified_df):
        return None
    
    closest_sample = enhanced_misclassified_df.iloc[min_index]
    
    # Calculate feature differences
    feature_differences = {}
    for feature in feature_columns:
        user_val = user_input_dict.get(feature, 0)
        sample_val = closest_sample[feature]
        
        # Handle different data types properly
        try:
            # Convert both values to float for numerical comparison
            user_val_float = float(user_val)
            sample_val_float = float(sample_val)
            difference = abs(user_val_float - sample_val_float)
        except (ValueError, TypeError):
            # For categorical or non-numeric data, use exact match comparison
            if str(user_val) == str(sample_val):
                difference = 0.0  # Exact match
            else:
                difference = 1.0  # Different categories
        
        feature_differences[feature] = {
            'user_value': user_val,
            'sample_value': sample_val,
            'difference': difference,
            'user_value_raw': user_val,  # Keep original values for display
            'sample_value_raw': sample_val
        }
    
    # Sort features by difference (largest differences first)
    sorted_differences = dict(sorted(feature_differences.items(), 
                                   key=lambda x: x[1]['difference'], reverse=True))
    
    sample_info = {
        'sample_id': closest_sample['sample_id'],
        'true_label': closest_sample['true_label'],
        'predicted_label': closest_sample['predicted_label'],
        'confidence': closest_sample['confidence'],
        'description': closest_sample['description'],
        'feature_differences': sorted_differences
    }
    
    return sample_info
    
    return min_distance, min_index

@st.cache_data
def load_baseline_distance_metrics():
    """Load baseline distance metrics from JSON file"""
    import json
    try:
        with open('baseline_distance_metrics.json', 'r') as f:
            metrics = json.load(f)
        return metrics
    except FileNotFoundError:
        st.warning("⚠️ Baseline distance metrics file not found. Using fallback value.")
        # Fallback to old hardcoded value if file not found
        return {
            'overall': {'weighted_baseline_distance': 9.341171},
            'class_specific': {
                'Introvert': {'weighted_baseline_distance': 9.341171},
                'Extrovert': {'weighted_baseline_distance': 9.341171}
            }
        }

def create_feature_descriptions():
    """Create feature descriptions for better user understanding"""
    return {
        # Updated descriptions based on the actual dataset variables
        'Time_spent_Alone': 'Average time (hours or scale) a person spends alone everyday.\nBut why 11 hours and not 24? Because on average people only get 11 out of 24 hours to actually choose whether they want to be alone or not.',
        'Stage_fear': 'Do you feel nervous or scared speaking or performing in front of a group? (Yes/No).\nEasier question is: Do you feel significantly higher heartbeats when you are about to speak in front of a group? If yes (most of the times), then you have stage fear.',
        'Social_event_attendance': 'Frequency of attending social events (0–10 scale).\n0 = never attend any avoidable social events and\n10 = always attend whenever possible.',
        'Going_outside': 'How many days in a week do you go outside? (0–7 scale). Not for work, but just for some ride, shopping, events, socializing etc.',
        'Drained_after_socializing': 'Do you start overthinking about your conversations after socializing? (Yes/No). Indicates feeling drained after social interactions.',
        'Friends_circle_size': 'Number of close friends you have i.e. the ones you can call almost anytime and talk without having major news to share.',
        'Post_frequency': 'How often do you post on social media? (0–10 scale).\n0 = never post and\n10 = always share whenever you get some content like sharing your P.O.V. on something, some news, small achievements, your recent experiences etc.',
    }

def get_feature_description(feature_name, descriptions):
    """Get description for a feature, with fallback for unknown features"""
    if feature_name in descriptions:
        return descriptions[feature_name]
    else:
        # Create a generic description
        clean_name = feature_name.replace('_', ' ').title()
        if 'score' in feature_name.lower():
            return f'{clean_name} - A numerical measurement or rating'
        elif 'time' in feature_name.lower():
            return f'{clean_name} - Time-related measurement'
        elif 'count' in feature_name.lower() or 'number' in feature_name.lower():
            return f'{clean_name} - Count or frequency measurement'
        elif 'frequency' in feature_name.lower():
            return f'{clean_name} - Frequency or rate measurement'
        elif 'size' in feature_name.lower():
            return f'{clean_name} - Size or quantity measurement'
        elif 'fear' in feature_name.lower():
            return f'{clean_name} - Fear or anxiety related characteristic'
        elif 'social' in feature_name.lower():
            return f'{clean_name} - Social behavior or preference'
        else:
            return f'{clean_name} - Personal characteristic or preference'

def create_gauge_chart(value, title, max_val=1.0):
    """Create a gauge chart for confidence score"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        gauge = {
            'axis': {'range': [None, max_val]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.5], 'color': "lightgray"},
                {'range': [0.5, 0.8], 'color': "yellow"},
                {'range': [0.8, max_val], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.9
            }
        }
    ))
    
    fig.update_layout(height=300, margin={'l': 20, 'r': 20, 't': 40, 'b': 20})
    return fig

def create_probability_chart(proba_scores, class_names):
    """Create a horizontal bar chart for class probabilities"""
    fig = go.Figure()
    
    colors = ['#ff7f0e', '#1f77b4']  # Orange for Introvert, Blue for Extrovert
    
    for i, (prob, name) in enumerate(zip(proba_scores, class_names)):
        fig.add_trace(go.Bar(
            y=[name],
            x=[prob],
            orientation='h',
            marker_color=colors[i],
            text=f'{prob:.1%}',
            textposition='inside'
        ))
    
    fig.update_layout(
        title="Prediction Probabilities",
        xaxis_title="Probability",
        yaxis_title="Personality Type",
        height=200,
        showlegend=False,
        margin={'l': 20, 'r': 20, 't': 40, 'b': 20},
        xaxis=dict(range=[0, 1], tickformat='.0%')
    )
    
    return fig

def main():
    # App title and description
    st.markdown('<h1 class="main-header">🧠 AI Personality Predictor - By Komil K. Parmar</h1>', unsafe_allow_html=True)
    
    # Add developer name with clickable link
    col_title1, col_title2, col_title3 = st.columns([1, 2, 1])
    with col_title2:
        st.markdown("""
        <div style="text-align: center; margin-top: -1rem; margin-bottom: 1rem;">
            <a href="#developer-section" style="
                color: #666;
                text-decoration: none;
                font-size: 1rem;
                padding: 0.5rem 1rem;
                border: 1px solid #ddd;
                border-radius: 20px;
                display: inline-block;
                transition: all 0.3s ease;
                background-color: #f8f9fa;
            " onmouseover="this.style.backgroundColor='#e9ecef'; this.style.borderColor='#adb5bd'; this.style.color='#495057'" 
               onmouseout="this.style.backgroundColor='#f8f9fa'; this.style.borderColor='#ddd'; this.style.color='#666'">
                👨‍💻 By Komil K. Parmar
            </a>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Discover your personality type using advanced Machine Learning! 
            Adjust the sliders below to input your characteristics and get an instant prediction.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add prominent "Know More" button at the top
    st.markdown("""
    <div style="text-align: center; margin: 1.5rem 0;">
        <style>
        .big-button {
            background: linear-gradient(90deg, #1f77b4, #ff7f0e);
            color: white;
            padding: 0.75rem 2rem;
            border-radius: 25px;
            font-size: 1.1rem;
            font-weight: bold;
            border: none;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
        }
        .big-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.3);
        }
        </style>
    </div>
    """, unsafe_allow_html=True)
    
    col_center1, col_center2, col_center3 = st.columns([1, 2, 1])
    with col_center2:
        if st.button("🔍 ✨ Learn More About This App & Competition ✨", 
                     type="secondary", 
                     use_container_width=True,
                     key="learn_more_btn"):
            # Scroll to the know more section smoothly
            st.markdown("""
            <script>
                document.getElementById('know-more-section').scrollIntoView({
                    behavior: 'smooth'
                });
            </script>
            """, unsafe_allow_html=True)
    
    # Load model components
    components = load_model_components()
    model = components['model']
    target_encoder = components['target_encoder']
    label_encoders = components['label_encoders']
    feature_columns = components['feature_columns']
    categorical_columns = components['categorical_columns']
    numerical_columns = components['numerical_columns']
    feature_stats = components['feature_stats']
    model_accuracy = components['accuracy']
    
    # Load misclassified data for distance analysis
    misclassified_features = load_misclassified_data()
    
    # Load model statistics
    model_stats = load_model_statistics()
    
    # Load enhanced misclassified data for detailed comparison
    enhanced_misclassified_df = load_enhanced_misclassified_data()
    
    # Load baseline distance metrics
    baseline_metrics = load_baseline_distance_metrics()
    
    # Feature descriptions
    descriptions = create_feature_descriptions()
    # --- Modal dialog state ---
    if 'show_modal' not in st.session_state:
        st.session_state['show_modal'] = False
    if 'modal_inputs' not in st.session_state:
        st.session_state['modal_inputs'] = None

    # --- Input UI ---
    st.markdown('<div style="text-align: center;"><h2>📊 Input Your Characteristics</h2></div>', unsafe_allow_html=True)
    all_inputs = {}
    if len(numerical_columns) > 0:
        st.markdown('<div style="text-align: center;"><h3>📈 Numerical Characteristics</h3></div>', unsafe_allow_html=True)
        for feature in numerical_columns:
            if feature in feature_stats:
                stats = feature_stats[feature]
                description = get_feature_description(feature, descriptions)
                st.markdown(f"**{feature.replace('_', ' ').title()}**")
                st.markdown(f'<p class="feature-description">{description}</p>', unsafe_allow_html=True)
                all_inputs[feature] = st.slider(
                    f"Select {feature.replace('_', ' ').lower()}",
                    min_value=float(stats['min']),
                    max_value=float(stats['max']),
                    value=float(stats['median']),
                    step=0.1,
                    key=feature
                )
                st.caption(f"Range: {stats['min']:.1f} - {stats['max']:.1f} | Average: {stats['mean']:.1f}")
                st.divider()
    if len(categorical_columns) > 0:
        st.markdown('<div style="text-align: center;"><h3>📋 Categorical Characteristics</h3></div>', unsafe_allow_html=True)
        for feature in categorical_columns:
            if feature in label_encoders:
                description = get_feature_description(feature, descriptions)
                st.markdown(f"**{feature.replace('_', ' ').title()}**")
                st.markdown(f'<p class="feature-description">{description}</p>', unsafe_allow_html=True)
                options = label_encoders[feature].classes_
                all_inputs[feature] = st.selectbox(
                    f"Select {feature.replace('_', ' ').lower()}",
                    options=options,
                    key=feature
                )
                st.divider()
    
    # Center-aligned predict section
    st.markdown('<div style="text-align: center;"><h3>🎯 Ready to Predict?</h3></div>', unsafe_allow_html=True)
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_button = st.button("🔮 Predict My Personality", type="primary", use_container_width=True)
    if predict_button:
        st.session_state['show_modal'] = True
        st.session_state['modal_inputs'] = all_inputs.copy()

    # --- Modal Dialog Implementation ---
    if st.session_state.get('show_modal', False) and st.session_state.get('modal_inputs'):
        # Modal dialog using Streamlit container with custom styling
        st.markdown("""
        <style>
        .modal-container {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.5);
            z-index: 1000;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .modal-content {
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            max-width: 95vw;
            max-height: 90vh;
            width: 900px;
            overflow-y: auto;
            padding: 2rem;
            margin: 2rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Create a container for modal content
        modal_container = st.container()
        
        with modal_container:
            # Close button at the top
            col_close1, col_close2, col_close3 = st.columns([1, 8, 1])
            with col_close1:
                pass  # Empty column for spacing
            with col_close2:
                st.markdown('<div style="text-align: center;"><h3>🎯 Prediction Results</h3></div>', unsafe_allow_html=True)
            with col_close3:
                if st.button("✖️ Close", key="close-modal-btn", help="Close dialog"):
                    st.session_state['show_modal'] = False
                    st.session_state['modal_inputs'] = None
                    st.rerun()

        # Add separation line
        st.markdown("---")
        
        # Navigation buttons for quick access to sections
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
            <h4 style="text-align: center; margin: 0; color: #1f77b4;">🧭 Quick Navigation to Results Sections</h4>
            <p style="text-align: center; margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #666;">
                Click any link below to jump directly to that section
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation using direct anchor links
        st.markdown("### 🧭 Quick Navigation")
        
        # Create navigation links using HTML anchors
        nav_links_html = """
        <div style="display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; margin: 1rem 0;">
            <a href="#confidence-score" style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 0.5rem 1rem;
                border-radius: 8px;
                text-decoration: none;
                font-weight: 500;
                transition: all 0.3s ease;
                display: inline-block;
                margin: 0.25rem;
                text-align: center;
                min-width: 150px;
            " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 4px 12px rgba(0,0,0,0.2)'" 
               onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='none'">
                📊 Confidence Score
            </a>
            <a href="#insights-about" style="
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                color: white;
                padding: 0.5rem 1rem;
                border-radius: 8px;
                text-decoration: none;
                font-weight: 500;
                transition: all 0.3s ease;
                display: inline-block;
                margin: 0.25rem;
                text-align: center;
                min-width: 150px;
            " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 4px 12px rgba(0,0,0,0.2)'" 
               onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='none'">
                💡 Insights & About
            </a>
            <a href="#reliability-analysis" style="
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                color: white;
                padding: 0.5rem 1rem;
                border-radius: 8px;
                text-decoration: none;
                font-weight: 500;
                transition: all 0.3s ease;
                display: inline-block;
                margin: 0.25rem;
                text-align: center;
                min-width: 150px;
            " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 4px 12px rgba(0,0,0,0.2)'" 
               onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='none'">
                🎯 Reliability Analysis
            </a>
            <a href="#community-map" style="
                background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
                color: white;
                padding: 0.5rem 1rem;
                border-radius: 8px;
                text-decoration: none;
                font-weight: 500;
                transition: all 0.3s ease;
                display: inline-block;
                margin: 0.25rem;
                text-align: center;
                min-width: 150px;
            " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 4px 12px rgba(0,0,0,0.2)'" 
               onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='none'">
                🗺️ Community Map
            </a>
            <a href="#feedback-section" style="
                background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
                color: white;
                padding: 0.5rem 1rem;
                border-radius: 8px;
                text-decoration: none;
                font-weight: 500;
                transition: all 0.3s ease;
                display: inline-block;
                margin: 0.25rem;
                text-align: center;
                min-width: 150px;
            " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 4px 12px rgba(0,0,0,0.2)'" 
               onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='none'">
                🤔 Feedback
            </a>
        </div>
        """
        
        st.markdown(nav_links_html, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Render prediction results in modal
        all_inputs = st.session_state['modal_inputs']
        input_data = pd.DataFrame([all_inputs])
        for feature in categorical_columns:
            if feature in input_data.columns and feature in label_encoders:
                le = label_encoders[feature]
                input_data[feature] = le.transform([all_inputs[feature]])
        input_data = input_data.reindex(columns=feature_columns, fill_value=0)
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        predicted_personality = target_encoder.inverse_transform([prediction])[0]
        confidence = max(prediction_proba)

        # Center-aligned prediction result
        col_pred1, col_pred2, col_pred3 = st.columns([1, 2, 1])
        with col_pred2:
            if predicted_personality == 'Extrovert':
                st.success(f"🎉 **You are predicted to be an {predicted_personality}!**")
            else:
                st.info(f"🤔 **You are predicted to be an {predicted_personality}!**")
        
        # Add HTML anchor
        st.markdown('<div id="confidence-score"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 0.75rem 1.5rem;
            border-radius: 10px;
            margin: 1rem 0 0.5rem 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        ">
            <h3 style="
                color: white;
                font-size: 1.5rem;
                font-weight: 600;
                margin: 0;
                text-shadow: 0 1px 2px rgba(0,0,0,0.2);
            ">📊 Confidence Score</h3>
        </div>
        """, unsafe_allow_html=True)
        gauge_fig = create_gauge_chart(confidence, "Model Confidence")
        st.plotly_chart(gauge_fig, use_container_width=True)
        
        st.subheader("📈 Probability Breakdown")
        prob_fig = create_probability_chart(prediction_proba, target_encoder.classes_)
        st.plotly_chart(prob_fig, use_container_width=True)
        
        # Add HTML anchor
        st.markdown('<div id="insights-about"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 0.75rem 1.5rem;
            border-radius: 10px;
            margin: 1rem 0 0.5rem 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        ">
            <h3 style="
                color: white;
                font-size: 1.5rem;
                font-weight: 600;
                margin: 0;
                text-shadow: 0 1px 2px rgba(0,0,0,0.2);
            ">💡 Insights</h3>
        </div>
        """, unsafe_allow_html=True)
        if confidence > 0.8:
            st.success(f"🎯 **High Confidence Prediction** (>{confidence:.1%})")
            st.write("The model is very confident in this prediction.")
        elif confidence > 0.6:
            st.warning(f"⚖️ **Moderate Confidence** ({confidence:.1%})")
            st.write("The model has moderate confidence. You might have characteristics of both personality types.")
        else:
            st.error(f"🤷 **Low Confidence** ({confidence:.1%})")
            st.write("The model has low confidence. Your characteristics show a mix of both personality types.")
        
        st.subheader(f"📖 About {predicted_personality}s")
        if predicted_personality == 'Extrovert':
            st.markdown("""
            **Extroverts** typically:
            - Gain energy from social interactions
            - Enjoy being around people
            - Are often outgoing and talkative
            - Prefer group activities
            - Think out loud and process externally
            """)
        else:
            st.markdown("""
            **Introverts** typically:
            - Gain energy from solitude
            - Prefer smaller social groups
            - Are often thoughtful and reflective
            - Enjoy quiet activities
            - Think internally before speaking
            """)
            
        if model_stats and 'by_class' in model_stats:
            st.subheader(f"📊 Model Performance for {predicted_personality}s")
            if predicted_personality in model_stats['by_class']:
                class_data = model_stats['by_class'][predicted_personality]
                with st.container():
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric(
                            label="Total Training Samples",
                            value=f"{class_data['total_samples']:,}",
                            help=f"Number of {predicted_personality}s in training data"
                        )
                    with col_stat2:
                        st.metric(
                            label="Model Accuracy",
                            value=f"{class_data['accuracy']:.1%}",
                            help=f"How often the model correctly identifies {predicted_personality}s"
                        )
                    with col_stat3:
                        st.metric(
                            label="Misclassified",
                            value=f"{class_data['misclassified']:,}",
                            delta=f"{class_data['misclassification_rate']:.1%} rate",
                            delta_color="inverse",
                            help=f"Number of {predicted_personality}s the model got wrong"
                        )
                if class_data['accuracy'] > 0.95:
                    accuracy_feedback = "🎯 **Excellent**: The model performs very well on this personality type"
                elif class_data['accuracy'] > 0.90:
                    accuracy_feedback = "✅ **Good**: The model has strong performance on this personality type"
                elif class_data['accuracy'] > 0.80:
                    accuracy_feedback = "⚖️ **Moderate**: The model has decent performance on this personality type"
                else:
                    accuracy_feedback = "⚠️ **Challenging**: This personality type is more difficult for the model"
                st.markdown(accuracy_feedback)
                
        if misclassified_features is not None:
            # Add HTML anchor
            st.markdown('<div id="reliability-analysis"></div>', unsafe_allow_html=True)
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                padding: 0.75rem 1.5rem;
                border-radius: 10px;
                margin: 1rem 0 0.5rem 0;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            ">
                <h3 style="
                    color: white;
                    font-size: 1.5rem;
                    font-weight: 600;
                    margin: 0;
                    text-shadow: 0 1px 2px rgba(0,0,0,0.2);
                ">🎯 Prediction Reliability Analysis</h3>
            </div>
            """, unsafe_allow_html=True)
            user_input_array = input_data.iloc[0].values
            min_distance, min_index = calculate_distance_to_misclassified(user_input_array, misclassified_features)
            if min_distance is not None:
                # Get baseline distance - use class-specific if available, otherwise overall
                if predicted_personality in baseline_metrics.get('class_specific', {}):
                    baseline_distance = baseline_metrics['class_specific'][predicted_personality]['weighted_baseline_distance']
                else:
                    baseline_distance = baseline_metrics['overall']['weighted_baseline_distance']
                
                distance_ratio = min_distance / baseline_distance
                with st.container():
                    st.markdown("**🔍 Distance Analysis:**")
                    col_dist1, col_dist2 = st.columns(2)
                    with col_dist1:
                        st.metric(
                            label="Distance to Nearest Problematic Sample",
                            value=f"{min_distance:.2f}",
                            help="Lower values indicate your input is similar to samples the model often misclassifies"
                        )
                    with col_dist2:
                        st.metric(
                            label="Reliability Score",
                            value=f"{distance_ratio:.2f}x",
                            delta=f"vs baseline ({baseline_distance:.2f})",
                            help="Higher values indicate more reliable predictions"
                        )
                if distance_ratio < 0.5:
                    st.error("⚠️ **High Risk Zone**: Your characteristics are very similar to samples the model often misclassifies. Take this prediction with caution.")
                elif distance_ratio < 1.0:
                    st.warning("⚡ **Moderate Risk**: Your characteristics are somewhat similar to problematic samples. The prediction may be less reliable.")
                elif distance_ratio < 1.5:
                    st.info("✅ **Safe Zone**: Your characteristics are in a reliable prediction region.")
                else:
                    st.success("🎯 **High Confidence Zone**: Your characteristics are well within reliable prediction boundaries.")
                
                if enhanced_misclassified_df is not None and min_index is not None:
                    st.subheader("🔍 Closest Misclassified Sample")
                    closest_sample_info = get_closest_misclassified_sample_info(
                        all_inputs, enhanced_misclassified_df, feature_columns, min_index
                    )
                    if closest_sample_info:
                        with st.container():
                            col_sample1, col_sample2, col_sample3 = st.columns(3)
                            with col_sample1:
                                st.metric(
                                    label="Sample ID",
                                    value=f"#{closest_sample_info['sample_id']}",
                                    help="Identifier of the closest misclassified sample"
                                )
                            with col_sample2:
                                st.metric(
                                    label="True Label",
                                    value=closest_sample_info['true_label'],
                                    help="What this sample's actual personality type was"
                                )
                            with col_sample3:
                                st.metric(
                                    label="Model's Prediction",
                                    value=closest_sample_info['predicted_label'],
                                    help=f"What the model incorrectly predicted (confidence: {closest_sample_info['confidence']:.1%})"
                                )
                        st.markdown("**📊 Feature Comparison with Closest Misclassified Sample:**")
                        comparison_data = []
                        for feature, diff_info in list(closest_sample_info['feature_differences'].items())[:5]:
                            user_val = diff_info['user_value']
                            sample_val = diff_info['sample_value']
                            difference = diff_info['difference']
                            if feature in categorical_columns:
                                user_display = decode_categorical_value(feature, user_val, label_encoders, categorical_columns)
                                sample_display = decode_categorical_value(feature, sample_val, label_encoders, categorical_columns)
                                diff_display = "Same" if difference == 0.0 else "Different"
                            elif isinstance(user_val, (int, float)) and isinstance(sample_val, (int, float)):
                                user_display = f"{user_val:.2f}"
                                sample_display = f"{sample_val:.2f}"
                                diff_display = f"{difference:.2f}"
                            else:
                                user_display = str(user_val)
                                sample_display = str(sample_val)
                                diff_display = "Same" if difference == 0.0 else "Different"
                            comparison_data.append({
                                'Feature': feature.replace('_', ' ').title(),
                                'Your Value': user_display,
                                'Sample Value': sample_display,
                                'Difference': diff_display
                            })
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                        max_diff_feature = list(closest_sample_info['feature_differences'].keys())[0]
                        max_diff_value = closest_sample_info['feature_differences'][max_diff_feature]['difference']
                        user_val = closest_sample_info['feature_differences'][max_diff_feature]['user_value']
                        sample_val = closest_sample_info['feature_differences'][max_diff_feature]['sample_value']
                        if isinstance(user_val, (int, float)) and isinstance(sample_val, (int, float)):
                            if max_diff_value > 2:
                                st.info(f"💡 **Key Insight**: Your biggest difference is in '{max_diff_feature.replace('_', ' ').title()}' (difference: {max_diff_value:.2f}). This might be why you're in a safer prediction zone.")
                            else:
                                st.warning(f"⚠️ **Key Insight**: Your characteristics are quite similar to this misclassified sample, especially in '{max_diff_feature.replace('_', ' ').title()}' (difference: {max_diff_value:.2f}). Be cautious about the prediction.")
                        else:
                            if max_diff_value == 0:
                                st.warning(f"⚠️ **Key Insight**: You have the exact same '{max_diff_feature.replace('_', ' ').title()}' as this misclassified sample. Be cautious about the prediction.")
                            else:
                                st.info(f"💡 **Key Insight**: Your biggest difference is in '{max_diff_feature.replace('_', ' ').title()}' (different category). This might be why you're in a safer prediction zone.")
                        
                        with st.expander("🔎 Detailed Feature Comparison"):
                            st.markdown(f"**Misclassified Sample Details:** {closest_sample_info['description']}")
                            detailed_comparison = []
                            for feature, diff_info in closest_sample_info['feature_differences'].items():
                                user_val = diff_info['user_value']
                                sample_val = diff_info['sample_value']
                                difference = diff_info['difference']
                                
                                # Decode categorical values for display
                                if feature in categorical_columns:
                                    user_display = decode_categorical_value(feature, user_val, label_encoders, categorical_columns)
                                    sample_display = decode_categorical_value(feature, sample_val, label_encoders, categorical_columns)
                                    diff_display = "Same" if difference == 0.0 else "Different"
                                    similarity = "100%" if difference == 0.0 else "0%"
                                elif isinstance(user_val, (int, float)) and isinstance(sample_val, (int, float)):
                                    user_display = f"{user_val:.3f}"
                                    sample_display = f"{sample_val:.3f}"
                                    diff_display = f"{difference:.3f}"
                                    if difference <= 10:
                                        similarity = f"{max(0, 100 - (difference * 10)):.1f}%"
                                    else:
                                        similarity = "Very Different"
                                else:
                                    user_display = str(user_val)
                                    sample_display = str(sample_val)
                                    diff_display = "Same" if difference == 0.0 else "Different"
                                    similarity = "100%" if difference == 0.0 else "0%"
                                detailed_comparison.append({
                                    'Feature': feature.replace('_', ' ').title(),
                                    'Your Value': user_display,
                                    'Sample Value': sample_display,
                                    'Absolute Difference': diff_display,
                                    'Similarity': similarity
                                })
                            detailed_df = pd.DataFrame(detailed_comparison)
                            st.dataframe(detailed_df, use_container_width=True, hide_index=True)
                            st.markdown("""
                            **How to interpret this:**
                            - **Smaller differences** mean you're more similar to this problematic sample
                            - **Larger differences** mean you're less likely to be misclassified for the same reasons
                            - **High similarity percentages** in multiple features indicate higher risk
                            """)
                            
                with st.expander("ℹ️ Understanding the Reliability Analysis"):
                    st.markdown(f"""
                    **What does this mean?**
                    
                    - **Distance to Problematic Samples**: We measure how similar your input is to samples that our model historically struggles with
                    - **Reliability Score**: Compares your distance to the average distance between correctly classified samples
                    - **Baseline**: The weighted average distance between correctly classified training samples ({baseline_distance:.2f})
                    
                    **Risk Zones:**
                    - 🔴 **High Risk** (< 0.5x): Very close to problematic samples
                    - 🟡 **Moderate Risk** (0.5x - 1.0x): Somewhat close to problematic samples  
                    - 🟢 **Safe Zone** (1.0x - 1.5x): Normal distance from problematic samples
                    - 🟦 **High Confidence** (> 1.5x): Far from any problematic samples
                    
                    This analysis helps you understand how much to trust the prediction based on the model's historical performance.
                    """)
        
        # --- Interactive PCA Visualization Section ---
        st.markdown("---")
        # Add HTML anchor
        st.markdown('<div id="community-map"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
            padding: 0.75rem 1.5rem;
            border-radius: 10px;
            margin: 1rem 0 0.5rem 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        ">
            <h3 style="
                color: white;
                font-size: 1.5rem;
                font-weight: 600;
                margin: 0;
                text-shadow: 0 1px 2px rgba(0,0,0,0.2);
            ">�️ Our Personality Map</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **Explore our community visualization!** See how you compare to others in our interactive 2D personality map 
        using Principal Component Analysis (PCA), and join our growing community of personality explorers.
        """)
        
        # === SECTION 1: Your Closest Community Match ===
        st.markdown("---")
        # Show closest person in the community map (if any members exist)
        try:
            # Load existing PCA submissions from Supabase
            db_manager = get_supabase_manager()
            existing_pca_data = db_manager.get_pca_submissions()
            
            if len(existing_pca_data) > 0:
                # Clean the data - remove any invalid entries
                existing_pca_data = existing_pca_data.dropna(subset=['display_name', 'actual_personality', 'feature_vector'])
                existing_pca_data = existing_pca_data[existing_pca_data['feature_vector'].apply(
                    lambda x: isinstance(x, list) and len(x) > 0 and all(isinstance(val, (int, float)) for val in x)
                )]
                
                if len(existing_pca_data) > 0:
                    # Calculate distances to all existing members
                    user_vector = input_data.iloc[0].values
                    distances = []
                    
                    for idx, member in existing_pca_data.iterrows():
                        member_vector = np.array(member['feature_vector'])
                        if len(member_vector) == len(user_vector):
                            # Calculate Euclidean distance
                            distance = np.sqrt(np.sum((user_vector - member_vector) ** 2))
                            distances.append({
                                'distance': distance,
                                'name': member['display_name'],
                                'personality': member['actual_personality'],
                                'linkedin': member.get('linkedin_profile', None),
                                'confidence': member['prediction_confidence'],
                                'submission_date': pd.to_datetime(member['timestamp']).strftime('%Y-%m-%d')
                            })
                    
                    if distances:
                        # Find the closest person
                        closest_person = min(distances, key=lambda x: x['distance'])
                        
                        # Display closest person info
                        st.markdown("### 🎯 Your Closest Community Match")
                        
                        # Create an attractive info box for the closest person
                        linkedin_link = ""
                        if closest_person['linkedin'] and pd.notna(closest_person['linkedin']):
                            linkedin_url = closest_person['linkedin']
                            linkedin_link = f' | <a href="{linkedin_url}" target="_blank" style="color: white; text-decoration: underline;">LinkedIn Profile</a>'
                        
                        # Calculate similarity percentage (inverse of distance, normalized)
                        max_possible_distance = np.sqrt(len(user_vector))  # Theoretical max for normalized features
                        similarity_pct = max(0, (1 - closest_person['distance'] / max_possible_distance) * 100)
                        
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            padding: 1rem 1.5rem;
                            border-radius: 10px;
                            margin: 0.5rem 0;
                            color: white;
                            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                        ">
                            <h4 style="margin: 0 0 0.5rem 0; color: white;">👤 {closest_person['name']}</h4>
                            <div style="display: flex; flex-wrap: wrap; gap: 1rem; align-items: center;">
                                <span><strong>🧬 Personality:</strong> {closest_person['personality']}</span>
                                <span><strong>📊 Model Confidence:</strong> {closest_person['confidence']:.1%}</span>
                                <span><strong>📅 Joined:</strong> {closest_person['submission_date']}</span>
                            </div>
                            <div style="margin-top: 0.5rem;">
                                <span><strong>🎯 Similarity Score:</strong> {similarity_pct:.1f}% | <strong>📏 Distance:</strong> {closest_person['distance']:.2f}</span>
                                {f'<span style="margin-left: 1rem;">{linkedin_link}</span>' if linkedin_link else ''}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Add interpretation
                        if similarity_pct >= 80:
                            st.success("🎉 **Excellent Match!** You have very similar personality characteristics to this community member.")
                        elif similarity_pct >= 60:
                            st.info("✨ **Good Match!** You share many personality traits with this community member.")
                        elif similarity_pct >= 40:
                            st.warning("🤔 **Moderate Match** - You have some similarities but also notable differences.")
                        else:
                            st.error("🌟 **Unique Profile!** You're quite different from existing members - you'll add great diversity to our community!")
                        
                        st.markdown("---")
        
        except FileNotFoundError:
            # No existing data file, which is fine
            pass
        except Exception as e:
            # Silently handle any errors in closest person calculation
            pass
        
        # === SECTION 2: Add Yourself to Our Personality Map ===
        st.markdown("---")
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 0.75rem 1.5rem;
            border-radius: 10px;
            margin: 1rem 0 0.5rem 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        ">
            <h4 style="
                color: white;
                font-size: 1.3rem;
                font-weight: 600;
                margin: 0;
                text-shadow: 0 1px 2px rgba(0,0,0,0.2);
            ">📊 Add Yourself to Our Personality Map!</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **Join our community visualization!** Share your personality data and see how you compare to others 
        in our interactive 2D personality map using Principal Component Analysis (PCA).
        """)
        
        # PCA submission form
        with st.container():
            
            # Ask if they want to be included
            include_in_pca = st.checkbox(
                "🗺️ I want to add my personality to the interactive community map",
                key="include_in_pca"
            )
            
            if include_in_pca:
                st.info("Great! Your data will be added to our community personality visualization.")
                
                # Personal information for the map
                col_pca1, col_pca2 = st.columns(2)
                
                with col_pca1:
                    display_name = st.text_input(
                        "Display Name (required) *",
                        placeholder="e.g., John D., Alex_2024, Sarah M., etc.",
                        key="pca_display_name",
                        help="This name will appear when hovering over your point on the map"
                    )
                
                with col_pca2:
                    linkedin_profile = st.text_input(
                        "LinkedIn Profile (optional)",
                        placeholder="e.g., https://linkedin.com/in/yourprofile",
                        key="pca_linkedin",
                        help="Optional: Add your LinkedIn profile for networking opportunities"
                    )
                
                # Actual personality confirmation
                actual_personality_pca = st.radio(
                    "What's your actual personality type? *",
                    options=["Introvert", "Extrovert"],
                    key="actual_personality_pca",
                    help="This will determine your point color on the map"
                )
                
                # Model prediction display and user confidence slider
                st.markdown("**🤖 Model's Prediction for You:**")
                col_pred_info1, col_pred_info2 = st.columns(2)
                
                with col_pred_info1:
                    st.info(f"**Predicted:** {predicted_personality}")
                
                with col_pred_info2:
                    st.info(f"**Model Confidence:** {confidence:.1%}")
                
                # Convert model confidence to 0-100 scale (0=Introvert, 100=Extrovert)
                if predicted_personality == "Extrovert":
                    model_confidence_scale = int(confidence * 100)
                else:  # Introvert
                    model_confidence_scale = int((1 - confidence) * 100)
                
                st.markdown("**🎯 Your Confidence Assessment:**")
                st.markdown("Use the slider below to indicate where you believe yourself to be on the personality spectrum:")
                
                user_confidence_scale = st.slider(
                    "Personality Confidence Scale",
                    min_value=0,
                    max_value=100,
                    value=model_confidence_scale,
                    step=1,
                    key="user_confidence_scale",
                    help="0 = Strongly Introvert, 50 = Balanced, 100 = Strongly Extrovert"
                )
                
                # Show interpretation and model pointer
                col_scale1, col_scale2, col_scale3 = st.columns(3)
                with col_scale1:
                    st.caption("← **0: Strong Introvert**")
                with col_scale2:
                    st.caption("**50: Balanced**")
                with col_scale3:
                    st.caption("**100: Strong Extrovert** →")
                
                # Show model's position and difference
                confidence_difference = abs(user_confidence_scale - model_confidence_scale)
                st.caption(f"🤖 **Model placed you at:** {model_confidence_scale}/100")
                
                if confidence_difference == 0:
                    st.success(f"✅ **Perfect Agreement!** You and the model are in complete alignment.")
                elif confidence_difference <= 10:
                    st.info(f"📍 **Close Agreement** (difference: {confidence_difference} points)")
                elif confidence_difference <= 25:
                    st.warning(f"⚖️ **Moderate Difference** (difference: {confidence_difference} points)")
                else:
                    st.error(f"❌ **Significant Disagreement** (difference: {confidence_difference} points)")
                
                # Privacy and data usage agreement
                st.markdown("**🔒 Privacy & Data Usage:**")
                privacy_agreement = st.checkbox(
                    "I agree to share my anonymized personality characteristics for research and community visualization purposes",
                    key="privacy_agreement"
                )
                
                # Submit to PCA map button
                if st.button("🗺️ Add Me to the Personality Map", type="primary", key="submit_to_pca"):
                    if display_name and privacy_agreement:
                        # Calculate confidence difference for color coding
                        confidence_diff = abs(user_confidence_scale - model_confidence_scale)
                        
                        # Prepare PCA submission data
                        pca_submission_data = {
                            "timestamp": pd.Timestamp.now().isoformat(),
                            "display_name": display_name,
                            "linkedin_profile": linkedin_profile if linkedin_profile else None,
                            "actual_personality": actual_personality_pca,
                            "predicted_personality": predicted_personality,
                            "prediction_confidence": float(confidence),
                            "model_confidence_scale": model_confidence_scale,
                            "user_confidence_scale": user_confidence_scale,
                            "confidence_difference": confidence_diff,
                            "user_characteristics": dict(all_inputs),
                            "reliability_score": float(distance_ratio) if 'distance_ratio' in locals() else None,
                            "distance_to_problematic": float(min_distance) if 'min_distance' in locals() and min_distance is not None else None,
                            "feature_vector": input_data.iloc[0].tolist()  # For PCA calculation
                        }
                        
                        # Save PCA submission data to Supabase
                        try:
                            db_manager = get_supabase_manager()
                            success = db_manager.save_pca_submission(pca_submission_data)
                            
                            if success:
                                st.success("🎉 Thank you! You've been added to our personality map.")
                                st.info("💡 Your data point will appear on the interactive map below (refresh the page to see updates).")
                                
                                if linkedin_profile:
                                    st.balloons()
                                    st.markdown(f"🌟 **Welcome to the community, {display_name}!** Your LinkedIn profile will be accessible to others for networking.")
                                else:
                                    st.markdown(f"🌟 **Welcome to the community, {display_name}!**")
                            else:
                                st.error("⚠️ There was an error saving your data to the personality map. Please try again later.")
                                
                        except Exception as e:
                            st.error("⚠️ There was an error saving your data to the personality map. Please try again later.")
                            st.caption(f"Error details: {str(e)}")
                    else:
                        if not display_name:
                            st.error("❌ Please enter a display name.")
                        if not privacy_agreement:
                            st.error("❌ Please agree to the privacy and data usage terms.")
                            
            else:
                st.markdown("💡 **Tip:** Joining the community map helps others see the diversity of personality types and creates networking opportunities!")
        
        # === SECTION 3: Interactive Community Personality Map ===
        st.markdown("---")
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            padding: 0.75rem 1.5rem;
            border-radius: 10px;
            margin: 1rem 0 0.5rem 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        ">
            <h4 style="
                color: white;
                font-size: 1.3rem;
                font-weight: 600;
                margin: 0;
                text-shadow: 0 1px 2px rgba(0,0,0,0.2);
            ">🗺️ Interactive Community Personality Map</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Important note about visibility issues
        st.info("""
        📍 **Important Note:** Due to the current amount of data, some community members might not be visible on this PCA graph. 
        This occurs when certain data points are positioned far away from the main cluster and fall outside the visible area. 
        I am actively debugging this issue to ensure all participants are properly displayed.
        
        🔄 **Not seeing yourself?** If you recently joined and don't see your point on the map, please check back in a day or two. 
        As we gather more community data, the visualization will become more stable and comprehensive, making all points visible.
        """)
        
        # Add visualization mode toggle
        viz_mode = st.radio(
            "Choose visualization mode:",
            options=["🎨 Personality Type", "🎯 Confidence Agreement"],
            key="viz_mode",
            help="Personality Type: Colors based on actual personality | Confidence Agreement: Colors based on agreement with model"
        )
        
        try:
            # Load existing PCA submissions from Supabase
            db_manager = get_supabase_manager()
            pca_submissions = db_manager.get_pca_submissions()
            
            # For existing entries without confidence data, assume 100% agreement
            if 'confidence_difference' not in pca_submissions.columns:
                pca_submissions['confidence_difference'] = 0
                pca_submissions['user_confidence_scale'] = 100  # Default assumption
                pca_submissions['model_confidence_scale'] = 100  # Default assumption
            else:
                # Fill missing values with 0 (perfect agreement)
                pca_submissions['confidence_difference'] = pca_submissions['confidence_difference'].fillna(0)
                pca_submissions['user_confidence_scale'] = pca_submissions['user_confidence_scale'].fillna(100)
                pca_submissions['model_confidence_scale'] = pca_submissions['model_confidence_scale'].fillna(100)
            
            if len(pca_submissions) >= 2:  # Need at least 2 samples to for PCA
                # Clean the data - remove any invalid entries
                pca_submissions = pca_submissions.dropna(subset=['display_name', 'actual_personality', 'feature_vector'])
                
                # Filter out any rows with empty or invalid feature vectors
                pca_submissions = pca_submissions[pca_submissions['feature_vector'].apply(
                    lambda x: isinstance(x, list) and len(x) > 0 and all(isinstance(val, (int, float)) for val in x)
                )]
                
                if len(pca_submissions) >= 2:  # Re-check after cleaning
                    # Create PCA visualization
                    from sklearn.decomposition import PCA
                    
                    # Extract feature vectors for PCA
                    feature_vectors = np.array(pca_submissions['feature_vector'].tolist())
                    
                    # Apply PCA to reduce to 2D
                    pca = PCA(n_components=2)
                    pca_coords = pca.fit_transform(feature_vectors)
                    
                    # Create visualization dataframe
                    viz_df = pd.DataFrame({
                        'PC1': pca_coords[:, 0],
                        'PC2': pca_coords[:, 1],
                        'Name': pca_submissions['display_name'],
                        'Personality': pca_submissions['actual_personality'],
                        'Predicted': pca_submissions['predicted_personality'],
                        'Confidence': pca_submissions['prediction_confidence'],
                        'User_Confidence_Scale': pca_submissions['user_confidence_scale'],
                        'Model_Confidence_Scale': pca_submissions['model_confidence_scale'],
                        'Confidence_Difference': pca_submissions['confidence_difference'],
                        'LinkedIn': pca_submissions['linkedin_profile'].fillna('Not provided'),
                        'Submission_Date': pd.to_datetime(pca_submissions['timestamp']).dt.strftime('%Y-%m-%d')
                    })
                    
                    # Ensure personality column contains only valid values
                    valid_personalities = ['Introvert', 'Extrovert']
                    viz_df = viz_df[viz_df['Personality'].isin(valid_personalities)].reset_index(drop=True)
                    
                    if len(viz_df) > 0:  # Double-check we still have data after filtering
                        
                        # Function to get color based on confidence difference
                        def get_confidence_color(diff):
                            """Get color based on confidence difference (0=green, 25=yellow, 50+=red)"""
                            if diff == 0:
                                return '#00ff00'  # Pure green
                            elif diff <= 25:
                                # Gradient from green to yellow (0-25)
                                ratio = diff / 25
                                return f'rgb({int(255 * ratio)}, 255, 0)'
                            else:
                                # Gradient from yellow to red (25-50+)
                                ratio = min((diff - 25) / 25, 1.0)  # Cap at 1.0 for diff >= 50
                                return f'rgb(255, {int(255 * (1 - ratio))}, 0)'
                        
                        # Create interactive scatter plot using go.Figure for more control
                        fig_pca = go.Figure()
                        
                        if viz_mode == "🎨 Personality Type":
                            # Original personality-based coloring
                            for personality in ['Introvert', 'Extrovert']:
                                personality_data = viz_df[viz_df['Personality'] == personality]
                                if len(personality_data) > 0:
                                    color = '#ff7f0e' if personality == 'Introvert' else '#1f77b4'
                                    
                                    fig_pca.add_trace(go.Scatter(
                                        x=personality_data['PC1'],
                                        y=personality_data['PC2'],
                                        mode='markers',
                                        name=personality,
                                        marker=dict(
                                            size=10,
                                            color=color,
                                            line=dict(width=1, color='white')
                                        ),
                                        customdata=list(zip(
                                            personality_data['Name'],
                                            personality_data['Personality'],
                                            personality_data['Predicted'],
                                            personality_data['Confidence'],
                                            personality_data['LinkedIn'],
                                            personality_data['Submission_Date']
                                        )),
                                        hovertemplate=
                                        "<b>%{customdata[0]}</b><br>" +
                                        "Actual: %{customdata[1]}<br>" +
                                        "Predicted: %{customdata[2]}<br>" +
                                        "Model Confidence: %{customdata[3]:.2%}<br>" +
                                        "LinkedIn: %{customdata[4]}<br>" +
                                        "Joined: %{customdata[5]}<br>" +
                                        "<extra></extra>"
                                    ))
                        else:
                            # Confidence agreement based coloring
                            colors = [get_confidence_color(diff) for diff in viz_df['Confidence_Difference']]
                            
                            fig_pca.add_trace(go.Scatter(
                                x=viz_df['PC1'],
                                y=viz_df['PC2'],
                                mode='markers',
                                name='Confidence Agreement',
                                marker=dict(
                                    size=12,
                                    color=colors,
                                    line=dict(width=2, color='white'),
                                    opacity=0.8
                                ),
                                customdata=list(zip(
                                    viz_df['Name'],
                                    viz_df['Personality'],
                                    viz_df['Predicted'],
                                    viz_df['Confidence'],
                                    viz_df['User_Confidence_Scale'],
                                    viz_df['Model_Confidence_Scale'],
                                    viz_df['Confidence_Difference'],
                                    viz_df['LinkedIn'],
                                    viz_df['Submission_Date']
                                )),
                                hovertemplate=
                                "<b>%{customdata[0]}</b><br>" +
                                "Actual: %{customdata[1]}<br>" +
                                "Predicted: %{customdata[2]}<br>" +
                                "Model Confidence: %{customdata[3]:.2%}<br>" +
                                "User Rating: %{customdata[4]}/100<br>" +
                                "Model Rating: %{customdata[5]}/100<br>" +
                                "Difference: %{customdata[6]} points<br>" +
                                "LinkedIn: %{customdata[7]}<br>" +
                                "Joined: %{customdata[8]}<br>" +
                                "<extra></extra>"
                            ))
                            
                            # Add color legend for confidence mode
                            st.markdown("""
                            **🎯 Confidence Agreement Color Legend:**
                            - 🟢 **Green**: Perfect agreement (difference = 0)
                            - 🟡 **Yellow**: Moderate difference (difference ≈ 25)
                            - 🔴 **Red**: High disagreement (difference ≥ 50)
                            """)
                        
                        # Update title based on mode
                        title_suffix = "by Personality Type" if viz_mode == "🎨 Personality Type" else "by Confidence Agreement"
                        
                        # Calculate axis ranges with padding to ensure all points are visible
                        pc1_range = viz_df['PC1'].max() - viz_df['PC1'].min()
                        pc2_range = viz_df['PC2'].max() - viz_df['PC2'].min()
                        pc1_padding = pc1_range * 0.1 if pc1_range > 0 else 1
                        pc2_padding = pc2_range * 0.1 if pc2_range > 0 else 1
                        
                        # Update layout
                        fig_pca.update_layout(
                            title=f"Community Personality Map ({len(viz_df)} members) - {title_suffix}",
                            xaxis_title=f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)',
                            yaxis_title=f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)',
                            height=600,
                            showlegend=True,
                            legend=dict(
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=0.01
                            ),
                            # Ensure all points are visible with proper axis ranges
                            xaxis=dict(
                                range=[viz_df['PC1'].min() - pc1_padding, viz_df['PC1'].max() + pc1_padding]
                            ),
                            yaxis=dict(
                                range=[viz_df['PC2'].min() - pc2_padding, viz_df['PC2'].max() + pc2_padding]
                            )
                        )
                        
                        st.plotly_chart(fig_pca, use_container_width=True)
                    else:
                        st.warning("No valid personality data found after filtering.")
                    
                    # Display community stats
                    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
                    
                    with col_stats1:
                        st.metric("Total Members", len(viz_df))
                    
                    with col_stats2:
                        introvert_count = (viz_df['Personality'] == 'Introvert').sum()
                        st.metric("Introverts", introvert_count)
                    
                    with col_stats3:
                        extrovert_count = (viz_df['Personality'] == 'Extrovert').sum()
                        st.metric("Extroverts", extrovert_count)
                    
                    with col_stats4:
                        avg_confidence = viz_df['Confidence'].mean()
                        st.metric("Avg Model Confidence", f"{avg_confidence:.1%}")
                    
                    # Additional stats for confidence agreement
                    if viz_mode == "🎯 Confidence Agreement":
                        st.markdown("### 📊 Confidence Agreement Statistics")
                        col_conf1, col_conf2, col_conf3, col_conf4 = st.columns(4)
                        
                        with col_conf1:
                            perfect_agreement = (viz_df['Confidence_Difference'] == 0).sum()
                            st.metric("Perfect Agreement", f"{perfect_agreement}/{len(viz_df)}")
                        
                        with col_conf2:
                            close_agreement = (viz_df['Confidence_Difference'] <= 10).sum()
                            st.metric("Close Agreement (≤10)", f"{close_agreement}/{len(viz_df)}")
                        
                        with col_conf3:
                            moderate_diff = ((viz_df['Confidence_Difference'] > 10) & (viz_df['Confidence_Difference'] <= 25)).sum()
                            st.metric("Moderate Diff (11-25)", f"{moderate_diff}/{len(viz_df)}")
                        
                        with col_conf4:
                            high_disagreement = (viz_df['Confidence_Difference'] > 25).sum()
                            st.metric("High Disagreement (>25)", f"{high_disagreement}/{len(viz_df)}")
                        
                        # Average confidence difference
                        avg_diff = viz_df['Confidence_Difference'].mean()
                        st.info(f"📊 **Average Confidence Difference:** {avg_diff:.1f} points")
                    

                    # Show recent additions
                    if len(viz_df) > 0:
                        st.markdown("**🆕 Recent Community Members:**")
                        recent_members = viz_df.sort_values('Submission_Date', ascending=False).head(5)
                        for _, member in recent_members.iterrows():
                            linkedin_text = f" ([LinkedIn]({member['LinkedIn']}))" if member['LinkedIn'] != 'Not provided' else ""
                            st.markdown(f"• **{member['Name']}** - {member['Personality']} (joined {member['Submission_Date']}){linkedin_text}")
                else:
                    st.info("🎉 We have community members but need at least 2 valid entries to create the interactive PCA map.")
            elif len(pca_submissions) == 1:
                # Show single member without PCA plot
                st.info("🎉 We have our first community member! We need at least 2 members to create the interactive PCA map.")
                
                # Display the single member
                member = pca_submissions.iloc[0]
                st.markdown("**👑 Pioneer Member:**")
                linkedin_text = f" ([LinkedIn]({member['linkedin_profile']}))" if pd.notna(member['linkedin_profile']) else ""
                submission_date = pd.to_datetime(member['timestamp']).strftime('%Y-%m-%d')
                st.markdown(f"• **{member['display_name']}** - {member['actual_personality']} (joined {submission_date}){linkedin_text}")
                
                # Show basic stats
                col_stats1, col_stats2, col_stats3 = st.columns(3)
                with col_stats1:
                    st.metric("Total Members", 1)
                with col_stats2:
                    personality_type = member['actual_personality']
                    if personality_type == 'Introvert':
                        st.metric("Introverts", 1)
                    else:
                        st.metric("Extroverts", 1)
                with col_stats3:
                    st.metric("Avg Confidence", f"{member['prediction_confidence']:.1%}")
            
            else:
                st.info("🏗️ No community members yet! Be the first to join our personality map.")
                
        except FileNotFoundError:
            st.info("🏗️ No community members yet! Be the first to join our personality map.")
        except Exception as e:
            st.error(f"Error loading community map: {str(e)}")
            # For debugging, show more details
            st.caption("If this error persists, please refresh the page or contact support.")

        # --- User Feedback Section ---
        st.markdown("---")
        # Add HTML anchor
        st.markdown('<div id="feedback-section"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
            padding: 0.75rem 1.5rem;
            border-radius: 10px;
            margin: 1rem 0 0.5rem 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        ">
            <h3 style="
                color: white;
                font-size: 1.5rem;
                font-weight: 600;
                margin: 0;
                text-shadow: 0 1px 2px rgba(0,0,0,0.2);
            ">🤔 Help Us Improve: Is This Prediction Correct?</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **Your feedback helps improve our model!** If you strongly believe this prediction is incorrect, 
        please let us know. Your input will help us understand edge cases and improve future predictions.
        """)
        
        # Feedback form
        with st.container():
            
            # Prediction accuracy feedback
            feedback_options = [
                "Select an option...",
                "✅ Yes, this prediction is accurate",
                "❌ No, this prediction is completely wrong",
                "🤷 I'm not sure / It's complicated"
            ]
            
            user_feedback = st.selectbox(
                "Do you agree with this personality prediction?",
                options=feedback_options,
                key="prediction_feedback"
            )
            
            if user_feedback == "❌ No, this prediction is completely wrong":
                st.warning("Thank you for the feedback! Please help us understand why:")
                
                # Ask for their actual personality type
                actual_personality = st.radio(
                    "What do you consider yourself to be?",
                    options=["Introvert", "Extrovert"],
                    key="actual_personality"
                )
                
                # Ask if they want to share their data
                st.markdown("**🔬 Help Improve Our Model:**")
                share_data = st.checkbox(
                    "I'm willing to share my characteristics as a misclassified example to help improve the model",
                    key="share_data_consent"
                )
                
                if share_data:
                    # Data sharing options
                    sharing_preference = st.radio(
                        "How would you like to share your data?",
                        options=[
                            "📊 Anonymously (no personal information)",
                            "📝 With optional name/identifier for credit"
                        ],
                        key="sharing_preference"
                    )
                    
                    # Optional name field
                    user_name = ""
                    if sharing_preference == "📝 With optional name/identifier for credit":
                        user_name = st.text_input(
                            "Name or identifier (optional):",
                            placeholder="e.g., John D., Anonymous123, etc.",
                            key="user_name"
                        )
                    
                    # Additional context
                    additional_context = st.text_area(
                        "Any additional context about why you think the prediction is wrong? (optional)",
                        placeholder="e.g., 'I'm an introvert but very social at work', 'I have social anxiety but love parties', etc.",
                        key="additional_context"
                    )
                    
                    # Submit feedback button
                    if st.button("📤 Submit Feedback", type="primary", key="submit_feedback"):
                        # Prepare feedback data
                        feedback_data = {

                            "timestamp": pd.Timestamp.now().isoformat(),
                            "predicted_personality": predicted_personality,
                            "actual_personality": actual_personality,
                            "confidence_score": float(confidence),
                            "user_characteristics": dict(all_inputs),
                            "sharing_preference": sharing_preference,
                            "user_name": user_name if user_name else "Anonymous",
                            "additional_context": additional_context,
                            "reliability_score": float(distance_ratio) if 'distance_ratio' in locals() else None,
                            "distance_to_problematic": float(min_distance) if 'min_distance' in locals() and min_distance is not None else None
                        }
                        
                        # Save feedback to Supabase
                        try:
                            db_manager = get_supabase_manager()
                            success = db_manager.save_user_feedback(feedback_data)
                            
                            if success:
                                st.success("✅ Thank you! Your feedback has been submitted successfully.")
                                st.info("💡 Your input will help us identify patterns in misclassifications and improve the model.")
                                
                                if user_name:
                                    st.balloons()
                                    st.markdown(f"🙏 **Special thanks to {user_name}** for contributing to model improvement!")
                            else:
                                st.error("⚠️ There was an error saving your feedback. Please try again later.")
                                
                        except Exception as e:
                            st.error("⚠️ There was an error saving your feedback. Please try again later.")
                            st.caption(f"Error details: {str(e)}")
                            
            elif user_feedback == "✅ Yes, this prediction is accurate":
                st.success("🎉 Great! We're glad our model got it right.")
                st.info("Your confirmation helps us understand when the model is performing well.")
                
            elif user_feedback == "🤷 I'm not sure / It's complicated":
                st.info("That's completely understandable! Personality is complex and can vary by situation.")
                st.markdown("""
                **Remember:** This is a simplified binary classification. In reality:
                - Most people have both introverted and extroverted traits
                - Personality can vary by context (work vs. social vs. family)
                - This tool is for educational purposes, not professional assessment
                """)
    # --- End Modal Dialog ---
    
    # Footer
    st.markdown("---")
    
    # Know More Section
    st.header("🔍 Know More About This App", anchor="know-more-section")
    
    # Create tabs for different information sections
    tab_about, tab_dataset, tab_tech = st.tabs(["📖 About", "📊 Dataset & Competition", "⚙️ Technical Details"])
    
    with tab_about:
        st.markdown("""
        ### 🧠 What This App Does
        
        This **AI Personality Predictor** uses advanced machine learning to predict whether you're an **Introvert** or **Extrovert** based on your behavioral characteristics and preferences. 
        
        **Key Features:**
        - **Real-time Prediction**: Get instant personality predictions as you adjust the sliders
        - **Confidence Analysis**: See how confident the model is in its prediction
        - **Reliability Assessment**: Discover how close your characteristics are to samples the model struggles with
        - **Detailed Comparison**: Compare your profile with the nearest misclassified sample
        - **Performance Statistics**: View model accuracy for your predicted personality type
        - **Community Map**: Join our interactive PCA visualization and connect with others
        - **Networking**: Optional LinkedIn integration for professional networking
        
        ### 🎯 How It Works
        
        1. **Input Your Characteristics**: Use the sliders and dropdowns to describe yourself
        2. **Get Prediction**: Click the predict button to see your personality type
        3. **Analyze Results**: Review confidence scores, reliability metrics, and detailed comparisons
        4. **Understand Risk**: See if you're in a "safe zone" or if the prediction might be uncertain
        5. **Join Community**: Add yourself to our interactive personality map and connect with others
        
        The app doesn't just give you a prediction—it helps you understand **why** and **how reliable** that prediction is, plus connects you with a community of like-minded individuals!
        """)
        
    
    with tab_dataset:
        st.markdown("""
        ### 📊 Dataset & Kaggle Competition
        
        This project is built on top of the dataset from **Kaggle's Playground Series Season 5, Episode 7 (S5E7)** competition, which focused on personality prediction.
        
        #### 🏆 Competition Highlights
        - **Competition**: Kaggle Playground Series S5E7
        - **Task**: Binary classification (Introvert vs Extrovert)
        - **Dataset Size**: Thousands of training samples
        - **Features**: 7 behavioral and social characteristics
        
        #### 🎲 Fun Facts
        - **97.5% Accuracy**: Even a Random Forest with mostly default parameters achieves 97.5% accuracy on the test dataset!
        - **The Missing 2.5%**: There are only **30 samples** in the test dataset, which means just 1 wrong prediction costs 3.3% accuracy. This small test size explains why models struggle to achieve perfect 100% accuracy.
        - **High-Quality Data**: The dataset is remarkably clean and well-structured, making it perfect for learning ML concepts.
        
        #### 📈 Dataset Features
        1. **Time_spent_Alone** - Hours spent alone daily
        2. **Stage_fear** - Fear of public speaking (Yes/No)
        3. **Social_event_attendance** - Frequency of attending social events (0-10)
        4. **Going_outside** - Days per week going outside (0-7)
        5. **Drained_after_socializing** - Feeling drained after socializing (Yes/No)
        6. **Friends_circle_size** - Number of close friends
        7. **Post_frequency** - Social media posting frequency (0-10)
        """)
    
    with tab_tech:
        st.markdown("""
        ### ⚙️ Technical Implementation
        
        #### 🤖 Machine Learning Pipeline
        - **Algorithm**: **XGBoost Classifier** - A powerful gradient boosting framework
        - **Preprocessing**: None
        - **Validation**: Cross-validation and train-test split for robust evaluation
        - **Hyperparameter Tuning**: Minimal (Self-tuning) for example: Depth=3.

        #### 🛠️ App Architecture
        - **Frontend**: **Streamlit** - Interactive web application
        - **Backend**: **Python** with scikit-learn, pandas, numpy
        - **Visualization**: **Plotly** for interactive charts and gauges
        - **Data Processing**: Real-time encoding and prediction
        
        #### 🔍 Advanced Features
        - **Distance Analysis**: Calculates Euclidean distance to misclassified samples
        - **Reliability Scoring**: Compares your distance to baseline correct predictions
        - **Risk Assessment**: Identifies if you're in a high-risk prediction zone
        - **Feature Comparison**: Shows exactly how you differ from problematic samples
        - **Community PCA Map**: Interactive 2D visualization of user submissions using Principal Component Analysis
        - **Networking Integration**: Optional LinkedIn profile integration for community building
        
        #### 📊 Performance Metrics
        - **Overall Accuracy**: ~96%+ on training data
        - **Class Balance**: Handles both personality types effectively
        - **Confidence Calibration**: Provides meaningful confidence scores
        - **Baseline Distance Metrics**: Scientifically calculated using weighted nearest neighbor analysis
        """)
        
    
    # Separate Developer Section
    st.markdown("---")
    st.header("👨‍💻 About the Developer - Komil K. Parmar", anchor="developer-section")
    
    # Introduction with visual appeal
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                padding: 1.5rem; border-radius: 10px; margin: 1rem 0; text-align: center;">
        <h2 style="color: white; margin: 0;">👋 Hello! I'm Komil Parmar</h2>
        <p style="color: white; margin: 0.5rem 0; font-size: 1.1rem;">
            🎂 20-year-old machine learning enthusiast from Vadodara, Gujarat
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Journey timeline
    st.subheader("🚀 My Journey in Computer Science")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - 🖌️ **Class 8:** Started with **3D Modeling**
        - 🛡️ **Class 9:** Explored **Ethical Hacking**
        - 🤖 **Class 10:** Found my passion in **Machine Learning**
        """)
    
    with col2:
        st.markdown("""
        - 📚 Completed my first course in **AI** alongside school
        - 🏆 Completed 2 more ML courses during vacation
        - 🎓 Cleared the **TensorFlow Developer Certification Exam**
        """)

    # Academic Journey
    st.subheader("📚 Academic Journey")
    st.markdown("""
    - 📘 Studied ML alongside high school
    - 📚 Faced challenges balancing school subjects and ML
    - 😫 Frustrated with unrelated subjects for JEE preparation
    - 🏫 Researched college curriculums but found them disappointing
    - 🚫 Decided to skip traditional college for self-learning
    """)

    # Self-learning choice
    st.subheader("🎯 Why I Chose Self-Learning")
    st.markdown("""
    - 💡 Believe I can learn faster independently
    - ❌ Avoid unrelated subjects like Chemistry, Biology, Graphics, Animations, Physics, Circuits, and workshops
    - 📜 Focus on gaining real knowledge rather than just a degree
    """)

    # Connection seeking
    st.subheader("🤝 Looking for Connections")
    st.markdown("""
    - 🌟 Eager to connect with like-minded individuals
    - 🫱🏻‍🫲🏻 Grateful for friends and mentors
    - 🌱 Let's learn and grow together!
    """)
                
    # Detailed story
    st.subheader("📖 My Detailed Journey")
    
    st.markdown("""
    Hello! I'm Komil Parmar, a 20-year-old machine-learning enthusiast from Vadodara, Gujarat. 
    My journey in computer science began in class 8, where I got into 3D modeling at first, switched to 
    Ethical hacking in class 9, and finally found my passion in machine learning in class 10. That year, 
    I completed my first course in AI alongside my school and then completed 2 more in ML during that 
    vacation by the end of which I also cleared the TensorFlow Developer Certification Exam. Despite the 
    academic interruptions during high school, I remained committed to my interests.
    """)
    
    st.markdown("---")
    
    st.markdown("""
    Studying machine learning alongside my high school studies was challenging. By the end of class 12, 
    I was so frustrated having to learn unrelated subjects not just out of curiosity but to the extent of 
    clearing JEE (which is known to be one of the toughest exams) that I was eagerly waiting for the 
    vacations so that I can continue my journey of self-learning. But as soon as the vacations arrived, 
    I was again stuck with exams like JEE, JEE Advance, and then college selection. I researched the 
    curriculums and syllabuses of colleges that were providing a Bachelor's degree in ML and I never 
    thought college selection would be that disappointing.
    """)
    
    st.markdown("---")
    
    st.markdown("""
    Though I'm not a professional and might not be aware of even some of the most common concepts in ML, 
    I'm confident I'll catch up faster at home than most college students in India, because there (at college), 
    I'd again have to study unrelated subjects like Chemistry, Biology, Graphics, Animations, Physics, 
    Circuits, workshops and what not, just for getting a piece of paper with a precious stamp on it. 
    Hence, I decided to skip the traditional way of learning.
    """)
    
    st.markdown("---")
    
    st.markdown("""
    Although the decision was very straightforward, skipping traditional offline college is still the 
    hardest decision I have ever made so far since it is not just about studies, but also about having 
    the experience, meeting people, making friends, etc. I hope I will somehow miraculously get in touch 
    with people who can help me compensate for these sacrifices. Hence, I am not only eager to connect 
    with like-minded individuals and make new friends but in fact, I would be very much grateful to those 
    who can connect with me, whether as a friend or a mentor. Let's learn and grow together!
    """)
    
    # Call to action
    st.markdown("""
    <div style="background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%); 
                padding: 1rem; border-radius: 10px; margin: 1.5rem 0; text-align: center;">
        <p style="color: white; margin: 0; font-size: 1.1rem;">
            💌 Feel free to connect with me and let's build something amazing together!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Social Media Links Section
    st.markdown("""
    <!-- Connect with me -->
    <div id="user-content-toc" style="margin: 2rem 0;">
      <div style="text-align: center;">
        <h2 style="display: inline-block; margin-bottom: 1rem; color: #1f77b4;">Connect With Me 🤝</h2>
      </div>
    </div>

    <!-- Icons and links -->
    <div style="text-align: center; margin: 1.5rem 0;">
        <a href="https://www.linkedin.com/in/komil-parmar-488967243/" target="_blank" style="margin: 0 10px;">
            <img src="https://user-images.githubusercontent.com/88904952/234979284-68c11d7f-1acc-4f0c-ac78-044e1037d7b0.png" 
                 alt="LinkedIn" height="50" width="50" style="transition: transform 0.3s ease;" 
                 onmouseover="this.style.transform='scale(1.1)'" 
                 onmouseout="this.style.transform='scale(1)'"/>
        </a>
        <a href="https://github.com/Komil-parmar/Komil-parmar" target="_blank" style="margin: 0 10px;">
            <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" 
                 alt="GitHub" height="50" width="50" style="transition: transform 0.3s ease;" 
                 onmouseover="this.style.transform='scale(1.1)'" 
                 onmouseout="this.style.transform='scale(1)'"/>
        </a>
        <a href="https://discordapp.com/users/opgamer01540" target="_blank" style="margin: 0 10px;">
            <img src="https://user-images.githubusercontent.com/88904952/234982627-019fd336-6248-453c-9b05-97c13fd1d207.png" 
                 alt="Discord" height="50" width="50" style="transition: transform 0.3s ease;" 
                 onmouseover="this.style.transform='scale(1.1)'" 
                 onmouseout="this.style.transform='scale(1)'"/>
        </a>
    </div>

    <!-- Email section -->
    <div style="text-align: center; margin: 1.5rem 0;">
        <a href="mailto:komilparmar57@gmail.com" style="text-decoration: none;">
            <img src="https://upload.wikimedia.org/wikipedia/commons/4/4e/Gmail_Icon.png" 
                 alt="Gmail" width="50" height="50" style="transition: transform 0.3s ease;" 
                 onmouseover="this.style.transform='scale(1.1)'" 
                 onmouseout="this.style.transform='scale(1)'"/>
        </a>
        <p style="margin: 0.5rem 0 0 0; color: #666; font-style: italic;">
            Email is always preferred.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer note
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🧠 <strong>AI Personality Predictor</strong> | Built with Streamlit & XGBoost | Kaggle S5E7 Dataset</p>
        <p><strong>Developed by Komil K. Parmar</strong> | <em>This is for educational purposes. Results should not be considered as professional psychological assessment.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
