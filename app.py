"""Streamlit app for MEM Shirt Size Predictor"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import joblib
import streamlit as st

# Add app directory to path for imports
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "app"))

MODEL_PATH = REPO_ROOT / "app" / "model.joblib"


@st.cache_resource
def load_model() -> Dict[str, Any]:
    """Load the trained model artifact"""
    if not MODEL_PATH.exists():
        st.error(f"Model not found at {MODEL_PATH}. Please train the model first.")
        st.stop()
    
    artifact = joblib.load(MODEL_PATH)
    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        st.error("Invalid model artifact format.")
        st.stop()
    
    return artifact


def main():
    # Page configuration
    st.set_page_config(
        page_title="MEM Shirt Size Predictor",
        page_icon="👕",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    # Load model
    artifact = load_model()
    pipeline = artifact["pipeline"]
    
    # Header
    st.title("👕 MEM Shirt Size Predictor")
    st.markdown("Get personalized shirt size recommendations based on your measurements")
    
    # Sidebar - Model Info
    with st.sidebar:
        st.header("ℹ️ Model Information")
        st.write(f"**Version:** {artifact.get('version', 'unknown')}")
        st.write(f"**Trained:** {artifact.get('trained_at', 'N/A')}")
        
        if "metrics" in artifact and artifact["metrics"]:
            st.subheader("Model Metrics")
            metrics = artifact["metrics"]
            if "accuracy" in metrics:
                st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
            if "f1_score" in metrics:
                st.metric("F1 Score", f"{metrics['f1_score']:.3f}")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("This AI model predicts your ideal shirt size based on your body measurements and preferences.")
    
    # Main form
    st.header("Enter Your Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        height_cm = st.number_input(
            "Height (cm)",
            min_value=120.0,
            max_value=230.0,
            value=170.0,
            step=1.0,
            help="Enter your height in centimeters"
        )
        
        weight_kg = st.number_input(
            "Weight (kg)",
            min_value=30.0,
            max_value=250.0,
            value=70.0,
            step=0.5,
            help="Enter your weight in kilograms"
        )
        
        age = st.number_input(
            "Age",
            min_value=10,
            max_value=100,
            value=25,
            step=1,
            help="Enter your age"
        )
    
    with col2:
        gender = st.selectbox(
            "Gender",
            options=["male", "female", "other"],
            help="Select your gender"
        )
        
        fit_preference = st.selectbox(
            "Fit Preference",
            options=["regular", "slim", "oversized"],
            help="How do you prefer your shirts to fit?"
        )
        
        build = st.selectbox(
            "Body Build",
            options=["average", "lean", "athletic", "curvy"],
            help="Select the option that best describes your body build"
        )
    
    st.markdown("---")
    
    # Prediction button
    if st.button("🔮 Predict My Size", type="primary", use_container_width=True):
        # Prepare input
        input_data = {
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "age": age,
            "gender": gender,
            "fit_preference": fit_preference,
            "build": build
        }
        
        try:
            # Make prediction
            with st.spinner("Analyzing your measurements..."):
                proba = pipeline.predict_proba([input_data])[0]
                labels = list(pipeline.classes_)
            
            # Process results
            probs = {}
            for label, p in zip(labels, proba):
                if label in ("S", "M", "L", "XL", "XXL"):
                    probs[label] = float(p)
            
            # Get recommendation
            recommended = max(probs.items(), key=lambda kv: kv[1])[0]
            confidence = float(probs[recommended])
            
            # Display results
            st.success("✅ Prediction Complete!")
            
            st.markdown("### 🎯 Your Recommended Size")
            st.markdown(f"# **{recommended}**")
            st.progress(confidence)
            st.caption(f"Confidence: {confidence:.1%}")
            
            # Show all probabilities
            st.markdown("### 📊 Size Probabilities")
            
            # Sort sizes in standard order
            size_order = ["S", "M", "L", "XL", "XXL"]
            sorted_probs = {size: probs.get(size, 0.0) for size in size_order if size in probs}
            
            for size, prob in sorted_probs.items():
                col_label, col_bar = st.columns([1, 4])
                with col_label:
                    st.write(f"**{size}**")
                with col_bar:
                    st.progress(prob)
                    st.caption(f"{prob:.1%}")
            
            # Additional insights
            st.markdown("### 💡 Insights")
            
            if confidence >= 0.7:
                st.info("🎯 High confidence prediction - this size should fit you well!")
            elif confidence >= 0.5:
                st.info("📍 Moderate confidence - you might also consider nearby sizes.")
            else:
                st.warning("⚠️ Lower confidence - consider trying on multiple sizes if possible.")
            
            # Show alternative sizes
            sorted_sizes = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_sizes) > 1 and sorted_sizes[1][1] > 0.2:
                st.info(f"💭 Alternative: Size **{sorted_sizes[1][0]}** ({sorted_sizes[1][1]:.1%} probability)")
        
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            st.exception(e)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.9em;'>
        MEM Shirt Size Predictor | Powered by Machine Learning
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
