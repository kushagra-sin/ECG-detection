
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import tensorflow as tf
from typing import List, Dict
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ECG Classification API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
MODEL = None
LABEL_CLASSES = None
MODEL_PATH = "ecg_classifier_300hz.h5"
LABEL_PATH = "label_classes.npy"

# ============= REQUEST/RESPONSE MODELS =============
class ECGSignalRequest(BaseModel):
    signal: List[float] = Field(..., description="ECG signal samples (normalized)")
    sample_rate: int = Field(default=300, description="Sampling rate in Hz")
    duration: float = Field(default=10, description="Duration in seconds")
    source: str = Field(default="unknown", description="Data source identifier")

class PredictionResult(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]

class ECGResponse(BaseModel):
    result: PredictionResult
    input_info: Dict
    debug_info: Dict = None

# ============= MODEL LOADING =============
def load_model_and_labels():
    """Load the trained model and label classes"""
    global MODEL, LABEL_CLASSES
    
    try:
        # Try loading .h5 model first
        if os.path.exists(MODEL_PATH):
            logger.info(f"Loading model from {MODEL_PATH}")
            MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)
            MODEL.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            logger.info("Model loaded successfully (H5 format)")
        elif os.path.exists("ecg_classifier_300hz.tflite"):
            # Fallback to TFLite
            logger.info("Loading TFLite model")
            interpreter = tf.lite.Interpreter(model_path="ecg_classifier_300hz.tflite")
            interpreter.allocate_tensors()
            MODEL = interpreter
            logger.info("TFLite model loaded successfully")
        else:
            raise FileNotFoundError("No model file found (.h5 or .tflite)")
        
        # Load label classes
        if os.path.exists(LABEL_PATH):
            LABEL_CLASSES = np.load(LABEL_PATH, allow_pickle=True)
            logger.info(f"Loaded label classes: {LABEL_CLASSES}")
        else:
            # Default labels if file not found
            LABEL_CLASSES = np.array(['Normal', 'Abnormal'])
            logger.warning("Label file not found, using default labels")
            
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

# ============= LABEL REVERSAL LOGIC =============
def reverse_labels(prediction: str, probabilities: Dict[str, float]) -> tuple:
    """
    
    Args:
        prediction: Original prediction from model
        probabilities: Original probability dictionary
        
    Returns:
        Tuple of (corrected_prediction, corrected_probabilities)
    """
    logger.info(f"[LABEL REVERSAL] Original prediction: {prediction}")
    logger.info(f"[LABEL REVERSAL] Original probabilities: {probabilities}")
    
    # Create copy of probabilities
    corrected_probs = probabilities.copy()
    
    # Swap the prediction label
    if prediction == 'Normal':
        corrected_prediction = 'Abnormal'
    elif prediction == 'Abnormal':
        corrected_prediction = 'Normal'
    else:
        # For other labels (like 'Myocardial Infarction'), keep as is
        corrected_prediction = prediction
    
    # Swap probability values for Normal and Abnormal
    if 'Normal' in corrected_probs and 'Abnormal' in corrected_probs:
        normal_prob = corrected_probs['Normal']
        abnormal_prob = corrected_probs['Abnormal']
        corrected_probs['Normal'] = abnormal_prob
        corrected_probs['Abnormal'] = normal_prob
    
    logger.info(f"[LABEL REVERSAL] Corrected prediction: {corrected_prediction}")
    logger.info(f"[LABEL REVERSAL] Corrected probabilities: {corrected_probs}")
    
    return corrected_prediction, corrected_probs

# ============= PREDICTION LOGIC =============
def preprocess_signal(signal: List[float], target_length: int = 3000) -> np.ndarray:
    """Preprocess ECG signal for model input"""
    signal_array = np.array(signal, dtype=np.float32)
    
    # Ensure correct length
    if len(signal_array) < target_length:
        # Pad with zeros
        padding = target_length - len(signal_array)
        signal_array = np.pad(signal_array, (0, padding), mode='constant')
    elif len(signal_array) > target_length:
        # Truncate
        signal_array = signal_array[:target_length]
    
    # Reshape for model: (1, timesteps, 1)
    signal_array = signal_array.reshape(1, target_length, 1)
    
    return signal_array

def predict_ecg(signal: List[float]) -> tuple:
    """
    Make prediction on ECG signal
    
    Returns:
        Tuple of (prediction_label, confidence, probabilities_dict)
    """
    try:
        # Preprocess
        processed_signal = preprocess_signal(signal)
        
        # Make prediction
        if isinstance(MODEL, tf.lite.Interpreter):
            # TFLite prediction
            input_details = MODEL.get_input_details()
            output_details = MODEL.get_output_details()
            MODEL.set_tensor(input_details[0]['index'], processed_signal)
            MODEL.invoke()
            predictions = MODEL.get_tensor(output_details[0]['index'])[0]
        else:
            # Keras model prediction
            predictions = MODEL.predict(processed_signal, verbose=0)[0]
        
        # Get prediction details
        predicted_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_idx])
        predicted_label = str(LABEL_CLASSES[predicted_idx])
        
        # Create probabilities dictionary
        probabilities = {}
        for idx, label in enumerate(LABEL_CLASSES):
            probabilities[str(label)] = float(predictions[idx])
        
        logger.info(f"Raw model output - Label: {predicted_label}, Confidence: {confidence:.4f}")
        
        return predicted_label, confidence, probabilities
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise

# ============= API ENDPOINTS =============
@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting ECG Classification API...")
    load_model_and_labels()
    logger.info("API ready!")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "ECG Classification API",
        "version": "1.0.0",
        "model_loaded": MODEL is not None,
        "label_reversal": "ENABLED - Normal/Abnormal swapped"
    }

@app.post("/predict_signal", response_model=ECGResponse)
async def predict_signal(request: ECGSignalRequest):
    """
    Main prediction endpoint
    """
    try:
        logger.info(f"Received prediction request from source: {request.source}")
        logger.info(f"Signal length: {len(request.signal)}, Sample rate: {request.sample_rate}")
        
        # Validate input
        if len(request.signal) == 0:
            raise HTTPException(status_code=400, detail="Empty signal received")
        
        if len(request.signal) < 100:
            raise HTTPException(status_code=400, detail="Signal too short (minimum 100 samples)")
        
        # Store original signal stats for debugging
        signal_array = np.array(request.signal)
        original_stats = {
            'original_mean': float(np.mean(signal_array)),
            'original_std': float(np.std(signal_array)),
            'original_min': float(np.min(signal_array)),
            'original_max': float(np.max(signal_array))
        }
        
        # Make prediction (this returns RAW model output)
        raw_prediction, raw_confidence, raw_probabilities = predict_ecg(request.signal)
        
        # ===== APPLY LABEL REVERSAL =====
        corrected_prediction, corrected_probabilities = reverse_labels(
            raw_prediction, 
            raw_probabilities
        )
        
        # Update confidence based on corrected prediction
        corrected_confidence = corrected_probabilities[corrected_prediction]
        
        # Prepare response
        response = ECGResponse(
            result=PredictionResult(
                prediction=corrected_prediction,
                confidence=corrected_confidence,
                probabilities=corrected_probabilities
            ),
            input_info={
                'received_samples': len(request.signal),
                'sample_rate': request.sample_rate,
                'duration': request.duration,
                'source': request.source
            },
            debug_info={
                'input_stats': original_stats,
                'raw_model_output': {
                    'prediction': raw_prediction,
                    'confidence': float(raw_confidence)
                },
                'label_reversal_applied': True
            }
        )
        
        logger.info(f"✓ Prediction complete: {corrected_prediction} ({corrected_confidence*100:.2f}%)")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "labels": LABEL_CLASSES.tolist() if LABEL_CLASSES is not None else None,
        "model_type": "H5" if not isinstance(MODEL, tf.lite.Interpreter) else "TFLite"
    }

# ============= MAIN =============
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
