"""
ECG Classification API with Label Reversal
Uses TFLite Runtime (no full TensorFlow dependency)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
from typing import List, Dict
import logging
import os

# TFLite Runtime import
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    # Fallback for local testing with full TensorFlow
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

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
MODEL_PATH = "ecg_classifier_300hz.tflite"
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
    """Load the TFLite model and label classes"""
    global MODEL, LABEL_CLASSES
    
    try:
        logger.info("="*70)
        logger.info("LOADING ECG CLASSIFICATION MODEL")
        logger.info("="*70)
        
        # Check current directory
        logger.info(f"Current directory: {os.getcwd()}")
        logger.info(f"Files in directory: {os.listdir('.')}")
        
        # Load TFLite model
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"TFLite model not found: {MODEL_PATH}\n"
                f"Please ensure 'ecg_classifier_300hz.tflite' is in the repository root."
            )
        
        logger.info(f"Loading TFLite model from: {MODEL_PATH}")
        MODEL = Interpreter(model_path=MODEL_PATH)
        MODEL.allocate_tensors()
        
        # Get model details
        input_details = MODEL.get_input_details()
        output_details = MODEL.get_output_details()
        
        logger.info(f"✓ TFLite model loaded successfully!")
        logger.info(f"  Input shape: {input_details[0]['shape']}")
        logger.info(f"  Input dtype: {input_details[0]['dtype']}")
        logger.info(f"  Output shape: {output_details[0]['shape']}")
        
        # Load label classes
        if os.path.exists(LABEL_PATH):
            LABEL_CLASSES = np.load(LABEL_PATH, allow_pickle=True)
            logger.info(f"✓ Loaded label classes: {LABEL_CLASSES.tolist()}")
        else:
            # Default labels if file not found
            LABEL_CLASSES = np.array(['Normal', 'Abnormal'])
            logger.warning(f"⚠️  Label file not found, using default: {LABEL_CLASSES.tolist()}")
        
        logger.info("="*70)
        logger.info("MODEL READY - Label Reversal ENABLED")
        logger.info("="*70)
            
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
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
        logger.info(f"Signal padded from {len(signal)} to {target_length}")
    elif len(signal_array) > target_length:
        # Truncate
        signal_array = signal_array[:target_length]
        logger.info(f"Signal truncated from {len(signal)} to {target_length}")
    
    # Reshape for model: (1, timesteps, 1)
    signal_array = signal_array.reshape(1, target_length, 1)
    
    return signal_array

def predict_ecg(signal: List[float]) -> tuple:
    """
    Make prediction on ECG signal using TFLite model
    
    Returns:
        Tuple of (prediction_label, confidence, probabilities_dict)
    """
    try:
        # Preprocess signal
        processed_signal = preprocess_signal(signal)
        
        # Get input and output details
        input_details = MODEL.get_input_details()
        output_details = MODEL.get_output_details()
        
        # Set input tensor
        MODEL.set_tensor(input_details[0]['index'], processed_signal)
        
        # Run inference
        MODEL.invoke()
        
        # Get output tensor
        predictions = MODEL.get_tensor(output_details[0]['index'])[0]
        
        # Get prediction details
        predicted_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_idx])
        predicted_label = str(LABEL_CLASSES[predicted_idx])
        
        # Create probabilities dictionary
        probabilities = {}
        for idx, label in enumerate(LABEL_CLASSES):
            probabilities[str(label)] = float(predictions[idx])
        
        logger.info(f"Raw model output - Label: {predicted_label}, Confidence: {confidence:.4f}")
        logger.info(f"Raw probabilities: {probabilities}")
        
        return predicted_label, confidence, probabilities
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        raise

# ============= API ENDPOINTS =============
@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("\n" + "="*70)
    logger.info("STARTING ECG CLASSIFICATION API")
    logger.info("="*70)
    load_model_and_labels()
    logger.info("✓ API READY TO ACCEPT REQUESTS")
    logger.info("="*70 + "\n")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "ECG Classification API",
        "version": "1.0.0",
        "model_loaded": MODEL is not None,
        "model_type": "TFLite",
        "label_reversal": "ENABLED - Normal/Abnormal swapped",
        "labels": LABEL_CLASSES.tolist() if LABEL_CLASSES is not None else None
    }

@app.post("/predict_signal", response_model=ECGResponse)
async def predict_signal(request: ECGSignalRequest):
    """
    Main prediction endpoint - Accepts ECG signal and returns classification with REVERSED labels
    """
    try:
        logger.info("\n" + "="*70)
        logger.info(f"NEW PREDICTION REQUEST")
        logger.info("="*70)
        logger.info(f"Source: {request.source}")
        logger.info(f"Signal length: {len(request.signal)} samples")
        logger.info(f"Sample rate: {request.sample_rate} Hz")
        logger.info(f"Duration: {request.duration}s")
        
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
        
        logger.info(f"Signal stats: mean={original_stats['original_mean']:.4f}, "
                   f"std={original_stats['original_std']:.4f}, "
                   f"range=[{original_stats['original_min']:.4f}, {original_stats['original_max']:.4f}]")
        
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
        
        logger.info("="*70)
        logger.info(f"✓ PREDICTION COMPLETE: {corrected_prediction} ({corrected_confidence*100:.2f}%)")
        logger.info("="*70 + "\n")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error processing request: {e}")
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
        "model_type": "TFLite",
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "label_reversal": "enabled"
    }

@app.get("/model-info")
async def model_info():
    """Get detailed model information"""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        input_details = MODEL.get_input_details()
        output_details = MODEL.get_output_details()
        
        return {
            "model_type": "TFLite",
            "model_path": MODEL_PATH,
            "input_shape": input_details[0]['shape'].tolist(),
            "input_dtype": str(input_details[0]['dtype']),
            "output_shape": output_details[0]['shape'].tolist(),
            "output_dtype": str(output_details[0]['dtype']),
            "labels": LABEL_CLASSES.tolist() if LABEL_CLASSES is not None else None,
            "num_classes": len(LABEL_CLASSES) if LABEL_CLASSES is not None else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

# ============= MAIN =============
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
