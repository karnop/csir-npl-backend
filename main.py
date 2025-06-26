from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import io
from typing import Dict
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Class Medical Image Classification API",
    description="API for classifying dental/medical images into 6 categories using DenseNet",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None
MODEL_PATH = "models/densenet_model.h5"

# Class definitions matching your new model
CLASS_LABELS = [
    "dyed-resection-margins",
    "esophagitis", 
    "normal-pylorus",
    "normal-z-line",
    "polyps",
    "ulcerative-colitis"
]

# Class descriptions with medical information
CLASS_DESCRIPTIONS = {
    "dyed-resection-margins": "Cancer recurrence risk: 10–15%. Requires histological assessment.",
    "esophagitis": "Inflammation of the esophagus. Cancer risk is typically low unless chronic.",
    "normal-pylorus": "Normal tissue near stomach exit (pylorus). No immediate concern.",
    "normal-z-line": "Normal Z-line at gastroesophageal junction. No pathology detected.",
    "polyps": "Polyps have a cancer risk of 5–10%, depending on type and dysplasia.",
    "ulcerative-colitis": "Chronic inflammation. Long-term UC may increase cancer risk significantly (>15%)."
}

# Risk levels for each class
RISK_LEVELS = {
    "dyed-resection-margins": "high",
    "esophagitis": "low",
    "normal-pylorus": "none",
    "normal-z-line": "none", 
    "polyps": "moderate",
    "ulcerative-colitis": "high"
}

def load_trained_model():
    """Load the trained model"""
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
        model = load_model(MODEL_PATH)
        logger.info("6-class DenseNet model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess the input image for model prediction
    
    Args:
        image: PIL Image object
    
    Returns:
        Preprocessed image array ready for model input
    """
    try:
        # Resize image to model input size
        image = image.resize((224, 224))
        
        # Convert to RGB if not already (handles RGBA, grayscale, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array and normalize to [0,1]
        img_array = np.array(image) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    success = load_trained_model()
    if not success:
        logger.error("Failed to load model on startup")
        # You might want to exit the application here in production
        # import sys
        # sys.exit(1)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Multi-Class Medical Image Classification API is running",
        "model_loaded": model is not None,
        "version": "2.0.0",
        "classes": len(CLASS_LABELS),
        "class_names": CLASS_LABELS
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "supported_classes": CLASS_LABELS
    }

@app.get("/classes")
async def get_classes():
    """Get information about all supported classes"""
    return {
        "classes": [
            {
                "id": i,
                "name": class_name,
                "description": CLASS_DESCRIPTIONS[class_name],
                "risk_level": RISK_LEVELS[class_name]
            }
            for i, class_name in enumerate(CLASS_LABELS)
        ]
    }

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)) -> Dict:
    """
    Predict the class of an uploaded medical image
    
    Args:
        file: Uploaded image file
    
    Returns:
        Dictionary containing prediction results
    """
    # Check if model is loaded
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server logs."
        )
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="File must be an image"
        )
    
    try:
        # Read image file
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_index = int(np.argmax(predictions[0]))
        predicted_class = CLASS_LABELS[predicted_index]
        confidence = float(predictions[0][predicted_index])
        
        # Get all class probabilities
        class_probabilities = {
            CLASS_LABELS[i]: round(float(predictions[0][i]), 4)
            for i in range(len(CLASS_LABELS))
        }
        
        return {
            "filename": file.filename,
            "prediction": {
                "class": predicted_class,
                "class_id": predicted_index,
                "confidence": round(confidence, 4),
                "description": CLASS_DESCRIPTIONS[predicted_class],
                "risk_level": RISK_LEVELS[predicted_class]
            },
            "all_probabilities": class_probabilities,
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing image: {str(e)}"
        )

@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)) -> Dict:
    """
    Predict multiple images at once
    
    Args:
        files: List of uploaded image files
    
    Returns:
        Dictionary containing batch prediction results
    """
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server logs."
        )
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(
            status_code=400, 
            detail="Maximum 10 files allowed per batch"
        )
    
    results = []
    
    for file in files:
        try:
            # Validate file type
            if not file.content_type.startswith('image/'):
                results.append({
                    "filename": file.filename,
                    "error": "File must be an image",
                    "status": "failed"
                })
                continue
            
            # Process image
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            processed_image = preprocess_image(image)
            
            # Make prediction
            predictions = model.predict(processed_image)
            predicted_index = int(np.argmax(predictions[0]))
            predicted_class = CLASS_LABELS[predicted_index]
            confidence = float(predictions[0][predicted_index])
            
            # Get all class probabilities
            class_probabilities = {
                CLASS_LABELS[i]: round(float(predictions[0][i]), 4)
                for i in range(len(CLASS_LABELS))
            }
            
            results.append({
                "filename": file.filename,
                "prediction": {
                    "class": predicted_class,
                    "class_id": predicted_index,
                    "confidence": round(confidence, 4),
                    "description": CLASS_DESCRIPTIONS[predicted_class],
                    "risk_level": RISK_LEVELS[predicted_class]
                },
                "all_probabilities": class_probabilities,
                "status": "success"
            })
            
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {str(e)}")
            results.append({
                "filename": file.filename,
                "error": str(e),
                "status": "failed"
            })
    
    return {
        "results": results,
        "total_files": len(files),
        "successful_predictions": len([r for r in results if r["status"] == "success"]),
        "failed_predictions": len([r for r in results if r["status"] == "failed"])
    }

@app.post("/predict/detailed")
async def predict_detailed(file: UploadFile = File(...)) -> Dict:
    """
    Get detailed prediction with top N classes and medical recommendations
    
    Args:
        file: Uploaded image file
    
    Returns:
        Dictionary containing detailed prediction results
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)[0]
        
        # Get top 3 predictions
        top_indices = np.argsort(predictions)[::-1][:3]
        top_predictions = [
            {
                "class": CLASS_LABELS[idx],
                "class_id": int(idx),
                "probability": round(float(predictions[idx]), 4),
                "description": CLASS_DESCRIPTIONS[CLASS_LABELS[idx]],
                "risk_level": RISK_LEVELS[CLASS_LABELS[idx]]
            }
            for idx in top_indices
        ]
        
        # Primary prediction
        primary = top_predictions[0]
        
        # Medical recommendations based on risk level
        recommendations = {
            "high": "Immediate medical consultation recommended. Follow up with specialist.",
            "moderate": "Medical consultation advised. Monitor for changes.",
            "low": "Routine monitoring recommended. Consult if symptoms persist.",
            "none": "No immediate medical intervention required. Continue regular check-ups."
        }
        
        return {
            "filename": file.filename,
            "primary_prediction": primary,
            "top_3_predictions": top_predictions,
            "medical_recommendation": recommendations[primary["risk_level"]],
            "all_class_probabilities": {
                CLASS_LABELS[i]: round(float(predictions[i]), 4)
                for i in range(len(CLASS_LABELS))
            },
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error during detailed prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded"
        )
    
    try:
        return {
            "model_summary": {
                "input_shape": str(model.input_shape),
                "output_shape": str(model.output_shape),
                "total_params": model.count_params(),
                "trainable_params": sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]),
                "non_trainable_params": sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
            },
            "classes": {
                str(i): {
                    "name": class_name,
                    "description": CLASS_DESCRIPTIONS[class_name],
                    "risk_level": RISK_LEVELS[class_name]
                }
                for i, class_name in enumerate(CLASS_LABELS)
            },
            "model_type": "6-class categorical classification",
            "architecture": "DenseNet"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error getting model info: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)










