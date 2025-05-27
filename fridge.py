from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
from ultralytics import YOLO
import io
from PIL import Image
import logging
from typing import List, Dict, Any
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Food Detection API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Food items we want to detect (matching your frontend list)
TARGET_FOODS = {
    'yogurt': ['yogurt', 'yoghurt'],
    'spinach': ['spinach', 'leafy greens'],
    'tomato': ['tomato', 'tomatoes'],
    'broccoli': ['broccoli', 'brocoli'],
    'lemon': ['lemon', 'lime', 'citrus'],
    'green chilli': ['chili', 'chilli', 'pepper', 'green pepper'],
    'milk': ['milk', 'dairy']
}

# Food categories
FOOD_CATEGORIES = {
    'yogurt': 'Dairy',
    'spinach': 'Vegetables',
    'tomato': 'Vegetables',
    'broccoli': 'Vegetables',
    'lemon': 'Fruits',
    'green chilli': 'Vegetables',
    'milk': 'Dairy'
}

class FoodDetector:
    def __init__(self):
        """Initialize the food detection model"""
        try:
            # Load YOLOv8 model (you can use yolov8n.pt, yolov8s.pt, yolov8m.pt, or yolov8l.pt)
            self.model = YOLO('yolov8n.pt')  # Using nano version for faster inference
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

    def detect_food_items(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect food items in the image
        
        Args:
            image: OpenCV image array
            
        Returns:
            List of detected food items with their details
        """
        try:
            # Run YOLO detection
            results = self.model(image)
            
            detected_items = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class name and confidence
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = self.model.names[class_id].lower()
                        
                        # Check if detected item matches our target foods
                        matched_food = self._match_food_item(class_name)
                        
                        if matched_food and confidence > 0.3:  # Minimum confidence threshold
                            detected_items.append({
                                'item': matched_food,
                                'count': 1,  # Basic count, can be enhanced
                                'confidence': confidence,
                                'category': FOOD_CATEGORIES.get(matched_food, 'Other'),
                                'detected_class': class_name
                            })
            
            # Group similar items and sum their counts
            grouped_items = self._group_similar_items(detected_items)
            
            logger.info(f"Detected {len(grouped_items)} food items")
            return grouped_items
            
        except Exception as e:
            logger.error(f"Error during food detection: {e}")
            raise

    def _match_food_item(self, detected_class: str) -> str:
        """
        Match detected class with our target food items
        
        Args:
            detected_class: Class name from YOLO detection
            
        Returns:
            Matched food item name or None
        """
        detected_class = detected_class.lower()
        
        for food_name, aliases in TARGET_FOODS.items():
            for alias in aliases:
                if alias in detected_class or detected_class in alias:
                    return food_name
        
        # Additional fuzzy matching for common food items
        food_mappings = {
            'apple': 'apple',  # Not in our target list
            'banana': None,  # Not in our target list
            'orange': 'lemon',  # Close enough to citrus
            'carrot': None,  # Not in our target list
            'bottle': 'milk',  # Might be milk bottle
            'bowl': None,  # Container, not food
        }
        
        return food_mappings.get(detected_class)

    def _group_similar_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Group similar detected items and sum their counts
        
        Args:
            items: List of detected items
            
        Returns:
            Grouped items with summed counts
        """
        grouped = {}
        
        for item in items:
            food_name = item['item']
            if food_name in grouped:
                grouped[food_name]['count'] += item['count']
                # Keep the highest confidence
                grouped[food_name]['confidence'] = max(
                    grouped[food_name]['confidence'], 
                    item['confidence']
                )
            else:
                grouped[food_name] = item.copy()
        
        return list(grouped.values())

# Initialize the detector
try:
    detector = FoodDetector()
except Exception as e:
    logger.error(f"Failed to initialize food detector: {e}")
    detector = None

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if detector is None:
        return {"status": "unhealthy", "message": "Model not loaded"}
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/process_image")
async def process_image(file: UploadFile = File(...)):
    """
    Process uploaded image and detect food items
    
    Args:
        file: Uploaded image file
        
    Returns:
        JSON response with detected food items
    """
    if detector is None:
        raise HTTPException(status_code=500, detail="Food detection model not available")
    
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process the image
        contents = await file.read()
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to OpenCV format
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Detect food items
        detections = detector.detect_food_items(opencv_image)
        
        logger.info(f"Successfully processed image, found {len(detections)} items")
        
        return JSONResponse(content={
            "success": True,
            "detections": detections,
            "total_items": len(detections),
            "timestamp": datetime.now().isoformat()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")

@app.get("/supported_foods")
async def get_supported_foods():
    """Get list of supported food items for detection"""
    return JSONResponse(content={
        "supported_foods": list(TARGET_FOODS.keys()),
        "categories": FOOD_CATEGORIES
    })

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Food Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/process_image": "Upload image for food detection",
            "/supported_foods": "Get list of supported food items"
        }
    }

# Additional endpoints that your frontend expects but we won't implement
# (since you want detection only, not storage)

@app.get("/inventory")
async def get_inventory():
    """Mock inventory endpoint - returns empty since we don't store items"""
    return {
        "items": {},
        "total_items": 0,
        "unique_items": 0,
        "categories": {}
    }

@app.post("/save_inventory")
async def save_inventory(items: dict):
    """Mock save endpoint - doesn't actually save anything"""
    return {"success": True, "message": "Items not saved (detection only mode)"}

@app.post("/reset")
async def reset_inventory():
    """Mock reset endpoint"""
    return {"success": True, "message": "Nothing to reset (detection only mode)"}

@app.delete("/inventory/{item_name}")
async def remove_item(item_name: str, count: int = 1):
    """Mock remove endpoint"""
    return {"success": True, "message": "Item not removed (detection only mode)"}

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "fridge:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )