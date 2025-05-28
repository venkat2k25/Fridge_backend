import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from ultralytics import YOLO
from PIL import Image
import io
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Smart Fridge Inventory API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class InventoryItem(BaseModel):
    name: str
    count: int
    category: str = "Other"
    confidence: float = 0.5
    expiry_days: Optional[int] = None
    storage_location: str = "main_compartment"

class SaveInventoryRequest(BaseModel):
    items: List[InventoryItem]

class UpdateItemRequest(BaseModel):
    count: int
    expiry_days: Optional[int] = None

# File-based inventory storage
INVENTORY_FILE = "data/inventory.json"

def load_inventory():
    global inventory_storage
    try:
        os.makedirs(os.path.dirname(INVENTORY_FILE), exist_ok=True)
        with open(INVENTORY_FILE, 'r') as f:
            inventory_storage = json.load(f)
        logger.info("Loaded inventory from file")
    except FileNotFoundError:
        inventory_storage = {
            "items": {},
            "total_items": 0,
            "unique_items": 0,
            "categories": {},
            "expiring_soon": [],
            "last_updated": None
        }
        logger.info("Initialized new inventory")
    except Exception as e:
        logger.error(f"Failed to load inventory: {e}")
        inventory_storage = {
            "items": {},
            "total_items": 0,
            "unique_items": 0,
            "categories": {},
            "expiring_soon": [],
            "last_updated": None
        }

def save_inventory_to_file():
    try:
        with open(INVENTORY_FILE, 'w') as f:
            json.dump(inventory_storage, f, indent=2)
        logger.info("Saved inventory to file")
    except Exception as e:
        logger.error(f"Failed to save inventory: {e}")

# Load inventory at startup
load_inventory()

# Fridge-specific food items
FRIDGE_FOODS = {
    'milk': {
        'aliases': ['milk', 'dairy', 'bottle', 'milk bottle'],
        'category': 'Dairy',
        'icon': '🥛',
        'typical_expiry_days': 7,
        'storage_location': 'door_shelf'
    },
    'yogurt': {
        'aliases': ['yogurt', 'yoghurt', 'dairy', 'cup', 'yogurt cup'],
        'category': 'Dairy',
        'icon': '🥛',
        'typical_expiry_days': 14,
        'storage_location': 'main_compartment'
    },
    'cheese': {
        'aliases': ['cheese', 'cheddar', 'mozzarella', 'dairy'],
        'category': 'Dairy',
        'icon': '🧀',
        'typical_expiry_days': 21,
        'storage_location': 'main_compartment'
    },
    'butter': {
        'aliases': ['butter', 'margarine'],
        'category': 'Dairy',
        'icon': '🧈',
        'typical_expiry_days': 30,
        'storage_location': 'door_shelf'
    },
    'egg': {
        'aliases': ['egg', 'eggs'],
        'category': 'Protein',
        'icon': '🥚',
        'typical_expiry_days': 21,
        'storage_location': 'door_shelf'
    },
    'spinach': {
        'aliases': ['spinach', 'leafy greens', 'leaves', 'greens', 'spinach leaves'],
        'category': 'Vegetables',
        'icon': '🥬',
        'typical_expiry_days': 5,
        'storage_location': 'crisper_drawer'
    },
    'tomato': {
        'aliases': ['tomato', 'tomatoes', 'cherry tomato'],
        'category': 'Vegetables',
        'icon': '🍅',
        'typical_expiry_days': 7,
        'storage_location': 'crisper_drawer'
    },
    'broccoli': {
        'aliases': ['broccoli', 'brocoli'],
        'category': 'Vegetables',
        'icon': '🥦',
        'typical_expiry_days': 5,
        'storage_location': 'crisper_drawer'
    },
    'carrot': {
        'aliases': ['carrot', 'carrots'],
        'category': 'Vegetables',
        'icon': '🥕',
        'typical_expiry_days': 14,
        'storage_location': 'crisper_drawer'
    },
    'green_chilli': {
        'aliases': ['chili', 'chilli', 'green pepper', 'hot pepper', 'green chili', 'pepper'],
        'category': 'Vegetables',
        'icon': '🌶️',
        'typical_expiry_days': 10,
        'storage_location': 'crisper_drawer'
    },
    'bell_pepper': {
        'aliases': ['bell pepper', 'capsicum', 'sweet pepper'],
        'category': 'Vegetables',
        'icon': '🫑',
        'typical_expiry_days': 7,
        'storage_location': 'crisper_drawer'
    },
    'cucumber': {
        'aliases': ['cucumber', 'cucumbers'],
        'category': 'Vegetables',
        'icon': '🥒',
        'typical_expiry_days': 7,
        'storage_location': 'crisper_drawer'
    },
    'lettuce': {
        'aliases': ['lettuce', 'salad', 'leafy greens'],
        'category': 'Vegetables',
        'icon': '🥬',
        'typical_expiry_days': 5,
        'storage_location': 'crisper_drawer'
    },
    'onion': {
        'aliases': ['onion', 'onions'],
        'category': 'Vegetables',
        'icon': '🧅',
        'typical_expiry_days': 21,
        'storage_location': 'main_compartment'
    },
    'garlic': {
        'aliases': ['garlic', 'garlic clove'],
        'category': 'Vegetables',
        'icon': '🧄',
        'typical_expiry_days': 30,
        'storage_location': 'main_compartment'
    },
    'apple': {
        'aliases': ['apple', 'apples'],
        'category': 'Fruits',
        'icon': '🍎',
        'typical_expiry_days': 21,
        'storage_location': 'crisper_drawer'
    },
    'orange': {
        'aliases': ['orange', 'oranges'],
        'category': 'Fruits',
        'icon': '🍊',
        'typical_expiry_days': 14,
        'storage_location': 'crisper_drawer'
    },
    'lemon': {
        'aliases': ['lemon', 'lime', 'citrus', 'lemons'],
        'category': 'Fruits',
        'icon': '🍋',
        'typical_expiry_days': 21,
        'storage_location': 'crisper_drawer'
    },
    'grapes': {
        'aliases': ['grapes', 'grape'],
        'category': 'Fruits',
        'icon': '🍇',
        'typical_expiry_days': 7,
        'storage_location': 'main_compartment'
    },
    'strawberry': {
        'aliases': ['strawberry', 'strawberries', 'berries'],
        'category': 'Fruits',
        'icon': '🍓',
        'typical_expiry_days': 3,
        'storage_location': 'main_compartment'
    },
    'ketchup': {
        'aliases': ['ketchup', 'tomato sauce', 'sauce'],
        'category': 'Condiments',
        'icon': '🍅',
        'typical_expiry_days': 180,
        'storage_location': 'door_shelf'
    },
    'mayonnaise': {
        'aliases': ['mayonnaise', 'mayo'],
        'category': 'Condiments',
        'icon': '🥄',
        'typical_expiry_days': 60,
        'storage_location': 'door_shelf'
    },
    'mustard': {
        'aliases': ['mustard'],
        'category': 'Condiments',
        'icon': '🥄',
        'typical_expiry_days': 365,
        'storage_location': 'door_shelf'
    },
    'juice': {
        'aliases': ['juice', 'orange juice', 'apple juice', 'fruit juice'],
        'category': 'Beverages',
        'icon': '🧃',
        'typical_expiry_days': 7,
        'storage_location': 'door_shelf'
    },
    'soda': {
        'aliases': ['soda', 'soft drink', 'cola', 'pepsi', 'coke'],
        'category': 'Beverages',
        'icon': '🥤',
        'typical_expiry_days': 90,
        'storage_location': 'door_shelf'
    },
    'leftover': {
        'aliases': ['leftover', 'leftovers', 'cooked food', 'prepared meal'],
        'category': 'Leftovers',
        'icon': '🥘',
        'typical_expiry_days': 3,
        'storage_location': 'main_compartment'
    },
    'chicken': {
        'aliases': ['chicken', 'poultry', 'chicken breast'],
        'category': 'Protein',
        'icon': '🍗',
        'typical_expiry_days': 2,
        'storage_location': 'main_compartment'
    },
    'fish': {
        'aliases': ['fish', 'salmon', 'tuna', 'seafood'],
        'category': 'Protein',
        'icon': '🐟',
        'typical_expiry_days': 2,
        'storage_location': 'main_compartment'
    },
    'ground_meat': {
        'aliases': ['ground beef', 'minced meat', 'hamburger'],
        'category': 'Protein',
        'icon': '🥩',
        'typical_expiry_days': 1,
        'storage_location': 'main_compartment'
    }
}

STORAGE_LOCATIONS = {
    'door_shelf': 'Door Shelf',
    'main_compartment': 'Main Compartment',
    'crisper_drawer': 'Crisper Drawer',
    'freezer': 'Freezer'
}

class FridgeFoodDetector:
    def __init__(self):
        logger.info("Initializing FridgeFoodDetector")
        try:
            self.model = YOLO('yolov8n.pt')
            logger.info("YOLO model loaded successfully for fridge detection")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model = None
            logger.warning("Running in mock mode - will return dummy fridge detections")

    def detect_fridge_items(self, image: np.ndarray) -> List[Dict[str, Any]]:
        try:
            if self.model is None:
                logger.info("Returning mock fridge detection data")
                return [
                    {
                        'item': 'tomato',
                        'count': 3,
                        'confidence': 0.89,
                        'category': 'Vegetables',
                        'icon': '🍅',
                        'expiry_days': 7,
                        'storage_location': 'crisper_drawer'
                    },
                    {
                        'item': 'spinach',
                        'count': 1,
                        'confidence': 0.85,
                        'category': 'Vegetables',
                        'icon': '🥬',
                        'expiry_days': 5,
                        'storage_location': 'crisper_drawer'
                    },
                    {
                        'item': 'yogurt',
                        'count': 2,
                        'confidence': 0.92,
                        'category': 'Dairy',
                        'icon': '🥛',
                        'expiry_days': 14,
                        'storage_location': 'main_compartment'
                    },
                    {
                        'item': 'milk',
                        'count': 1,
                        'confidence': 0.88,
                        'category': 'Dairy',
                        'icon': '🥛',
                        'expiry_days': 7,
                        'storage_location': 'door_shelf'
                    }
                ]

            results = self.model(image, conf=0.25)
            detected_items = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = self.model.names[class_id].lower()
                        logger.info(f"Detected: {class_name} with confidence {confidence}")
                        matched_food = self._match_fridge_item(class_name)
                        if matched_food and confidence > 0.25:
                            food_info = FRIDGE_FOODS[matched_food]
                            detected_items.append({
                                'item': matched_food,
                                'count': 1,
                                'confidence': confidence,
                                'category': food_info['category'],
                                'icon': food_info['icon'],
                                'expiry_days': food_info['typical_expiry_days'],
                                'storage_location': food_info['storage_location'],
                                'detected_class': class_name
                            })
                        else:
                            logger.info(f"Unmatched detection: {class_name} (conf: {confidence})")
            grouped_items = self._group_similar_items(detected_items)
            logger.info(f"Final grouped fridge detections: {grouped_items}")
            return grouped_items
        except Exception as e:
            logger.error(f"Error during fridge food detection: {e}")
            return []

    def _match_fridge_item(self, detected_class: str) -> Optional[str]:
        detected_class = detected_class.lower().strip()
        for food_name, food_info in FRIDGE_FOODS.items():
            aliases = food_info['aliases']
            if detected_class in aliases:
                return food_name
            for alias in aliases:
                if alias.lower() in detected_class or detected_class in alias.lower():
                    return food_name
        coco_to_fridge_mappings = {
            'bottle': 'milk',
            'cup': 'yogurt',
            'bowl': 'leftover',
            'banana': None,
            'apple': 'apple',
            'orange': 'orange',
            'broccoli': 'broccoli',
            'carrot': 'carrot',
            'sandwich': 'leftover',
            'pizza': 'leftover',
            'cake': 'leftover',
            'donut': None,
            'hot dog': 'leftover',
            'wine glass': None
        }
        return coco_to_fridge_mappings.get(detected_class)

    def _group_similar_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        grouped = {}
        for item in items:
            food_name = item['item']
            if food_name in grouped:
                grouped[food_name]['count'] += item['count']
                grouped[food_name]['confidence'] = max(
                    grouped[food_name]['confidence'],
                    item['confidence']
                )
            else:
                grouped[food_name] = item.copy()
        return list(grouped.values())

# Initialize the detector
try:
    detector = FridgeFoodDetector()
    logger.info("Fridge food detector initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize fridge food detector: {e}")
    detector = None

def update_inventory_storage(items: List[Dict[str, Any]]):
    global inventory_storage
    for item_data in items:
        item_name = item_data['name']
        count = item_data['count']
        category = item_data.get('category', 'Other')
        expiry_days = item_data.get('expiry_days')
        storage_location = item_data.get('storage_location', 'main_compartment')
        expiry_date = None
        if expiry_days:
            expiry_date = (datetime.now() + timedelta(days=expiry_days)).isoformat()
        if item_name in inventory_storage['items']:
            inventory_storage['items'][item_name]['count'] += count
            if expiry_date:
                inventory_storage['items'][item_name]['expiry_date'] = expiry_date
        else:
            food_info = FRIDGE_FOODS.get(item_name, {})
            inventory_storage['items'][item_name] = {
                'name': item_name,
                'count': count,
                'category': category,
                'icon': food_info.get('icon', '🥘'),
                'storage_location': storage_location,
                'expiry_date': expiry_date,
                'added_date': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat()
            }
    _update_inventory_stats()
    save_inventory_to_file()

def _update_inventory_stats():
    global inventory_storage
    inventory_storage['total_items'] = sum(
        item['count'] for item in inventory_storage['items'].values()
    )
    inventory_storage['unique_items'] = len(inventory_storage['items'])
    inventory_storage['last_updated'] = datetime.now().isoformat()
    categories = {}
    for item in inventory_storage['items'].values():
        category = item.get('category', 'Other')
        categories[category] = categories.get(category, 0) + item['count']
    inventory_storage['categories'] = categories
    expiring_soon = []
    current_time = datetime.now()
    for item_name, item_data in inventory_storage['items'].items():
        expiry_date_str = item_data.get('expiry_date')
        if expiry_date_str:
            expiry_date = datetime.fromisoformat(expiry_date_str)
            days_until_expiry = (expiry_date - current_time).days
            if days_until_expiry <= 3:
                expiring_soon.append({
                    'name': item_name,
                    'days_until_expiry': days_until_expiry,
                    'count': item_data['count'],
                    'category': item_data['category'],
                    'icon': item_data.get('icon', '🥘')
                })
    inventory_storage['expiring_soon'] = expiring_soon

@app.get("/health")
async def health_check():
    if detector is None:
        return {"status": "unhealthy", "message": "Fridge detection model not loaded"}
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "supported_items": len(FRIDGE_FOODS),
        "storage_locations": list(STORAGE_LOCATIONS.keys())
    }

@app.post("/process_image")
async def process_fridge_image(file: UploadFile = File(...)):
    logger.info(f"Received fridge image upload: {file.filename}, Content-Type: {file.content_type}")
    if detector is None:
        logger.error("Fridge food detection model not available")
        raise HTTPException(status_code=500, detail="Fridge food detection model not available")
    try:
        if not file.content_type or not file.content_type.startswith('image/'):
            logger.error(f"Invalid file type: {file.content_type}")
            raise HTTPException(status_code=400, detail="File must be an image")
        contents = await file.read()
        logger.info(f"Image size: {len(contents)} bytes")
        pil_image = Image.open(io.BytesIO(contents))
        logger.info(f"PIL Image mode: {pil_image.mode}, Size: {pil_image.size}")
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        detections = detector.detect_fridge_items(opencv_image)
        logger.info(f"Successfully processed fridge image, found {len(detections)} items: {detections}")
        response_data = {
            "success": True,
            "detections": detections,
            "total_items": len(detections),
            "categories_found": list(set(item['category'] for item in detections)),
            "timestamp": datetime.now().isoformat()
        }
        logger.info(f"Sending response: {response_data}")
        return JSONResponse(content=response_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing fridge image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process fridge image: {str(e)}")

@app.get("/inventory")
async def get_fridge_inventory():
    _update_inventory_stats()
    logger.info(f"Returning fridge inventory: {inventory_storage}")
    return JSONResponse(content=inventory_storage)

@app.post("/save_inventory")
async def save_fridge_inventory(request: SaveInventoryRequest):
    try:
        logger.info(f"Saving fridge inventory items: {request.items}")
        items_data = [item.dict() for item in request.items]
        update_inventory_storage(items_data)
        logger.info(f"Updated fridge inventory storage: {inventory_storage}")
        return JSONResponse(content={
            "success": True,
            "message": f"Successfully saved {len(request.items)} items to fridge",
            "inventory": inventory_storage
        })
    except Exception as e:
        logger.error(f"Error saving fridge inventory: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save fridge inventory: {str(e)}")

@app.put("/inventory/{item_name}")
async def update_fridge_item(item_name: str, request: UpdateItemRequest):
    try:
        if item_name not in inventory_storage['items']:
            raise HTTPException(status_code=404, detail="Item not found in fridge inventory")
        inventory_storage['items'][item_name]['count'] = request.count
        if request.expiry_days is not None:
            expiry_date = (datetime.now() + timedelta(days=request.expiry_days)).isoformat()
            inventory_storage['items'][item_name]['expiry_date'] = expiry_date
        inventory_storage['items'][item_name]['last_updated'] = datetime.now().isoformat()
        _update_inventory_stats()
        save_inventory_to_file()
        return JSONResponse(content={
            "success": True,
            "message": f"Updated {item_name}",
            "item": inventory_storage['items'][item_name],
            "inventory": inventory_storage
        })
    except Exception as e:
        logger.error(f"Error updating fridge item: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update fridge item: {str(e)}")

@app.delete("/inventory/{item_name}")
async def remove_fridge_item(item_name: str, count: int = 1):
    try:
        if item_name in inventory_storage['items']:
            current_count = inventory_storage['items'][item_name]['count']
            new_count = max(0, current_count - count)
            if new_count == 0:
                del inventory_storage['items'][item_name]
                logger.info(f"Completely removed {item_name} from fridge")
            else:
                inventory_storage['items'][item_name]['count'] = new_count
                inventory_storage['items'][item_name]['last_updated'] = datetime.now().isoformat()
                logger.info(f"Reduced {item_name} count to {new_count}")
            _update_inventory_stats()
            save_inventory_to_file()
            return JSONResponse(content={
                "success": True,
                "message": f"Removed {count} of {item_name} from fridge",
                "inventory": inventory_storage
            })
        else:
            raise HTTPException(status_code=404, detail="Item not found in fridge inventory")
    except Exception as e:
        logger.error(f"Error removing fridge item: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to remove fridge item: {str(e)}")

@app.post("/reset")
async def reset_fridge_inventory():
    global inventory_storage
    inventory_storage = {
        "items": {},
        "total_items": 0,
        "unique_items": 0,
        "categories": {},
        "expiring_soon": [],
        "last_updated": datetime.now().isoformat()
    }
    save_inventory_to_file()
    logger.info("Fridge inventory reset")
    return JSONResponse(content={"success": True, "message": "Fridge inventory reset successfully"})

@app.get("/supported_foods")
async def get_supported_fridge_foods():
    foods_with_details = {}
    for food_name, food_info in FRIDGE_FOODS.items():
        foods_with_details[food_name] = {
            'category': food_info['category'],
            'icon': food_info['icon'],
            'typical_expiry_days': food_info['typical_expiry_days'],
            'storage_location': food_info['storage_location'],
            'aliases': food_info['aliases']
        }
    return JSONResponse(content={
        "supported_fridge_foods": foods_with_details,
        "categories": list(set(info['category'] for info in FRIDGE_FOODS.values())),
        "storage_locations": STORAGE_LOCATIONS
    })

@app.get("/expiring_soon")
async def get_expiring_items():
    _update_inventory_stats()
    return JSONResponse(content={
        "expiring_soon": inventory_storage['expiring_soon'],
        "count": len(inventory_storage['expiring_soon'])
    })

@app.get("/")
async def root():
    return {
        "message": "Smart Fridge Inventory API",
        "version": "1.0.0",
        "status": "running",
        "supported_items": len(FRIDGE_FOODS),
        "endpoints": {
            "/health": "Health check",
            "/process_image": "Upload fridge image for food detection",
            "/inventory": "Get current fridge inventory",
            "/save_inventory": "Save items to fridge inventory",
            "/inventory/{item_name}": "Update or delete specific fridge item",
            "/reset": "Reset fridge inventory",
            "/supported_foods": "Get list of supported fridge food items",
            "/expiring_soon": "Get items expiring within 3 days"
        }
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False
    )