# Minimal FastAPI Application
# Use this template for prototypes, learning, or simple APIs
#
# Run: uvicorn main:app --reload
# Docs: http://localhost:8000/docs

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI(
    title="My API",
    description="A minimal FastAPI application",
    version="1.0.0"
)

# Models
class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float

class ItemResponse(Item):
    id: int

# In-memory storage (replace with database in production)
items_db: dict[int, Item] = {}
item_id_counter = 0

# Endpoints
@app.get("/")
def root():
    return {"message": "Welcome to the API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/items", response_model=list[ItemResponse])
def list_items():
    return [ItemResponse(id=id, **item.model_dump()) for id, item in items_db.items()]

@app.get("/items/{item_id}", response_model=ItemResponse)
def get_item(item_id: int):
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    return ItemResponse(id=item_id, **items_db[item_id].model_dump())

@app.post("/items", response_model=ItemResponse, status_code=201)
def create_item(item: Item):
    global item_id_counter
    item_id_counter += 1
    items_db[item_id_counter] = item
    return ItemResponse(id=item_id_counter, **item.model_dump())

@app.delete("/items/{item_id}", status_code=204)
def delete_item(item_id: int):
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    del items_db[item_id]
