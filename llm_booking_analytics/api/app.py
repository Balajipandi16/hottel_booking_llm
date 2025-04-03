# FastAPI app setup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from .endpoints import router as api_router

def create_app():
    """Create and configure the FastAPI application"""
    app = FastAPI(
        title="Hotel Booking Analytics API",
        description="API for hotel booking data analytics",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API router
    app.include_router(api_router, prefix="/api")
    
    @app.get("/")
    async def root():
        """Root endpoint for the API"""
        return {
            "message": "Welcome to the Hotel Booking Analytics API",
            "docs": "/docs",
            "version": "1.0.0"
        }
    
    return app

app = create_app()

if __name__ == "__main__":
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)