#!/usr/bin/env python3
"""
Startup script for Kolam AI Generator
This script starts both the FastAPI backend and provides instructions for the React frontend
"""

import subprocess
import sys
import os
import time
import webbrowser
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    try:
        import fastapi
        import uvicorn
        import requests
        print("✅ All required Python packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def start_backend():
    """Start the FastAPI backend server"""
    print("🚀 Starting FastAPI backend server...")
    print("📍 Backend will be available at: http://localhost:8000")
    print("📚 API documentation at: http://localhost:8000/docs")
    
    try:
        # Start the FastAPI server
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "app:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Backend server stopped")
    except Exception as e:
        print(f"❌ Error starting backend: {e}")

def print_frontend_instructions():
    """Print instructions for starting the React frontend"""
    print("\n" + "="*60)
    print("🎨 KOLAM AI GENERATOR - SETUP INSTRUCTIONS")
    print("="*60)
    print("\n📋 To complete the setup, you need to start the React frontend:")
    print("\n1. Open a new terminal/command prompt")
    print("2. Navigate to the sd-frontend directory:")
    print("   cd sd-frontend")
    print("3. Install dependencies (if not already done):")
    print("   npm install")
    print("4. Start the React development server:")
    print("   npm start")
    print("\n🌐 The frontend will be available at: http://localhost:3000")
    print("\n⚠️  IMPORTANT: Make sure Stable Diffusion WebUI is running with --api flag")
    print("   Example: python launch.py --api --listen")
    print("\n" + "="*60)

def main():
    print("🎨 Kolam AI Generator - Starting Server")
    print("="*50)
    
    # Check if we're in the right directory
    if not os.path.exists("app.py"):
        print("❌ Error: app.py not found. Please run this script from the project root directory.")
        sys.exit(1)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Print frontend instructions
    print_frontend_instructions()
    
    # Wait a moment for user to read instructions
    print("\n⏳ Starting backend in 3 seconds...")
    time.sleep(3)
    
    # Start backend
    start_backend()

if __name__ == "__main__":
    main()
