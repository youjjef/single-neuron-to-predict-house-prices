#!/usr/bin/env python3
"""
Setup script for AI Therapist project
This script helps users set up the project quickly
"""

import os
import subprocess
import sys
import json
import argparse
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def install_dependencies(requirements_file="requirements.txt"):
    """Install Python dependencies"""
    print("📦 Installing dependencies...")
    
    if not os.path.exists(requirements_file):
        print(f"❌ {requirements_file} not found")
        return False
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        "data",
        "models/checkpoints",
        "rag_db",
        "logs",
        "outputs"
    ]
    
    print("📁 Creating directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   Created: {directory}")
    
    print("✅ Directories created")

def create_environment_file():
    """Create a .env file template"""
    env_content = """# AI Therapist Environment Variables
# Copy this to .env and fill in your values

# Hugging Face (for model hosting and access)
HUGGING_FACE_TOKEN=your_hf_token_here

# Optional: OpenAI (for fallback/comparison)
OPENAI_API_KEY=your_openai_key_here

# Optional: Anthropic (for fallback/comparison)
ANTHROPIC_API_KEY=your_anthropic_key_here

# Application settings
APP_NAME=AI Therapist
DEBUG=False
LOG_LEVEL=INFO

# Model settings
DEFAULT_MODEL=microsoft/DialoGPT-medium
USE_QUANTIZATION=True
MAX_MEMORY_GB=4

# RAG settings
RAG_DB_PATH=./rag_db
EMBEDDING_MODEL=all-MiniLM-L6-v2
"""
    
    env_file = ".env.example"
    with open(env_file, "w") as f:
        f.write(env_content)
    
    print(f"✅ Created {env_file} - copy to .env and fill in your values")

def setup_training_data():
    """Set up initial training data"""
    print("📊 Setting up training data...")
    
    try:
        # Import and run the data preparation script
        sys.path.append(".")
        from data.prepare_training_data import create_expanded_dataset
        
        create_expanded_dataset()
        print("✅ Training data created")
        return True
    except Exception as e:
        print(f"❌ Failed to create training data: {e}")
        return False

def setup_rag_database():
    """Initialize the RAG database"""
    print("🔍 Setting up RAG database...")
    
    try:
        sys.path.append(".")
        from rag.rag_system import RAGTherapist
        
        # Initialize RAG system (this will create the database)
        rag_system = RAGTherapist()
        stats = rag_system.get_collection_stats()
        
        print(f"✅ RAG database initialized with {stats.get('total_documents', 0)} documents")
        return True
    except Exception as e:
        print(f"❌ Failed to setup RAG database: {e}")
        return False

def create_run_scripts():
    """Create convenience scripts for running the application"""
    
    # Local run script
    local_run_script = """#!/bin/bash
# Local development script
echo "🚀 Starting AI Therapist locally..."
export PYTHONPATH="${PYTHONPATH}:."
streamlit run app/main.py --server.port 8501 --server.address localhost
"""
    
    with open("run_local.sh", "w") as f:
        f.write(local_run_script)
    
    # Training script
    training_script = """#!/bin/bash
# Training script
echo "🏋️ Starting model training..."
export PYTHONPATH="${PYTHONPATH}:."
python models/train.py
"""
    
    with open("run_training.sh", "w") as f:
        f.write(training_script)
    
    # Data preparation script
    data_script = """#!/bin/bash
# Data preparation script
echo "📊 Preparing training data..."
python data/prepare_training_data.py
"""
    
    with open("prepare_data.sh", "w") as f:
        f.write(data_script)
    
    # Make scripts executable
    for script in ["run_local.sh", "run_training.sh", "prepare_data.sh"]:
        os.chmod(script, 0o755)
    
    print("✅ Created run scripts (run_local.sh, run_training.sh, prepare_data.sh)")

def create_config_file():
    """Create a default configuration file"""
    config = {
        "app": {
            "name": "AI Therapist",
            "version": "1.0.0",
            "debug": False
        },
        "model": {
            "default_model": "microsoft/DialoGPT-medium",
            "use_quantization": True,
            "max_memory_gb": 4,
            "temperature": 0.7,
            "max_length": 512
        },
        "rag": {
            "db_path": "./rag_db",
            "embedding_model": "all-MiniLM-L6-v2",
            "n_results": 3
        },
        "training": {
            "batch_size": 4,
            "learning_rate": 2e-4,
            "num_epochs": 3,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05
        },
        "deployment": {
            "platform": "streamlit_cloud",
            "auto_deploy": False
        }
    }
    
    with open("config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Created config.json")

def run_quick_test():
    """Run a quick test to make sure everything works"""
    print("🧪 Running quick test...")
    
    try:
        # Test imports
        sys.path.append(".")
        from rag.rag_system import RAGTherapist
        from models.inference import TherapistModel
        
        print("   ✅ All imports successful")
        
        # Test RAG system
        rag = RAGTherapist()
        context = rag.get_relevant_context("I feel anxious")
        if context:
            print("   ✅ RAG system working")
        else:
            print("   ⚠️ RAG system returned empty context")
        
        print("✅ Quick test completed")
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*60)
    print("🎉 AI Therapist setup completed!")
    print("="*60)
    print("\n📋 Next Steps:")
    print("\n1. 🔐 Configure your environment:")
    print("   cp .env.example .env")
    print("   # Edit .env with your API keys")
    
    print("\n2. 🚀 Run the application locally:")
    print("   ./run_local.sh")
    print("   # Or: streamlit run app/main.py")
    
    print("\n3. 🏋️ Train your custom model (optional):")
    print("   ./run_training.sh")
    print("   # Or use the Jupyter notebook in notebooks/")
    
    print("\n4. 🌐 Deploy to the cloud:")
    print("   # See deployment/deploy_guide.md for options")
    print("   # Recommended: Streamlit Cloud (free)")
    
    print("\n📚 Documentation:")
    print("   - README.md - Project overview")
    print("   - deployment/deploy_guide.md - Deployment options")
    print("   - notebooks/ - Training notebooks")
    
    print("\n🆘 Need help?")
    print("   - Check the documentation")
    print("   - Review the code comments")
    print("   - Create an issue on GitHub")
    
    print("\n⚠️ Important Disclaimers:")
    print("   - This is for educational purposes only")
    print("   - Not a replacement for professional therapy")
    print("   - Please review and follow ethical AI guidelines")

def main():
    parser = argparse.ArgumentParser(description="Set up AI Therapist project")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip-data", action="store_true", help="Skip data setup")
    parser.add_argument("--skip-rag", action="store_true", help="Skip RAG setup")
    parser.add_argument("--skip-test", action="store_true", help="Skip quick test")
    parser.add_argument("--minimal", action="store_true", help="Minimal setup (directories only)")
    
    args = parser.parse_args()
    
    print("🧠 AI Therapist Setup")
    print("=" * 30)
    
    # Check Python version
    check_python_version()
    
    # Create directories
    create_directories()
    
    if args.minimal:
        print("✅ Minimal setup completed")
        return
    
    # Install dependencies
    if not args.skip_deps:
        if not install_dependencies():
            print("⚠️ Continuing without dependency installation")
    
    # Create configuration files
    create_environment_file()
    create_config_file()
    create_run_scripts()
    
    # Set up training data
    if not args.skip_data:
        if not setup_training_data():
            print("⚠️ Continuing without training data")
    
    # Set up RAG database
    if not args.skip_rag:
        if not setup_rag_database():
            print("⚠️ Continuing without RAG setup")
    
    # Run quick test
    if not args.skip_test:
        if not run_quick_test():
            print("⚠️ Some components may not be working correctly")
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()