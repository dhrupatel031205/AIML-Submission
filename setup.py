#!/usr/bin/env python3
"""
Setup script for AmbedkarGPT RAG System
This script helps verify prerequisites and set up the environment.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and return success status"""
    print(f"\n{'='*50}")
    print(f"Step: {description}")
    print(f"Command: {command}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ SUCCESS")
            if result.stdout:
                print(f"Output: {result.stdout}")
            return True
        else:
            print("❌ FAILED")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible. Requires Python 3.8+")
        return False

def check_ollama():
    """Check if Ollama is installed and Mistral model is available"""
    print("Checking Ollama installation...")
    
    # Check if Ollama is installed
    if not run_command("ollama --version", "Check Ollama installation"):
        print("❌ Ollama is not installed. Please install from https://ollama.ai/")
        return False
    
    # Check if Mistral model is available
    if not run_command("ollama list | grep mistral", "Check Mistral model"):
        print("❌ Mistral model is not installed. Run: ollama pull mistral")
        return False
    
    print("✅ Ollama and Mistral model are available")
    return True

def check_virtual_env():
    """Check if virtual environment is activated"""
    print("Checking virtual environment...")
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment is activated")
        return True
    else:
        print("❌ Virtual environment is not activated")
        print("Please activate the virtual environment:")
        print("  Windows: ambedkar_env\\Scripts\\activate")
        print("  macOS/Linux: source ambedkar_env/bin/activate")
        return False

def install_dependencies():
    """Install Python dependencies"""
    print("Installing Python dependencies...")
    return run_command("pip install -r requirements.txt", "Install dependencies")

def verify_imports():
    """Verify that all required packages can be imported"""
    print("Verifying package imports...")
    
    required_packages = [
        'langchain',
        'langchain_community', 
        'chromadb',
        'sentence_transformers',
        'transformers',
        'torch',
        'requests',
        'numpy'
    ]
    
    for package in required_packages:
        try:
            if package == 'langchain_community':
                import langchain_community
            elif package == 'sentence_transformers':
                import sentence_transformers
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            return False
    
    return True

def main():
    """Main setup function"""
    print("🚀 AmbedkarGPT RAG System Setup")
    print("="*60)
    
    steps = [
        ("Python Version Check", check_python_version),
        ("Virtual Environment Check", check_virtual_env),
        ("Ollama Check", check_ollama),
        ("Install Dependencies", install_dependencies),
        ("Verify Imports", verify_imports)
    ]
    
    all_passed = True
    
    for step_name, step_func in steps:
        print(f"\n🔍 {step_name}")
        if not step_func():
            all_passed = False
            print(f"\n❌ Setup failed at: {step_name}")
            break
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 SETUP COMPLETE! 🎉")
        print("\nYou can now run the system with:")
        print("python main.py")
        print("\nFor troubleshooting, check the README.md file")
    else:
        print("❌ SETUP INCOMPLETE")
        print("\nPlease resolve the issues above and run setup again")
        print("Check README.md for detailed troubleshooting instructions")
    
    print("="*60)

if __name__ == "__main__":
    main()
