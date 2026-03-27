#!/usr/bin/env python3
"""
Quick Fix Script for Car Damage Detection System
Resolves NumPy compatibility and environment issues
"""

import subprocess
import sys
import os
import platform

def run_command(command, description=""):
    """Run a command and return success status"""
    print(f"\n{'='*50}")
    print(f"Running: {description or command}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.stdout:
            print("Output:", result.stdout)
        
        if result.stderr and result.returncode != 0:
            print("Error:", result.stderr)
            return False
            
        print("✅ Success!")
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and 8 <= version.minor <= 11:
        print("✅ Python version is compatible")
        return True
    else:
        print("❌ Python version should be 3.8-3.11")
        return False

def fix_numpy_compatibility():
    """Fix NumPy compatibility issue"""
    print("\n🔧 Fixing NumPy compatibility issue...")
    
    commands = [
        ("pip uninstall numpy -y", "Uninstalling current NumPy"),
        ("pip uninstall pyarrow -y", "Uninstalling PyArrow"),
        ("pip install numpy==1.24.3", "Installing compatible NumPy"),
        ("pip install pyarrow", "Reinstalling PyArrow"),
        ("pip install --upgrade streamlit", "Upgrading Streamlit"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    
    return True

def install_requirements():
    """Install all requirements"""
    print("\n📦 Installing requirements...")
    
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found")
        return False
    
    return run_command("pip install -r requirements.txt", "Installing all requirements")

def verify_installation():
    """Verify that all modules can be imported"""
    print("\n🔍 Verifying installation...")
    
    modules = [
        "streamlit",
        "cv2",
        "ultralytics", 
        "numpy",
        "torch"
    ]
    
    for module in modules:
        try:
            __import__(module)
            print(f"✅ {module} imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import {module}: {e}")
            return False
    
    return True

def check_model_file():
    """Check if YOLO model file exists"""
    model_path = r"C:\Users\saima\Downloads\allyolov8best.pt"
    
    if os.path.exists(model_path):
        print(f"✅ Model file found: {model_path}")
        return True
    else:
        print(f"⚠️  Model file not found: {model_path}")
        print("The application will download a default model automatically")
        return True

def main():
    """Main fix function"""
    print("🚗 Car Damage Detection - Environment Fix Script")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Please install Python 3.8-3.11")
        return False
    
    # Fix NumPy compatibility
    if not fix_numpy_compatibility():
        print("\n❌ Failed to fix NumPy compatibility")
        return False
    
    # Install requirements
    if not install_requirements():
        print("\n❌ Failed to install requirements")
        return False
    
    # Verify installation
    if not verify_installation():
        print("\n❌ Installation verification failed")
        return False
    
    # Check model file
    check_model_file()
    
    print("\n" + "=" * 60)
    print("🎉 Environment setup completed successfully!")
    print("=" * 60)
    print("\nYou can now run the application:")
    print("streamlit run stcar.py")
    print("\nOr use the alternative command:")
    print("python -m streamlit run stcar.py")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Script interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)