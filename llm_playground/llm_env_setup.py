import os
import sys
import subprocess
import torch
from pathlib import Path

def check_environment():
    """Check if running in virtual environment"""
    in_venv = sys.prefix != sys.base_prefix
    if not in_venv:
        print("Warning: Not running in a virtual environment!")
        response = input("Would you like to create one now? (y/n): ")
        if response.lower() == 'y':
            create_venv()
        else:
            print("Continuing without virtual environment...")
    return in_venv

def create_venv():
    """Create virtual environment"""
    venv_path = Path('llm_env')
    if venv_path.exists():
        print("Virtual environment already exists!")
        return
    
    subprocess.run([sys.executable, '-m', 'venv', 'llm_env'])
    print("Virtual environment created. Please activate it and run this script again.")
    if os.name == 'nt':  # Windows
        print("Activate by running: llm_env\\Scripts\\activate")
    else:  # Linux/Mac
        print("Activate by running: source llm_env/bin/activate")
    sys.exit(0)

def install_requirements():
    """Install required packages"""
    packages = [
        'torch torchvision --index-url https://download.pytorch.org/whl/cu118',
        'transformers',
        'accelerate',
        'safetensors'
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.run(f"pip install {package}", shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package}: {e}")
            return False
    return True

def setup_model_directory():
    """Setup custom directory for model storage"""
    model_dir = Path.home() / 'llm_models'
    model_dir.mkdir(exist_ok=True)
    
    # Set environment variable for transformers
    os.environ['TRANSFORMERS_CACHE'] = str(model_dir)
    return model_dir

def verify_installation():
    """Verify critical components are installed correctly"""
    checks = {
        'PyTorch': lambda: torch.cuda.is_available(),
        'GPU Support': lambda: torch.cuda.device_count() > 0,
        'CUDA Version': lambda: torch.version.cuda if torch.cuda.is_available() else None
    }
    
    all_passed = True
    for name, check in checks.items():
        try:
            result = check()
            status = "✓" if result else "✗"
            print(f"{name}: {status} ({result})")
            if not result:
                all_passed = False
        except Exception as e:
            print(f"{name}: ✗ (Error: {e})")
            all_passed = False
    
    return all_passed

def main():
    print("Starting LLM environment setup...")
    
    # Check/create virtual environment
    if not check_environment():
        return
    
    # Install requirements
    print("\nInstalling required packages...")
    if not install_requirements():
        print("Failed to install all requirements!")
        return
    
    # Setup model directory
    print("\nSetting up model directory...")
    model_dir = setup_model_directory()
    print(f"Model directory set to: {model_dir}")
    
    # Verify installation
    print("\nVerifying installation...")
    if verify_installation():
        print("\nSetup completed successfully!")
        print(f"\nNext steps:")
        print(f"1. Your models will be stored in: {model_dir}")
        print(f"2. Run your LLM scripts from this environment")
        print(f"3. To deactivate the environment, run 'deactivate'")
    else:
        print("\nSetup completed with some issues. Please review the verification results above.")

if __name__ == "__main__":
    main()