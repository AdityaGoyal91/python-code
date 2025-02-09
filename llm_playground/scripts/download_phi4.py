import sys
import torch
from pathlib import Path
import logging

# Add the phi4_llm directory to Python path
phi4_path = str(Path(__file__).parent.parent)

# check that the string is returning correctly
print(phi4_path)

sys.path.append(phi4_path)

import phi4_llm
from phi4_llm.loader import Phi4ModelLoader, get_phi4_system_info  # Updated import path

def main():
    # Print system information
    print("\nSystem Information:")
    for key, value in get_phi4_system_info().items():
        print(f"{key}: {value}")
    
    # Initialize model loader
    loader = Phi4ModelLoader()
    
    # Check system requirements
    if not loader.check_system_requirements():
        response = input("System does not meet minimum requirements. Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Download and load model
    print("\nDownloading and loading Phi-4 model...")
    try:
        model, tokenizer = loader.load_model()
        print(f"\nPhi-4 model loaded successfully!")
        print(f"Model size on disk: {loader.get_model_size():.2f}GB")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Test the model
    test_prompt = "This is a test prompt to verify the model is working."
    print("\nTesting Phi-4 model with a simple prompt...")
    inputs = tokenizer(test_prompt, return_tensors="pt").to(loader.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=50,
            num_return_sequences=1
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\nTest response:", response)
    print("\nPhi-4 model is ready for use!")

if __name__ == "__main__":
    main()