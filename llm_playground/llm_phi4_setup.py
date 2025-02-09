from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

def load_model(model_name="microsoft/phi-4", custom_path=None):
    # Set custom path if provided
    if custom_path:
        os.environ['TRANSFORMERS_CACHE'] = custom_path
    
    try:
        # Load with error handling and progress display
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            progress_bar=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            progress_bar=True
        )
        
        return tokenizer, model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None