import torch
from .loader import Phi4ModelLoader

def ask_phi4(prompt: str, max_length: int = 256, temperature: float = 0.7, debug: bool = False) -> str:
    """
    Send a prompt to Phi-4 and get the response.
    
    Args:
        prompt (str): The input prompt for the model
        max_length (int, optional): Maximum length of the generated response. Defaults to 256.
        temperature (float, optional): Sampling temperature. Higher values make output more random. 
                                     Defaults to 0.7.
        debug (bool, optional): Print debug information. Defaults to False.
    
    Returns:
        str: The model's response
    """
    # Get or initialize model and tokenizer using the loader
    if not hasattr(ask_phi4, '_loader'):
        ask_phi4._loader = Phi4ModelLoader()
        ask_phi4._model = ask_phi4._loader.model
        ask_phi4._tokenizer = ask_phi4._loader.tokenizer
    
    # Prepare input with an instruction format
    formatted_prompt = f"Instruction: {prompt}\nResponse:"
    inputs = ask_phi4._tokenizer(formatted_prompt, return_tensors="pt").to(ask_phi4._model.device)
    
    if debug:
        print(f"Input prompt: {formatted_prompt}")
    
    # Generate response
    with torch.no_grad():
        outputs = ask_phi4._model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=ask_phi4._tokenizer.pad_token_id,
            eos_token_id=ask_phi4._tokenizer.eos_token_id,  # Add explicit EOS token
            early_stopping=True  # Stop when EOS is generated
        )
    
    # Decode full response
    full_response = ask_phi4._tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if debug:
        print(f"Full model output: {full_response}")
    
    # Extract just the response part after our instruction prompt
    try:
        response_start = full_response.find("Response:") + len("Response:")
        if response_start >= 0:
            response = full_response[response_start:].strip()
        else:
            response = full_response[len(formatted_prompt):].strip()
    except Exception as e:
        if debug:
            print(f"Error processing response: {e}")
            print(f"Falling back to full response")
        response = full_response
    
    if debug and not response:
        print("Warning: Empty response generated")
        
    return response

# Example usage in notebook:
"""
# Import and use with debug mode to see what's happening
response = ask_phi4("What is 1 + 1?", debug=True)
print(f"Final response: {response}")

# Once working, use normally
response = ask_phi4(
    "Write a haiku about programming.",
    max_length=128,
    temperature=0.9
)
print(response)
"""