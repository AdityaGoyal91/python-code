from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import warnings

class Phi4Agent:
    def __init__(self, use_4bit=False):  # Changed default to False
        """
        Initialize Phi-4 model and tokenizer.
        Args:
            use_4bit (bool): Whether to use 4-bit quantization for efficiency.
                            Requires bitsandbytes package.
        """
        self.model_name = "microsoft/phi-4"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading tokenizer from {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        print(f"Loading model from {self.model_name}...")
        try:
            if use_4bit:
                try:
                    import bitsandbytes
                    # Load in 4-bit quantization for memory efficiency
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        device_map="auto",
                        torch_dtype=torch.bfloat16,
                        quantization_config={"load_in_4bit": True}
                    )
                except ImportError:
                    warnings.warn("bitsandbytes not found. Falling back to regular loading. "
                                "Install with: pip install bitsandbytes")
                    self._load_regular_model()
            else:
                self._load_regular_model()
            
            print("Model loaded successfully!")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def _load_regular_model(self):
        """Load model without quantization"""
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )

    def ask(self, prompt: str, max_length: int = 512, temperature: float = 0.7) -> str:
        """
        Send a prompt to Phi-4 and get a response.
        
        Args:
            prompt (str): The input prompt
            max_length (int): Maximum length of generated response
            temperature (float): Sampling temperature (higher = more random)
            
        Returns:
            str: The model's response
        """
        # Format the prompt
        formatted_prompt = f"Instruction: {prompt}\nResponse:"
        
        # Tokenize
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=True
            )
        
        # Decode
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the response part
        response_start = full_response.find("Response:") + len("Response:")
        response = full_response[response_start:].strip()
        
        return response

def main():
    """Example usage of the Phi4Agent"""
    # Initialize the agent
    print("Initializing Phi-4 agent...")
    agent = Phi4Agent(use_4bit=False)  # Changed default to False
    
    # Example prompts
    prompts = [
        "What is 1 + 1?",
        "Write a one-sentence story.",
        "What is the capital of France?"
    ]
    
    # Test each prompt
    print("\nTesting prompts:")
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        response = agent.ask(prompt)
        print(f"Response: {response}")

if __name__ == "__main__":
    main()