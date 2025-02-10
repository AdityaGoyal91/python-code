# phi4_llm/real_estate_agent.py

import logging
from pathlib import Path
import torch
from typing import Dict, List, Optional, Union
from .loader import Phi4ModelLoader
from transformers import AutoTokenizer
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealEstateAgent:
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the Real Estate Agent with Phi-4 model."""
        self.loader = Phi4ModelLoader(cache_dir)
        self.model = None
        self.tokenizer = None
        
        # Simplified base context for testing
        self.base_context = """You are a real estate analyst. Analyze the following property and provide:
        1. Estimated value
        2. Key features
        3. Comparison with similar properties
        4. Recommendations"""

    def analyze_property(
        self,
        property_details: Dict[str, Union[str, float, int]],
        comps: List[Dict[str, Union[str, float, int]]],
        market_conditions: Optional[Dict[str, str]] = None,
        max_length: int = 2048,  # Increased max length
        temperature: float = 0.8  # Adjusted temperature
    ) -> str:
        """Analyze a property based on provided details and comparables."""
        if self.model is None or self.tokenizer is None:
            self.load_model()

        # Format the input prompt
        prompt = self._format_analysis_prompt(property_details, comps, market_conditions)
        
        # Debug logging
        logger.info(f"Generated prompt length: {len(prompt)}")
        logger.info("First 100 chars of prompt: " + prompt[:100])
        
        try:
            # Tokenize with padding
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                return_attention_mask=True
            ).to(self.loader.device)
            
            logger.info(f"Input tensor shape: {inputs['input_ids'].shape}")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=max_length,
                    do_sample=True,
                    temperature=temperature,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    top_k=50,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    length_penalty=1.0,
                    no_repeat_ngram_size=2,
                    min_length=100  # Ensure some minimum output
                )
            
            logger.info(f"Output tensor shape: {outputs.shape}")
            
            # Decode and clean up the response
            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            logger.info(f"Generated text length: {len(generated_text)}")
            
            # Find where the prompt ends and response begins
            prompt_end = len(prompt)
            analysis = generated_text[prompt_end:].strip()
            
            logger.info(f"Analysis length: {len(analysis)}")
            
            if not analysis:
                # Try alternative extraction method
                analysis = generated_text.replace(prompt, "").strip()
                
            if not analysis:
                return "Error: Model generated empty response. Please try again with different parameters."
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating analysis: {e}")
            raise

    def _format_analysis_prompt(
        self,
        property_details: Dict[str, Union[str, float, int]],
        comps: List[Dict[str, Union[str, float, int]]],
        market_conditions: Optional[Dict[str, str]] = None
    ) -> str:
        """Format the input data into a prompt for the model."""
        
        # Create a more direct prompt
        prompt = f"""{self.base_context}

            Subject Property Analysis Request:
            Address: {property_details.get('address', 'N/A')}
            Price: ${property_details.get('price', 0):,.2f}
            Square Feet: {property_details.get('sqft', 0):,}
            Bedrooms: {property_details.get('beds', 0)}
            Bathrooms: {property_details.get('baths', 0)}

            Comparable Properties:
        """
                
        for i, comp in enumerate(comps, 1):
            prompt += f"""
                Comp {i}:
                Address: {comp.get('address', 'N/A')}
                Price: ${comp.get('price', 0):,.2f}
                Square Feet: {comp.get('sqft', 0):,}
            """

        if market_conditions:
            prompt += "\nMarket Conditions:\n"
            for key, value in market_conditions.items():
                prompt += f"{key}: {value}\n"

        prompt += "\nPlease provide your analysis:"
        
        return prompt

    def update_context(self, new_context: str) -> None:
        """Update the base context for the agent."""
        self.base_context = new_context
        logger.info("Context updated successfully")

    def __del__(self):
        """Cleanup when the object is deleted."""
        if hasattr(self, 'loader'):
            self.loader.clear_gpu_memory()

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_property = {
        "address": "123 Main St, Palo Alto, CA",
        "price": 2500000,
        "sqft": 1800,
        "beds": 3,
        "baths": 2,
        "lot_size": 5000,
        "year_built": 1985,
        "condition": "Good",
        "features": ["Updated Kitchen", "Solar Panels", "EV Charger"]
    }
    
    sample_comps = [
        {
            "address": "456 Oak St, Palo Alto, CA",
            "price": 2600000,
            "sqft": 1850,
            "beds": 3,
            "baths": 2,
            "dom": 15
        },
        {
            "address": "789 Elm St, Palo Alto, CA",
            "price": 2400000,
            "sqft": 1750,
            "beds": 3,
            "baths": 2,
            "dom": 25
        }
    ]
    
    sample_market = {
        "Median DOM": "18 days",
        "Price Trend": "Up 5% YoY",
        "Inventory": "2.1 months",
        "Sale/List Ratio": "102%"
    }
    
    # Initialize and test the agent
    agent = RealEstateAgent()
    analysis = agent.analyze_property(sample_property, sample_comps, sample_market)
    print(analysis)