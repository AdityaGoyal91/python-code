# phi4_llm/real_estate_agent.py

import logging
from pathlib import Path
import torch
from typing import Dict, List, Optional, Union
from .loader import Phi4ModelLoader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealEstateAgent:
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the Real Estate Agent with Phi-4 model.
        
        Args:
            cache_dir (str, optional): Custom directory to store the model
        """
        self.loader = Phi4ModelLoader(cache_dir)
        self.model = None
        self.tokenizer = None
        
        # Define the base context for real estate analysis
        self.base_context = """You are an experienced real estate investment analyst and buyers' agent 
        specializing in the San Francisco Bay Area market. Your expertise includes:

        1. Comparative Market Analysis (CMA):
           - Recent sales within the last 6 months
           - Properties within a 1-mile radius
           - Similar square footage (±20%)
           - Similar property characteristics
           - Property condition and upgrades
           - Micro-neighborhood factors

        2. Market Trends Analysis:
           - Price per square foot trends
           - Days on market
           - Sale price to list price ratios
           - Seasonal market patterns
           - Local economic indicators

        3. Location Assessment:
           - School district quality
           - Crime rates and safety
           - Proximity to amenities
           - Public transportation access
           - Future development plans
           - Zoning changes and regulations

        Provide analysis in this format:
        1. Initial value estimate
        2. Key factors affecting the valuation
        3. Comparable property analysis
        4. Market trend implications
        5. Specific recommendations
        """
        
    def load_model(self) -> None:
        """Load the Phi-4 model and tokenizer."""
        try:
            self.model, self.tokenizer = self.loader.load_model()
            logger.info("Real Estate Agent model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def analyze_property(
        self,
        property_details: Dict[str, Union[str, float, int]],
        comps: List[Dict[str, Union[str, float, int]]],
        market_conditions: Optional[Dict[str, str]] = None,
        max_length: int = 1024,
        temperature: float = 0.7
    ) -> str:
        """
        Analyze a property based on provided details and comparables.
        
        Args:
            property_details: Dict containing property information
                {
                    "address": str,
                    "price": float,
                    "sqft": int,
                    "beds": int,
                    "baths": float,
                    "lot_size": Optional[int],
                    "year_built": Optional[int],
                    "condition": Optional[str],
                    "features": Optional[List[str]]
                }
            comps: List of comparable properties with similar structure
            market_conditions: Optional dict of current market conditions
            max_length: Maximum length of generated response
            temperature: Temperature for text generation
            
        Returns:
            str: Detailed property analysis
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()

        # Format the input prompt
        prompt = self._format_analysis_prompt(property_details, comps, market_conditions)
        
        # Generate the analysis
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.loader.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    temperature=temperature,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            analysis = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return analysis.replace(prompt, "").strip()
            
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
        
        # Format property details
        prop_str = f"""
        Subject Property:
        - Address: {property_details.get('address', 'N/A')}
        - Price: ${property_details.get('price', 0):,.2f}
        - Square Feet: {property_details.get('sqft', 0):,}
        - Bedrooms: {property_details.get('beds', 0)}
        - Bathrooms: {property_details.get('baths', 0)}
        - Lot Size: {property_details.get('lot_size', 'N/A')}
        - Year Built: {property_details.get('year_built', 'N/A')}
        - Condition: {property_details.get('condition', 'N/A')}
        """
        
        if property_details.get('features'):
            prop_str += f"- Features: {', '.join(property_details['features'])}\n"

        # Format comparable properties
        comps_str = "\nComparable Properties:\n"
        for i, comp in enumerate(comps, 1):
            comps_str += f"""
            Comp {i}:
            - Address: {comp.get('address', 'N/A')}
            - Sale Price: ${comp.get('price', 0):,.2f}
            - Square Feet: {comp.get('sqft', 0):,}
            - Bedrooms: {comp.get('beds', 0)}
            - Bathrooms: {comp.get('baths', 0)}
            - Days on Market: {comp.get('dom', 'N/A')}
            """

        # Format market conditions if provided
        market_str = ""
        if market_conditions:
            market_str = "\nMarket Conditions:\n"
            for key, value in market_conditions.items():
                market_str += f"- {key}: {value}\n"

        # Combine everything into final prompt
        full_prompt = f"""{self.base_context}

        Please analyze the following property:
        {prop_str}
        {comps_str}
        {market_str}

        Provide a detailed analysis including value estimate, key factors, comparable analysis, 
        market trends, and specific recommendations for the buyer.
        """
        
        return full_prompt

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