# phi4_llm/loader.py

import os
import logging
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import psutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Phi4ModelLoader:  # Renamed from ModelLoader
    def __init__(self, cache_dir=None):
        """
        Initialize the Phi-4 model loader with configuration.
        
        Args:
            cache_dir (str, optional): Custom directory to store the model
        """
        self.model_name = "microsoft/phi-4"  # Hardcoded since this is Phi-4 specific
        self.cache_dir = cache_dir or str(Path.home() / 'llm_models' / 'phi4')  # Updated path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        
        # Create cache directory if it doesn't exist
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Set environment variable for transformers
        os.environ['TRANSFORMERS_CACHE'] = self.cache_dir

    def check_system_requirements(self):
        """Check if system meets minimum requirements."""
        min_ram_gb = 16
        min_gpu_memory_gb = 8
        
        # Check RAM
        ram_gb = psutil.virtual_memory().total / (1024**3)
        if ram_gb < min_ram_gb:
            logger.warning(f"System RAM ({ram_gb:.1f}GB) is below recommended minimum ({min_ram_gb}GB)")
            return False
            
        # Check GPU if available
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory_gb < min_gpu_memory_gb:
                logger.warning(f"GPU memory ({gpu_memory_gb:.1f}GB) is below recommended minimum ({min_gpu_memory_gb}GB)")
                return False
                
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {gpu_memory_gb:.1f}GB")
        else:
            logger.warning("No GPU detected, using CPU (this will be slow)")
            
        logger.info(f"System RAM: {ram_gb:.1f}GB")
        return True
        
    def load_model(self, force_reload=False):
        """
        Load the model and tokenizer.
        
        Args:
            force_reload (bool): If True, reload model even if already loaded
            
        Returns:
            tuple: (model, tokenizer)
        """
        if self.model is not None and self.tokenizer is not None and not force_reload:
            logger.info("Model already loaded")
            return self.model, self.tokenizer
            
        try:
            logger.info(f"Loading tokenizer from {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            logger.info(f"Loading model from {self.model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            logger.info("Model and tokenizer loaded successfully")
            return self.model, self.tokenizer
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
    def get_model_size(self):
        """Get the size of the model on disk."""
        model_path = Path(self.cache_dir) / "models--microsoft--phi-4"
        if model_path.exists():
            size_bytes = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
            size_gb = size_bytes / (1024**3)
            return size_gb
        return 0
        
    def clear_gpu_memory(self):
        """Clear GPU memory if using CUDA."""
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU memory cleared")
                
    def __del__(self):
        """Cleanup when the object is deleted."""
        self.clear_gpu_memory()

# Rename the function to be more specific
def get_phi4_system_info():
    """Get information about the system for running Phi-4."""
    info = {
        "System RAM": f"{psutil.virtual_memory().total / (1024**3):.1f}GB",
        "CPU Cores": psutil.cpu_count(logical=False),
        "CPU Threads": psutil.cpu_count(logical=True),
    }
    
    if torch.cuda.is_available():
        info.update({
            "GPU": torch.cuda.get_device_name(0),
            "GPU Memory": f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB",
            "CUDA Version": torch.version.cuda
        })
    
    return info

# For backward compatibility
ModelLoader = Phi4ModelLoader  # Alias for existing code
get_model_info = get_phi4_system_info  # Alias for existing code