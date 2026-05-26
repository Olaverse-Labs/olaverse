import os

class LegalPeace:
    """
    Interface for the LegalPeace model family (Beta).
    Base Model: Mistral-7B-v0.3 (via unsloth 4-bit quantization).
    Fine-tuned for Contract Analysis & Legal Reasoning.

    Warning:
        This is a beta model. Outputs should always be reviewed by a qualified legal professional.
        Not recommended for production use.
    """
    def __init__(self, model_name="olaverse/legal-peace-v1.0", max_seq_length=2048, load_in_4bit=True):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.tokenizer = None
        self._loaded = False
        
    def load(self):
        """
        Load the model and tokenizer using unsloth.
        """
        if self._loaded:
            return
            
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            raise ImportError(
                "The 'unsloth' library is required to load LegalPeace. "
                "Please install it using: pip install unsloth"
            )
            
        # Initialize model and tokenizer
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=self.load_in_4bit,
        )
        FastLanguageModel.for_inference(self.model)
        self._loaded = True
        
    def generate(self, prompt, max_new_tokens=300, temperature=0.7, **kwargs):
        """
        Generate legal analysis or reasoning for a prompt.
        
        Args:
            prompt (str): Prompt or clause to analyze.
            max_new_tokens (int): Maximum new tokens to generate.
            temperature (float): Temperature for generation.
            
        Returns:
            str: Decoded generation response.
        """
        if not self._loaded:
            self.load()
            
        import torch
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
            
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            temperature=temperature, 
            **kwargs
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
