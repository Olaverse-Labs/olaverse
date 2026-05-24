import os

class LIDNeural5:
    """
    Interface for the LIDNeural5 transformer-based language detection model.
    Base Model: microsoft/Multilingual-MiniLM-L12-H384
    Fine-tuned on 5 languages: Yoruba ('yor'), Hausa ('hau'), Igbo ('ibo'), Pidgin ('pcm'), and English ('eng').
    """
    def __init__(self, model_name="olaverse/lid-neural-5"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._loaded = False
        self.classes = ['eng', 'hau', 'ibo', 'pcm', 'yor']
        
    def load(self):
        """
        Load the model and tokenizer using transformers.
        """
        if self._loaded:
            return
            
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
        except ImportError:
            raise ImportError(
                "The 'transformers' and 'torch' libraries are required to load LIDNeural5. "
                "Please install them using: pip install transformers torch"
            )
            
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        
        # Read class labels from model config if populated
        if hasattr(self.model.config, "id2label") and self.model.config.id2label:
            # Sort keys to align classes
            id2lbl = self.model.config.id2label
            self.classes = [id2lbl[i] if i in id2lbl else id2lbl[str(i)] for i in range(len(id2lbl))]
            
        self.model.eval()
        self._loaded = True
        
    def predict_proba(self, text):
        """
        Predict the language probabilities for the text.
        """
        if not self._loaded:
            self.load()
            
        import torch
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).squeeze().tolist()
            
        if not isinstance(probs, list):
            probs = [probs]
            
        return {self.classes[i]: probs[i] for i in range(len(self.classes))}
        
    def predict(self, text):
        """
        Predict the language of the text.
        Returns: 'yor', 'hau', 'ibo', 'pcm', or 'eng'.
        """
        probs = self.predict_proba(text)
        return max(probs, key=probs.get)
