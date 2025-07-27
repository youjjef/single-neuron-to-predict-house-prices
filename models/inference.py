import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    pipeline
)
from peft import PeftModel
import os
from typing import List, Dict, Optional
import json

class TherapistModel:
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-medium",
                 peft_model_path: Optional[str] = None,
                 use_quantization: bool = True,
                 device: str = "auto"):
        """
        Initialize the therapist model for inference
        
        Args:
            model_name (str): Base model name or path
            peft_model_path (str): Path to fine-tuned PEFT model (optional)
            use_quantization (bool): Whether to use 4-bit quantization
            device (str): Device to run on
        """
        self.model_name = model_name
        self.peft_model_path = peft_model_path
        self.use_quantization = use_quantization
        self.device = device
        
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer"""
        try:
            print(f"Loading model: {self.model_name}")
            
            # Configure quantization for memory efficiency
            if self.use_quantization and torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
            else:
                quantization_config = None
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Add pad token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map=self.device if self.device != "auto" else "auto",
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            # Load PEFT model if provided
            if self.peft_model_path and os.path.exists(self.peft_model_path):
                print(f"Loading PEFT model from: {self.peft_model_path}")
                self.model = PeftModel.from_pretrained(self.model, self.peft_model_path)
                self.model = self.model.merge_and_unload()  # Merge for inference
            
            # Create pipeline for easier inference
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map=self.device if self.device != "auto" else "auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to a simpler model
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load a fallback model if the main model fails"""
        try:
            print("Loading fallback model...")
            fallback_model = "microsoft/DialoGPT-small"
            
            self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                fallback_model,
                torch_dtype=torch.float32
            )
            
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer
            )
            
            print("Fallback model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading fallback model: {e}")
            self.pipeline = None
    
    def generate_response(self, 
                         user_input: str, 
                         context: str = "", 
                         conversation_history: List[Dict] = None,
                         max_length: int = 512,
                         temperature: float = 0.7,
                         do_sample: bool = True) -> str:
        """
        Generate a therapeutic response
        
        Args:
            user_input (str): User's message
            context (str): RAG context
            conversation_history (list): Previous conversation
            max_length (int): Maximum response length
            temperature (float): Sampling temperature
            do_sample (bool): Whether to use sampling
            
        Returns:
            str: Generated therapeutic response
        """
        if self.pipeline is None:
            return self._get_rule_based_response(user_input)
        
        try:
            # Construct prompt
            prompt = self._construct_prompt(user_input, context, conversation_history)
            
            # Generate response
            generated = self.pipeline(
                prompt,
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                length_penalty=1.0
            )
            
            # Extract and clean response
            response = generated[0]['generated_text']
            response = self._clean_response(response, prompt)
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return self._get_rule_based_response(user_input)
    
    def _construct_prompt(self, 
                         user_input: str, 
                         context: str = "", 
                         conversation_history: List[Dict] = None) -> str:
        """
        Construct a prompt for the model
        
        Args:
            user_input (str): Current user input
            context (str): RAG context
            conversation_history (list): Previous messages
            
        Returns:
            str: Formatted prompt
        """
        # System prompt for therapy
        system_prompt = """You are a compassionate AI therapist. Your role is to:
- Provide emotional support and validation
- Ask thoughtful, open-ended questions
- Offer evidence-based therapeutic techniques
- Maintain appropriate boundaries
- Encourage self-reflection and growth
- Never provide medical advice or diagnoses

Remember to be empathetic, non-judgmental, and professional."""
        
        # Add context from RAG if available
        context_section = ""
        if context.strip():
            context_section = f"\n\nRelevant therapeutic knowledge:\n{context}"
        
        # Add conversation history
        history_section = ""
        if conversation_history:
            recent_history = conversation_history[-3:]  # Last 3 exchanges
            history_items = []
            for msg in recent_history:
                role = "Client" if msg.get("role") == "user" else "Therapist"
                content = msg.get("content", "")
                history_items.append(f"{role}: {content}")
            
            if history_items:
                history_section = f"\n\nRecent conversation:\n" + "\n".join(history_items)
        
        # Construct full prompt
        prompt = f"""{system_prompt}{context_section}{history_section}

Client: {user_input}
Therapist:"""
        
        return prompt
    
    def _clean_response(self, response: str, prompt: str) -> str:
        """
        Clean and format the generated response
        
        Args:
            response (str): Raw generated response
            prompt (str): Original prompt
            
        Returns:
            str: Cleaned response
        """
        # Remove the prompt from the response
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        # Remove common prefixes
        prefixes_to_remove = ["Therapist:", "AI:", "Assistant:", "Response:"]
        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        # Split by common delimiters and take the first part
        delimiters = ["\nClient:", "\nUser:", "\nHuman:", "\n\n\n"]
        for delimiter in delimiters:
            if delimiter in response:
                response = response.split(delimiter)[0].strip()
        
        # Ensure response isn't too short or empty
        if len(response.strip()) < 10:
            return self._get_rule_based_response("")
        
        # Limit response length
        if len(response) > 1000:
            sentences = response.split('. ')
            response = '. '.join(sentences[:3]) + '.'
        
        return response.strip()
    
    def _get_rule_based_response(self, user_input: str) -> str:
        """
        Generate a rule-based response as fallback
        
        Args:
            user_input (str): User's input
            
        Returns:
            str: Rule-based therapeutic response
        """
        user_lower = user_input.lower()
        
        # Emotional keywords and responses
        if any(word in user_lower for word in ['sad', 'depressed', 'down', 'low']):
            return "I hear that you're feeling sad right now. That sounds really difficult. Can you tell me more about what's contributing to these feelings? Remember that it's okay to feel this way, and I'm here to support you."
        
        elif any(word in user_lower for word in ['anxious', 'worried', 'nervous', 'panic']):
            return "It sounds like you're experiencing some anxiety. That can feel overwhelming. Let's try to slow down for a moment. Can you take a deep breath with me? What specific thoughts or situations are triggering these anxious feelings?"
        
        elif any(word in user_lower for word in ['angry', 'mad', 'frustrated', 'upset']):
            return "I can sense your frustration, and those feelings are completely valid. Anger often tells us that something important to us has been affected. What do you think might be underneath this anger? What need or value feels threatened right now?"
        
        elif any(word in user_lower for word in ['stress', 'overwhelmed', 'pressure']):
            return "Feeling overwhelmed can be exhausting. It sounds like you have a lot on your plate right now. Sometimes when we're stressed, it helps to break things down into smaller, more manageable pieces. What feels most pressing to you right now?"
        
        elif any(word in user_lower for word in ['alone', 'lonely', 'isolated']):
            return "Loneliness can be one of the most painful emotions. Thank you for sharing that with me - it takes courage to express those feelings. You're not alone in this conversation right now. Can you tell me about your support system?"
        
        elif any(word in user_lower for word in ['help', 'support', 'advice']):
            return "I'm glad you're reaching out for support - that shows real strength and self-awareness. I'm here to listen and help you explore your thoughts and feelings. What specific area would you like to focus on today?"
        
        elif any(word in user_lower for word in ['thank', 'grateful', 'appreciate']):
            return "I'm touched that you feel that way. It's wonderful that you can recognize and express gratitude - that's actually a powerful tool for mental well-being. How are you feeling overall today?"
        
        elif len(user_input.strip()) == 0:
            return "I notice you might be taking a moment to gather your thoughts. That's perfectly okay. Take your time. What's on your mind today?"
        
        else:
            return "Thank you for sharing that with me. I'm here to listen and support you. Can you tell me more about what you're experiencing? What feels most important for us to talk about right now?"
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "peft_model_path": self.peft_model_path,
            "use_quantization": self.use_quantization,
            "device": self.device,
            "model_loaded": self.model is not None,
            "pipeline_available": self.pipeline is not None
        }