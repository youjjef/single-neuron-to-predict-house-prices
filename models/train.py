"""
Training script for AI Therapist fine-tuning
This script can be run locally or adapted for Google Colab/Kaggle
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import json
import os
import argparse
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TherapistTrainer:
    def __init__(self, config):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.training_dataset = None
        
    def load_data(self):
        """Load and prepare training data"""
        logger.info("Loading training data...")
        
        # Load the conversation data
        with open(self.config['data_path'], 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Format conversations for training
        formatted_texts = []
        for item in data:
            conversation = item['conversation']
            
            # Format as: <|client|> user_message <|therapist|> assistant_message <|endoftext|>
            client_msg = conversation[0]['content']
            therapist_msg = conversation[1]['content']
            
            formatted_text = f"<|client|> {client_msg} <|therapist|> {therapist_msg} <|endoftext|>"
            formatted_texts.append(formatted_text)
        
        logger.info(f"Formatted {len(formatted_texts)} conversations")
        
        # Create dataset
        dataset = Dataset.from_dict({"text": formatted_texts})
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=self.config['max_length'],
                return_overflowing_tokens=False,
            )
        
        self.training_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )
        
        logger.info(f"Tokenized dataset size: {len(self.training_dataset)}")
        
    def setup_model(self):
        """Setup model and tokenizer"""
        logger.info(f"Loading model: {self.config['model_name']}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model_name'],
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Add special tokens
        special_tokens = ["<|client|>", "<|therapist|>"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure quantization if requested
        quantization_config = None
        if self.config['use_quantization'] and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model_name'],
            quantization_config=quantization_config,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Resize token embeddings for new special tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Enable gradient checkpointing to save memory
        self.model.gradient_checkpointing_enable()
        
        logger.info("Model loaded successfully")
        
    def setup_lora(self):
        """Setup LoRA for parameter-efficient fine-tuning"""
        logger.info("Setting up LoRA...")
        
        # Determine target modules based on model architecture
        target_modules = ["c_attn", "c_proj", "c_fc"]  # For DialoGPT
        
        # For other models, you might need different target modules
        if "llama" in self.config['model_name'].lower():
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "mistral" in self.config['model_name'].lower():
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config['lora_r'],
            lora_alpha=self.config['lora_alpha'],
            lora_dropout=self.config['lora_dropout'],
            target_modules=target_modules,
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        logger.info("LoRA setup complete")
        
    def train(self):
        """Train the model"""
        logger.info("Starting training...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config['output_dir'],
            per_device_train_batch_size=self.config['batch_size'],
            gradient_accumulation_steps=self.config['gradient_accumulation_steps'],
            warmup_steps=self.config['warmup_steps'],
            num_train_epochs=self.config['num_epochs'],
            learning_rate=self.config['learning_rate'],
            fp16=torch.cuda.is_available(),
            logging_steps=self.config['logging_steps'],
            save_steps=self.config['save_steps'],
            evaluation_strategy="no",
            save_strategy="steps",
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            report_to=None,  # Disable wandb logging
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.training_dataset,
            data_collator=data_collator,
        )
        
        # Clear cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Train
        trainer.train()
        
        logger.info("Training completed!")
        
        # Save the final model
        self.save_model()
        
    def save_model(self):
        """Save the trained model"""
        save_path = os.path.join(self.config['output_dir'], "final_model")
        
        logger.info(f"Saving model to {save_path}")
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save configuration
        config_path = os.path.join(save_path, "training_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info("Model saved successfully!")
        
    def test_model(self):
        """Test the trained model with sample inputs"""
        logger.info("Testing the trained model...")
        
        test_prompts = [
            "<|client|> I've been feeling really anxious lately <|therapist|>",
            "<|client|> I can't sleep and my mind races <|therapist|>",
            "<|client|> I feel like I'm not good enough <|therapist|>"
        ]
        
        for prompt in test_prompts:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Response: {response}")
            logger.info("-" * 50)

def get_default_config():
    """Get default training configuration"""
    return {
        # Model settings
        'model_name': 'microsoft/DialoGPT-medium',
        'use_quantization': True,
        
        # Data settings
        'data_path': 'data/therapy_training_data.json',
        'max_length': 512,
        
        # Training settings
        'batch_size': 4,
        'gradient_accumulation_steps': 4,
        'num_epochs': 3,
        'learning_rate': 2e-4,
        'warmup_steps': 50,
        'logging_steps': 10,
        'save_steps': 100,
        
        # LoRA settings
        'lora_r': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.05,
        
        # Output settings
        'output_dir': f'./therapy-model-{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    }

def main():
    parser = argparse.ArgumentParser(description="Train AI Therapist model")
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--model_name', type=str, help='Base model name')
    parser.add_argument('--data_path', type=str, help='Path to training data')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--test_only', action='store_true', help='Only test existing model')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = get_default_config()
    
    # Override with command line arguments
    if args.model_name:
        config['model_name'] = args.model_name
    if args.data_path:
        config['data_path'] = args.data_path
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.num_epochs:
        config['num_epochs'] = args.num_epochs
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    
    # Check if data file exists
    if not os.path.exists(config['data_path']):
        logger.error(f"Training data not found at {config['data_path']}")
        logger.info("Please run 'python data/prepare_training_data.py' first")
        return
    
    # Initialize trainer
    trainer = TherapistTrainer(config)
    
    if not args.test_only:
        # Full training pipeline
        trainer.load_data()
        trainer.setup_model()
        trainer.setup_lora()
        trainer.train()
        trainer.test_model()
    else:
        # Test only mode
        trainer.setup_model()
        trainer.test_model()
    
    logger.info("Training script completed!")

if __name__ == "__main__":
    main()