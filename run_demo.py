#!/usr/bin/env python3
"""
Demo script for AI Therapist
Run a simple command-line demo to test the system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag.rag_system import RAGTherapist
from models.inference import TherapistModel

def main():
    print("🧠 AI Therapist Demo")
    print("=" * 30)
    print("This is a simple demo to test the AI therapist system.")
    print("Type 'quit' to exit, 'help' for commands.")
    print()
    
    # Initialize systems
    print("Loading AI Therapist...")
    try:
        rag_system = RAGTherapist()
        model = TherapistModel()
        print("✅ AI Therapist loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading AI Therapist: {e}")
        print("💡 Try running: python setup.py")
        return
    
    print("\n" + "="*50)
    print("💬 Chat with the AI Therapist")
    print("="*50)
    print("AI Therapist: Hello! I'm here to listen and support you.")
    print("              What would you like to talk about today?")
    
    conversation_history = []
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nAI Therapist: Take care of yourself. Remember, seeking support is a sign of strength.")
                break
                
            if user_input.lower() == 'help':
                print("\nCommands:")
                print("  quit/exit/bye - End the conversation")
                print("  help - Show this help")
                print("  clear - Clear conversation history")
                continue
                
            if user_input.lower() == 'clear':
                conversation_history = []
                print("\n✅ Conversation history cleared.")
                continue
            
            # Get RAG context
            try:
                rag_context = rag_system.get_relevant_context(user_input)
            except Exception as e:
                print(f"⚠️ RAG system error: {e}")
                rag_context = ""
            
            # Generate response
            try:
                response = model.generate_response(
                    user_input,
                    context=rag_context,
                    conversation_history=conversation_history[-5:]  # Last 5 exchanges
                )
            except Exception as e:
                print(f"⚠️ Model error: {e}")
                response = "I'm sorry, I'm having some technical difficulties right now. Can you try rephrasing that?"
            
            # Add to conversation history
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "therapist", "content": response})
            
            # Display response
            print(f"\nAI Therapist: {response}")
            
        except KeyboardInterrupt:
            print("\n\nAI Therapist: I understand you need to go. Take care!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            print("Please try again or type 'quit' to exit.")

if __name__ == "__main__":
    main()