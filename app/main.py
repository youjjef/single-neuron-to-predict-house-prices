import streamlit as st
import os
import sys
from datetime import datetime
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.rag_system import RAGTherapist
from models.inference import TherapistModel

# Page config
st.set_page_config(
    page_title="AI Therapist",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #4A90E2;
        font-size: 3rem;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
    }
    .user-message {
        background-color: #E3F2FD;
        margin-left: 2rem;
    }
    .therapist-message {
        background-color: #F1F8E9;
        margin-right: 2rem;
    }
    .sidebar-content {
        background-color: #FAFAFA;
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if "therapist" not in st.session_state:
        st.session_state.therapist = None

def load_therapist_model():
    """Load the therapist model with caching"""
    if st.session_state.therapist is None:
        try:
            with st.spinner("Loading AI Therapist... This may take a moment."):
                # Initialize RAG system
                rag_system = RAGTherapist()
                
                # Initialize model
                model = TherapistModel()
                
                # Combine both systems
                st.session_state.therapist = {
                    'rag': rag_system,
                    'model': model
                }
                st.success("AI Therapist loaded successfully!")
        except Exception as e:
            st.error(f"Error loading AI Therapist: {str(e)}")
            return False
    return True

def get_therapy_response(user_input):
    """Get response from the AI therapist"""
    try:
        # Get RAG context
        rag_context = st.session_state.therapist['rag'].get_relevant_context(user_input)
        
        # Generate response using fine-tuned model
        response = st.session_state.therapist['model'].generate_response(
            user_input, 
            context=rag_context,
            conversation_history=st.session_state.messages[-5:]  # Last 5 messages for context
        )
        
        return response
    except Exception as e:
        return f"I apologize, but I'm having technical difficulties. Please try again. Error: {str(e)}"

def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 AI Therapist</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.header("Session Information")
        st.write(f"**Session ID:** {st.session_state.session_id}")
        st.write(f"**Messages:** {len(st.session_state.messages)}")
        
        st.header("Disclaimer")
        st.warning("""
        ⚠️ **Important Notice**
        
        This AI therapist is for educational and support purposes only. 
        It is not a replacement for professional mental health care.
        
        If you're experiencing a mental health crisis, please contact:
        - Emergency services: 911
        - Crisis Text Line: Text HOME to 741741
        - National Suicide Prevention Lifeline: 988
        """)
        
        if st.button("Clear Conversation"):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("Download Session"):
            session_data = {
                "session_id": st.session_state.session_id,
                "messages": st.session_state.messages,
                "timestamp": datetime.now().isoformat()
            }
            st.download_button(
                label="Download JSON",
                data=json.dumps(session_data, indent=2),
                file_name=f"therapy_session_{st.session_state.session_id}.json",
                mime="application/json"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content area
    col1, col2, col3 = st.columns([1, 8, 1])
    
    with col2:
        # Load model
        if not load_therapist_model():
            st.stop()
        
        # Introduction message
        if not st.session_state.messages:
            intro_message = """
            Hello! I'm your AI therapist. I'm here to provide a safe, non-judgmental space for you to express your thoughts and feelings.
            
            I can help you with:
            - Processing emotions and thoughts
            - Developing coping strategies
            - Exploring patterns in your thinking
            - Providing psychoeducational information
            - Offering mindfulness and relaxation techniques
            
            How are you feeling today? What would you like to talk about?
            """
            st.session_state.messages.append({
                "role": "therapist",
                "content": intro_message,
                "timestamp": datetime.now().isoformat()
            })
        
        # Display conversation history
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message therapist-message">
                    <strong>AI Therapist:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
        
        # Chat input
        user_input = st.chat_input("Type your message here...")
        
        if user_input:
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat()
            })
            
            # Get therapist response
            with st.spinner("Thinking..."):
                therapist_response = get_therapy_response(user_input)
            
            # Add therapist response
            st.session_state.messages.append({
                "role": "therapist",
                "content": therapist_response,
                "timestamp": datetime.now().isoformat()
            })
            
            # Rerun to update the display
            st.rerun()

if __name__ == "__main__":
    main()