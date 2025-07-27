import chromadb
import os
import json
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import numpy as np

class RAGTherapist:
    def __init__(self, persist_directory="./rag_db", model_name="all-MiniLM-L6-v2"):
        """
        Initialize the RAG system for therapy knowledge base
        
        Args:
            persist_directory (str): Directory to persist the ChromaDB
            model_name (str): Sentence transformer model name
        """
        self.persist_directory = persist_directory
        self.model_name = model_name
        
        # Initialize embedding model
        self.embedder = SentenceTransformer(model_name)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Create or get collection
        self.collection_name = "therapy_knowledge"
        try:
            self.collection = self.client.get_collection(self.collection_name)
            print(f"Loaded existing collection: {self.collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Therapy knowledge base for RAG"}
            )
            print(f"Created new collection: {self.collection_name}")
            self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """Initialize the knowledge base with therapy-related content"""
        
        therapy_knowledge = [
            {
                "id": "cbt_basics",
                "content": "Cognitive Behavioral Therapy (CBT) focuses on identifying and changing negative thought patterns and behaviors. It helps people understand the connection between thoughts, feelings, and behaviors.",
                "category": "CBT",
                "techniques": ["thought challenging", "behavioral activation", "cognitive restructuring"]
            },
            {
                "id": "mindfulness_intro",
                "content": "Mindfulness is the practice of being present and fully engaged with whatever we're doing at the moment. It involves observing thoughts and feelings without judgment.",
                "category": "Mindfulness",
                "techniques": ["breathing exercises", "body scan", "present moment awareness"]
            },
            {
                "id": "anxiety_management",
                "content": "Anxiety can be managed through various techniques including deep breathing, progressive muscle relaxation, grounding techniques, and gradual exposure to feared situations.",
                "category": "Anxiety",
                "techniques": ["deep breathing", "progressive muscle relaxation", "grounding", "exposure therapy"]
            },
            {
                "id": "depression_support",
                "content": "Depression often involves negative thinking patterns, social withdrawal, and loss of interest in activities. Treatment includes behavioral activation, social connection, and addressing negative thought patterns.",
                "category": "Depression",
                "techniques": ["behavioral activation", "social engagement", "mood monitoring", "pleasant activity scheduling"]
            },
            {
                "id": "stress_management",
                "content": "Stress management involves identifying stressors, developing coping strategies, practicing relaxation techniques, and maintaining work-life balance.",
                "category": "Stress",
                "techniques": ["time management", "relaxation", "problem-solving", "boundary setting"]
            },
            {
                "id": "emotional_regulation",
                "content": "Emotional regulation involves understanding emotions, identifying triggers, and using healthy coping strategies to manage intense feelings.",
                "category": "Emotional Regulation",
                "techniques": ["emotion identification", "distress tolerance", "healthy coping", "self-soothing"]
            },
            {
                "id": "self_care",
                "content": "Self-care includes activities that promote physical, mental, and emotional well-being. It's essential for maintaining mental health and preventing burnout.",
                "category": "Self-Care",
                "techniques": ["exercise", "sleep hygiene", "nutrition", "social connection", "hobbies"]
            },
            {
                "id": "therapeutic_relationship",
                "content": "A strong therapeutic relationship is built on trust, empathy, unconditional positive regard, and genuine understanding. It provides a safe space for healing and growth.",
                "category": "Therapeutic Process",
                "techniques": ["active listening", "empathy", "validation", "unconditional positive regard"]
            },
            {
                "id": "grounding_techniques",
                "content": "Grounding techniques help bring attention to the present moment and can be helpful during anxiety, panic, or dissociation. Examples include the 5-4-3-2-1 technique and deep breathing.",
                "category": "Grounding",
                "techniques": ["5-4-3-2-1 technique", "deep breathing", "physical grounding", "mental grounding"]
            },
            {
                "id": "cognitive_distortions",
                "content": "Cognitive distortions are irrational or exaggerated thought patterns that can worsen anxiety and depression. Common ones include all-or-nothing thinking, catastrophizing, and mind reading.",
                "category": "CBT",
                "techniques": ["thought challenging", "evidence examination", "balanced thinking", "cognitive restructuring"]
            }
        ]
        
        # Add documents to collection
        for doc in therapy_knowledge:
            self.add_document(
                document_id=doc["id"],
                content=doc["content"],
                metadata={
                    "category": doc["category"],
                    "techniques": doc["techniques"]
                }
            )
        
        print(f"Initialized knowledge base with {len(therapy_knowledge)} documents")
    
    def add_document(self, document_id: str, content: str, metadata: Dict[str, Any] = None):
        """
        Add a document to the knowledge base
        
        Args:
            document_id (str): Unique identifier for the document
            content (str): Text content of the document
            metadata (dict): Additional metadata for the document
        """
        # Generate embedding
        embedding = self.embedder.encode(content).tolist()
        
        # Add to collection
        self.collection.add(
            documents=[content],
            embeddings=[embedding],
            metadatas=[metadata or {}],
            ids=[document_id]
        )
    
    def get_relevant_context(self, query: str, n_results: int = 3) -> str:
        """
        Retrieve relevant context for a given query
        
        Args:
            query (str): User's input/question
            n_results (int): Number of relevant documents to retrieve
            
        Returns:
            str: Concatenated relevant context
        """
        try:
            # Generate query embedding
            query_embedding = self.embedder.encode(query).tolist()
            
            # Search for relevant documents
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # Extract and format context
            contexts = []
            if results and results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    category = metadata.get('category', 'General')
                    
                    context_item = f"[{category}] {doc}"
                    contexts.append(context_item)
            
            return "\n\n".join(contexts) if contexts else ""
            
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return ""
    
    def add_therapy_resource(self, title: str, content: str, category: str, techniques: List[str] = None):
        """
        Add a new therapy resource to the knowledge base
        
        Args:
            title (str): Title/ID of the resource
            content (str): Content of the resource
            category (str): Category (e.g., CBT, Mindfulness, etc.)
            techniques (list): List of associated techniques
        """
        metadata = {
            "category": category,
            "techniques": techniques or []
        }
        
        self.add_document(
            document_id=title.lower().replace(" ", "_"),
            content=content,
            metadata=metadata
        )
    
    def search_by_category(self, category: str, n_results: int = 5) -> List[Dict]:
        """
        Search documents by category
        
        Args:
            category (str): Category to search for
            n_results (int): Maximum number of results
            
        Returns:
            list: List of matching documents with metadata
        """
        try:
            # Get all documents (this is a limitation of ChromaDB - no direct metadata filtering in query)
            all_results = self.collection.get()
            
            # Filter by category
            filtered_docs = []
            for i, metadata in enumerate(all_results['metadatas']):
                if metadata.get('category', '').lower() == category.lower():
                    filtered_docs.append({
                        'id': all_results['ids'][i],
                        'content': all_results['documents'][i],
                        'metadata': metadata
                    })
                    
                    if len(filtered_docs) >= n_results:
                        break
            
            return filtered_docs
            
        except Exception as e:
            print(f"Error searching by category: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        try:
            count = self.collection.count()
            
            # Get sample of documents to analyze categories
            sample_results = self.collection.get()
            categories = set()
            
            for metadata in sample_results['metadatas']:
                if 'category' in metadata:
                    categories.add(metadata['category'])
            
            return {
                "total_documents": count,
                "categories": list(categories),
                "embedding_model": self.model_name
            }
            
        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return {"error": str(e)}