from sentence_transformers import SentenceTransformer
import json

# Load a pre-trained SBERT model
SBERT_embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Get sentence embeddings
embeddings = SBERT_embedding_model.encode("\"I'm a perfectionist by nature, and this product's attention to detail was music to my ears. The sturdy construction and seamless functionality made it a joy to use, and I must say, it's given me a newfound sense of confidence in my ability to tackle even the most daunting tasks. 5 stars!\"")
