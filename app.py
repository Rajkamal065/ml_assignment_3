import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os

# Define the directory where models are stored
MODEL_DIR = "models"

# ============================================
# 1. MODEL CLASS
# ============================================
class WordPredictorMLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, block_size, dropout_rate, padding_idx):
        super(WordPredictorMLP, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        
        self.fc1 = nn.Linear(block_size * embedding_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x); x = self.relu1(x); x = self.dropout1(x)
        x = self.fc2(x); x = self.relu2(x); x = self.dropout2(x)
        x = self.fc3(x)
        return x

# ============================================
# 2. TEXT GENERATION FUNCTION
# ============================================
def generate_text(model, seed_text, num_words_to_gen, block_size, word_to_idx, idx_to_word, temperature=1.0, seed=42, device='cpu'):
    torch.manual_seed(seed)
    model.eval()
    
    padding_idx = 0 
    unknown_idx = 1
    
    # Split and lowercase the seed text
    words = seed_text.strip().lower().split()
    generated_text = words[:]
    
    # Get the last block_size words as context
    context = words[-block_size:] if len(words) >= block_size else words
    
    # Convert context to indices
    if len(context) < block_size:
        pad_indices = [padding_idx] * (block_size - len(context))
        context_idx = pad_indices + [word_to_idx.get(w, unknown_idx) for w in context]
    else:
        context_idx = [word_to_idx.get(w, unknown_idx) for w in context]

    with torch.no_grad():
        for _ in range(num_words_to_gen):
            input_tensor = torch.tensor([context_idx], dtype=torch.long).to(device)
            logits = model(input_tensor)
            
            # Apply temperature
            logits_scaled = logits / temperature
            probs = F.softmax(logits_scaled, dim=1)
            
            # Sample from the distribution
            next_idx = torch.multinomial(probs, num_samples=1).item()
            
            # Get the word from vocabulary
            next_word = idx_to_word.get(next_idx, '<UNK>')
            
            # Skip if it's padding or unknown
            if next_word not in ['<PAD>', '.']:
                generated_text.append(next_word)
            else:
                generated_text.append(next_word)
            
            # Update context (sliding window)
            context_idx = context_idx[1:] + [next_idx]

    return ' '.join(generated_text)

# ============================================
# 3. LOAD VOCABULARY
# ============================================
@st.cache_data
def load_vocab(dataset_prefix):
    vocab_path = os.path.join(MODEL_DIR, f"{dataset_prefix}_vocab.json")
    if os.path.exists(vocab_path):
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
        
        # Convert string keys to integers for idx_to_word
        idx_to_word = {}
        for k, v in vocab_data['idx_to_word'].items():
            idx_to_word[int(k)] = v
        
        return vocab_data['word_to_idx'], idx_to_word
    return None, None

# ============================================
# 4. LOAD MODELS
# ============================================
@st.cache_resource
def load_models(dataset_prefix, vocab_size):
    models = {}
    device = torch.device('cpu')
    
    # Fixed hyperparameters (must match training)
    FIXED_CONFIG = {
        'vocab_size': vocab_size,
        'embedding_dim': 64,
        'hidden_dim': 1024,
        'block_size': 5,
        'dropout_rate': 0.4,
        'padding_idx': 0
    }
    
    model_files = ['underfit', 'good_fit', 'overfit']
    
    for file_type in model_files:
        model_name = f"{dataset_prefix}_{file_type}.pth"
        model_path = os.path.join(MODEL_DIR, model_name)
        
        if os.path.exists(model_path):
            try:
                model = WordPredictorMLP(**FIXED_CONFIG).to(device)
                
                # Load state dict
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict)
                model.eval()
                
                models[file_type] = model
                st.success(f" Loaded {model_name}")
            except Exception as e:
                st.error(f"Error loading {model_name}: {str(e)}")
        else:
            st.warning(f"Model file not found: {model_name}")
            
    return models

# ============================================
# 5. STREAMLIT UI
# ============================================
st.set_page_config(layout="wide", page_title="Next-Word Predictor MLP")
st.title(" Next-Word Predictor MLP (Section 1.4)")

# --- Dataset Selector ---
st.header("1️ Choose Dataset")
dataset_map = {
    "Category I: Natural Language": "nl",
    "Category II: Structured Code": "code"
}
dataset_choice = st.selectbox("Select Model Source:", list(dataset_map.keys()))
dataset_prefix = dataset_map[dataset_choice]

# --- Load Assets ---
word_to_idx, idx_to_word = load_vocab(dataset_prefix)

if not word_to_idx:
    st.error(f" Could not load {dataset_prefix}_vocab.json from the '{MODEL_DIR}' folder.")
    st.stop()
else:
    vocab_size = len(word_to_idx)
    st.info(f" Vocabulary size: {vocab_size} words")
    
    models = load_models(dataset_prefix, vocab_size)
    
    if not models:
        st.error(f" No models found for {dataset_prefix}. Make sure .pth files are in the '{MODEL_DIR}' folder.")
        st.stop()
    
    # --- Sidebar Controls ---
    st.sidebar.header(" Generation Controls")
    
    model_choice_name = st.sidebar.selectbox(
        " Choose Model Checkpoint:",
        list(models.keys()),
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    temperature = st.sidebar.slider(
        " Temperature (Randomness):",
        min_value=0.1, max_value=2.0, value=0.8, step=0.1,
        help="Lower = more predictable, Higher = more random"
    )
    
    num_words = st.sidebar.slider(
        " Words to Generate:",
        min_value=10, max_value=200, value=50, step=10
    )
    
    seed = st.sidebar.number_input(" Random Seed", value=42, min_value=0)
    
    # --- Display Model Info ---
    st.sidebar.header(" Model Hyperparameters")
    st.sidebar.json({
        "Context Length (Block Size)": 5,
        "Embedding Dimension": 64,
        "Hidden Dimension": 1024,
        "Activation Function": "ReLU",
        "Dropout Rate": 0.4
    })

    # --- Main App Area ---
    st.header("2 Input and Output")
    
    # Default text based on dataset
    if dataset_prefix == "nl":
        default_text = "sherlock holmes was a"
    else:
        default_text = "if ( x > 0 ) {"
    
    seed_text = st.text_area(
        "Enter your seed text:", 
        value=default_text,
        height=100,
        help="Type any text to start generation"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        generate_button = st.button(" Generate Text", type="primary")
    
    if generate_button:
        if not seed_text.strip():
            st.error(" Please enter some seed text!")
        elif model_choice_name not in models:
            st.error(f" Selected model '{model_choice_name}' was not loaded.")
        else:
            with st.spinner(f"Generating text using '{model_choice_name}' model..."):
                model_to_use = models[model_choice_name]
                
                # Generate text
                output_text = generate_text(
                    model=model_to_use,
                    seed_text=seed_text,
                    num_words_to_gen=num_words,
                    block_size=5,
                    word_to_idx=word_to_idx,
                    idx_to_word=idx_to_word,
                    temperature=temperature,
                    seed=seed,
                    device='cpu'
                )
                
                # Display results
                st.subheader(" Generated Sequence")
                
                # Highlight seed text
                seed_len = len(seed_text)
                generated_part = output_text[seed_len:]
                
                st.markdown(f"""
                <div style='padding: 20px; background-color: #f0f2f6; border-radius: 10px; border-left: 5px solid #4CAF50;'>
                    <span style='background-color: #ffeb3b; padding: 2px 5px; border-radius: 3px;'>{seed_text}</span><span style='color: #1976D2;'>{generated_part}</span>
                </div>
                """, unsafe_allow_html=True)
                
                st.info(f"ℹ Any word in your seed text not in the {vocab_size}-word vocabulary was mapped to '<UNK>' token.")
                
                # Display generation statistics
                st.sidebar.subheader("Generation Stats")
                st.sidebar.metric("Seed Words", len(seed_text.split()))
                st.sidebar.metric("Generated Words", num_words)
                st.sidebar.metric("Total Words", len(output_text.split()))

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p> <strong>Tip:</strong> Try different temperatures to see how it affects creativity!</p>
    <p>Lower temperature (0.1-0.5): More focused and deterministic</p>
    <p>Higher temperature (1.0-2.0): More random and creative</p>
</div>
""", unsafe_allow_html=True)