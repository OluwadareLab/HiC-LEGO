import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
import time
from scipy.stats import spearmanr

from normalization import preprocess_hic_file, kr_normalization
from embeddings import generate_embeddings
from utils import prepare_tensors, contact_to_distance, write_pdb, set_seed, resolve_torch_device
from models import UniversalHiCGNN

def train_hic_gnn(filepath, device_name="cuda"):
    # --- Start Timer ---
    start_time = time.time()

    # --- 0. Setup Directories ---
    filename = os.path.splitext(os.path.basename(filepath))[0]

    if not os.path.exists('Outputs'):
        os.makedirs('Outputs')
        print(" -> Created 'Outputs/' directory.")
    # if not os.path.exists('Data'):   # <--- ADD THIS
    #     os.makedirs('Data')
    #     print(" -> Created 'Data/' directory.")        
        # filename = os.path.splitext(os.path.basename(filepath))[0]

    # --- 1. Setup Device ---
    device = resolve_torch_device(device_name if device_name else "cuda")
    print(f"Running on: {device}")
    
    # --- 2. Load & Normalize Data ---
    print(" -> Loading Data...")
    raw_matrix, _ = preprocess_hic_file(filepath)
    np.fill_diagonal(raw_matrix, 0)
    
    print(" -> Normalizing (KR)...")
    norm_matrix = kr_normalization(raw_matrix)
    
    # --- 3. Generate Embeddings ---
    embeddings = generate_embeddings(raw_matrix, device=device)
    # SAVE THE EMBEDDINGS IMMEDIATELY AFTER GENERATION
    # embedding_save_path = f"Data/{filename}_domain_interactions_embeddings.txt"
    # np.savetxt(embedding_save_path, embeddings)           
    # print(f" -> Saved embeddings to {embedding_save_path}") 

    # --- 4. Prepare Tensors ---
    features, adj_tensor = prepare_tensors(norm_matrix, embeddings, device)
    
    # --- 5. Training Loop ---
    conversions = np.arange(0.1, 2.0, 0.1)
    
    best_score = -1
    best_loss = float('inf')
    best_struct = None
    best_model_state = None
    best_alpha = None
    
    print(" -> Starting Training...")
       
    # Auto-detect input dimension
    input_dim = embeddings.shape[1]

    for alpha in conversions:
        # Seeding for reproducibility
        set_seed(42)

        model = UniversalHiCGNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Ground Truth
        truth = contact_to_distance(raw_matrix, alpha).to(device)
        
        prev_loss = float('inf')
        
        model.train()
        for epoch in range(1000): # Max epochs
            optimizer.zero_grad()
            output_dist = model(features, adj_tensor)
            
            loss = criterion(output_dist, truth)
            loss.backward()
            optimizer.step()
            
            if abs(prev_loss - loss.item()) < 1e-8:
                break
            prev_loss = loss.item()
            
        # Evaluation
        model.eval()
        with torch.no_grad():
            struct = model.get_structure(features, adj_tensor)
            dist_out = torch.cdist(struct, struct).cpu().numpy()
            dist_truth = truth.cpu().numpy()
            
            idx = np.triu_indices(dist_truth.shape[0], k=1)
            score = spearmanr(dist_truth[idx], dist_out[idx])[0]
            
            print(f"   Alpha: {alpha:.1f} | Loss: {loss.item():.6f} | dSCC: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_loss = loss.item()
                best_struct = struct.cpu().numpy()
                best_model_state = model.state_dict()
                best_alpha = alpha
    
    # --- Stop Timer ---
    end_time = time.time() 
    total_time = end_time - start_time

    # --- 6. Save Results ---
    print(f"\nFinal Result -> Best dSCC: {best_score:.4f} (Alpha: {best_alpha:.1f})")
    print(f"Total time sec: {total_time:.4f}")
    
    # A. Save Embeddings (The part you were asking about)
    embedding_path = f"Outputs/{filename}_embeddings.txt"
    np.savetxt(embedding_path, embeddings)
    print(f" -> Saved embeddings to {embedding_path}")

    # Save Weights
    weight_path = f"Outputs/{filename}_weights.pt"
    torch.save(best_model_state, weight_path)
    print(f" -> Saved weights to {weight_path}")
    
    # Save Structure (PDB)
    pdb_path = f"Outputs/{filename}_structure.pdb"
    write_pdb(best_struct, pdb_path)
    print(f" -> Saved structure to {pdb_path}")

    # Save Log File (inside Output folder)
    log_path = f"Outputs/{filename}_log.txt"
    with open(log_path, "w") as f:
        f.write(f"Optimal conversion factor: {best_alpha}\n")
        f.write(f"Optimal dSCC: {best_score}\n")
        f.write(f"Final MSE loss: {best_loss}\n")
    print(f" -> Saved logs to {log_path}")
    
    return best_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to input file")
    args = parser.parse_args()
    
    train_hic_gnn(args.file)
