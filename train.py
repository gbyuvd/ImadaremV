#!/usr/bin/env python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import random
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# ====================================================
# 1. Setup 
# ====================================================
seed = 2025
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ====================================================
# 2. Dataset Class
# ====================================================
class TokenDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len=90):
        self.texts = texts
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        toks = self.tokenizer.encode(self.texts[idx])
        actual_length = len(toks)
        
        toks = toks[:self.seq_len]
        if len(toks) < self.seq_len:
            toks += [self.tokenizer.pad_token_id] * (self.seq_len - len(toks))
        
        return torch.tensor(toks, dtype=torch.long), torch.tensor(min(actual_length, self.seq_len), dtype=torch.long)

# ====================================================
# 3. Training Function
# ====================================================
def train_model_z(model, train_loader, test_loader, epochs=10, lr=1e-4):
    from ranger21 import Ranger21
    from ranger21.chebyshev_lr_functions import ChebyshevLR

    optimizer = Ranger21(
        model.parameters(),
        lr=lr,
        weight_decay=0.01,
        use_adabelief=True,
        use_madgrad=True,
        num_epochs=epochs,
        using_gc=True,
        num_batches_per_epoch=len(train_loader)
    )

    total_steps = epochs * len(train_loader)
    scheduler = ChebyshevLR(
        optimizer,
        total_steps=total_steps,
        lr_start=lr,
        lr_end=lr * 0.1,
        min_lr=lr * 0.05
    )

    history = {
        "train_loss": [], "train_recon": [], "train_diversity": [],
        "val_loss": [], "val_recon": [], "val_accuracy": [],
        "learning_rate": []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # ========== Training ==========
        model.train()
        epoch_loss = epoch_recon = epoch_div = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch, lengths in pbar:
            batch, lengths = batch.to(device), lengths.to(device)
            
            optimizer.zero_grad()
            losses = model.loss(batch, lengths=lengths)
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += losses['total'].item()
            epoch_recon += losses['recon'].item()
            epoch_div += losses['diversity'].item()
            
            pbar.set_postfix({
                'loss': f"{losses['total']:.4f}",
                'recon': f"{losses['recon']:.4f}",
                'div': f"{losses['diversity']:.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
        
        # Log averages
        history["train_loss"].append(epoch_loss / len(train_loader))
        history["train_recon"].append(epoch_recon / len(train_loader))
        history["train_diversity"].append(epoch_div / len(train_loader))
        history["learning_rate"].append(scheduler.get_last_lr()[0])
        
        # ========== Validation ==========
        model.eval()
        val_loss = val_recon = val_acc = 0
        acc_batches = 0
        
        with torch.no_grad():
            for batch, lengths in tqdm(test_loader, desc="Validating", leave=False):
                batch, lengths = batch.to(device), lengths.to(device)
                losses = model.loss(batch, lengths=lengths)
                val_loss += losses['total'].item()
                val_recon += losses['recon'].item()
                
                # Accuracy: only on first few batches 
                if acc_batches < 5:
                    samples = model.sample(
                        batch_size=batch.size(0),
                        max_len=batch.size(1),
                        device=device
                    )
                    sample_tensors = []
                    for s in samples:
                        s_tensor = torch.tensor(s + [model.pad_token_id] * (batch.size(1) - len(s)), device=device)
                        sample_tensors.append(s_tensor)
                    samples_tensor = torch.stack(sample_tensors)
                    
                    gt_mask = (batch != model.pad_token_id)
                    pred_mask = (samples_tensor != model.pad_token_id)
                    valid_mask = gt_mask & pred_mask
                    if valid_mask.sum() > 0:
                        acc = ((batch == samples_tensor) & valid_mask).float().sum() / valid_mask.float().sum()
                        val_acc += acc.item()
                        acc_batches += 1
        
        avg_val_acc = val_acc / max(1, acc_batches)
        history["val_loss"].append(val_loss / len(test_loader))
        history["val_recon"].append(val_recon / len(test_loader))
        history["val_accuracy"].append(avg_val_acc)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {history['train_loss'][-1]:.4f}, "
              f"Val Loss: {history['val_loss'][-1]:.4f}, "
              f"Val Acc: {avg_val_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'config': model.config
            }, "best_chemistrySELFIESmodel.pth")
            print(f"  ✓ New best validation loss: {best_val_loss:.4f}")
    
    return model, history
# ====================================================
# 4. Analysis Functions
# ====================================================
def generate_and_decode(model, tokenizer, num_samples=5, max_len=90, temperature=1.0):
    """Generate chemistry molecules using adaptive refinement."""
    model.eval()
    print(f"\n🧪 Generating {num_samples} SELFIES molecules...")
    print(f"   Temperature: {temperature}")
    print("="*70)
    
    with torch.no_grad():
        samples = model.sample(batch_size=2, max_len=20, device='cuda')
    
    # Compute actual lengths (stop at first pad or eos)
    actual_lengths = []
    for s in samples:
        # Find first pad or eos
        s_list = s if isinstance(s, list) else s.tolist()
        length = len(s_list)
        for i, tok in enumerate(s_list):
            if tok == tokenizer.pad_token_id or tok == tokenizer.eos_token_id:
                length = i
                break
        actual_lengths.append(torch.tensor(length))
    
    for i, (sample, length) in enumerate(zip(samples, actual_lengths)):
        decoded = tokenizer.decode(sample[:length.item()], skip_special_tokens=True)
        print(f"{i+1}. (len={length.item()}) {decoded}")
    
    print("="*70)

def plot_training_curves(history):
    """Plot training progression"""
    print("\n📊 Plotting training curves...")
    
    has_length = len(history.get('train_length', [])) > 0
    fig, axes = plt.subplots(2, 3 if has_length else 2, figsize=(18 if has_length else 14, 10))
    
    # Loss
    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Train', alpha=0.8)
    ax.plot(history['val_loss'], label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Reconstruction Loss
    ax = axes[0, 1]
    ax.plot(history['train_recon'], label='Train', alpha=0.8)
    ax.plot(history['val_recon'], label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reconstruction Loss')
    ax.set_title('Reconstruction Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Length Loss
    if has_length:
        ax = axes[0, 2]
        ax.plot(history['train_length'], label='Train', alpha=0.8, color='purple')
        if 'val_length_mae' in history and len(history['val_length_mae']) > 0:
            ax2 = ax.twinx()
            ax2.plot(history['val_length_mae'], label='Val MAE', linewidth=2, color='orange')
            ax2.set_ylabel('Length MAE (tokens)', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Length Loss', color='purple')
        ax.set_title('Length Prediction')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
    
    # Accuracy
    ax = axes[1, 0]
    ax.plot(history['val_accuracy'], linewidth=2, color='green')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Validation Accuracy')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Learning Rate
    ax = axes[1, 1]
    ax.plot(history['learning_rate'], color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Critic Loss
    if len(history.get('train_critic', [])) > 0:
        ax = axes[1, 2] if has_length else axes[1, 1]
        ax.plot(history['train_critic'], linewidth=2, color='red')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Critic Loss')
        ax.set_title('Token Critic Loss')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves_chemistry.png', dpi=150, bbox_inches='tight')
    print("✅ Training curves saved to 'training_curves_chemistry.png'")

# ====================================================
# 5. Main Execution
# ====================================================
if __name__ == "__main__":
    print(f"🚀 Using {device}")
    
    # ========== Load Tokenizer ==========
    print("\n📦 Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gbyuvd/bionat-selfies-gen-tokenizer-wordlevel")
    
    if not hasattr(tokenizer, 'mask_token_id') or tokenizer.mask_token_id is None:
        print("⚠️  Adding [MASK] token to vocabulary...")
        tokenizer.add_special_tokens({'mask_token': '[MASK]'})
        mask_token_id = tokenizer.mask_token_id
    else:
        mask_token_id = tokenizer.mask_token_id
    
    print(f"✅ Special tokens bound: {tokenizer.pad_token_id} {tokenizer.bos_token_id} {tokenizer.eos_token_id} {tokenizer.unk_token_id} {mask_token_id}")
    print(f"📤 Vocab size: {tokenizer.vocab_size}")
    print(f"🎭 [MASK] token ID: {mask_token_id}")
    
    # ========== Load Data ==========
    print("\n📊 Loading chemistry data...")
    df = pd.read_csv("./data/test.csv")
    texts = df["SELFIES"].astype(str).tolist() if "SELFIES" in df.columns else df.iloc[:, 0].astype(str).tolist()
    
    texts = [t for t in texts if len(tokenizer.encode(t)) >= 10]
    print(f"   Total sequences (filtered): {len(texts)}")
    
    random.seed(seed)
    indices = list(range(len(texts)))
    random.shuffle(indices)
    split_idx = int(0.1 * len(indices))
    train_texts = [texts[i] for i in indices[split_idx:]]
    test_texts = [texts[i] for i in indices[:split_idx]]
    
    seq_len = 90
    train_ds = TokenDataset(train_texts, tokenizer, seq_len)
    test_ds = TokenDataset(test_texts, tokenizer, seq_len)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    
    print(f"   Train samples: {len(train_ds)}")
    print(f"   Test samples: {len(test_ds)}")
    
    # ========== Initialize Model ==========
    print("\n🔧 Initializing Model Z (tokenizer-aware + length prediction)...")
    from model import ImplicitRefinementModel, ImplicitRefinementConfig 

    config = ImplicitRefinementConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=320,
        num_layers=6,
        num_heads=4,
        max_seq_len=seq_len,          
        max_refinement_steps=10,
        dropout=0.1,
        use_self_cond=True,
        stop_threshold=0.02,          # for early stopping
        min_refine_uncertainty=0.1,
        ema_decay=0.995,
        diversity_weight=0.05,
        sampling_temperature=1.0
    )

    model = ImplicitRefinementModel(config, tokenizer=tokenizer).to(device)
    print(f"✅ Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"🎭 Using tokenizer-defined mask_token_id={model.mask_token_id}, pad_token_id={model.pad_token_id}")
    
    # ========== Train ==========
    print("\n" + "="*70)
    print("🚀 Starting Adaptive Refinement Training (Model Z + Length Prediction)")
    print("="*70 + "\n")
    
    model, history = train_model_z(
        model, 
        train_loader, 
        test_loader, 
        epochs=1,
        lr=1e-4
    )
    
    # ========== Save Final Model ==========
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': history
    }, "chemistrySELFIESmodelfinal.pth")
    print("\n✅ Final model saved to 'chemistrySELFIESmodelfinal.pth'")
    
    # ========== Visualizations ==========
    plot_training_curves(history)
    
    # ========== Generate Molecules ==========
    print("\n🧪 Generating with different temperatures...")
    for temp in [0.8, 1.0]:
        print(f"\n--- Temperature {temp} ---")
        generate_and_decode(model, tokenizer, num_samples=5, max_len=seq_len)
    
    print("\n Training and analysis complete!")
    