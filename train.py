#!/usr/bin/env python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import random
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os

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
# 3. Training Function (FIXED)
# ====================================================
def train_model_v_plus_m(model, train_loader, test_loader, epochs=10, lr=1e-4):
    """
    Fixed training function with proper metric tracking and validation.
    """
    # Setup optimizer and scheduler
    try:
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
        print("✅ Using Ranger21 optimizer")
    except ImportError:
        print("⚠️ Ranger21 not found, using AdamW + CosineAnnealingLR instead")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=epochs * len(train_loader),
            eta_min=lr * 0.05
        )

    # Initialize history with all metrics
    history = {
        "train_loss": [],
        "train_recon": [],
        "train_diversity": [],
        "val_loss": [],
        "val_recon": [],
        "val_diversity": [],
        "val_accuracy": [],
        "learning_rate": []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # ========== Training ==========
        model.train()
        train_metrics = {
            'total': 0.0,
            'recon': 0.0,
            'diversity': 0.0,
            'count': 0
        }
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (batch, lengths) in enumerate(pbar):
            batch, lengths = batch.to(device), lengths.to(device)
            
            optimizer.zero_grad()
            
            # Use the standard loss method (no memory-specific method exists)
            losses = model.loss(batch, lengths=lengths)
            
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Accumulate metrics
            train_metrics['total'] += losses['total'].item()
            train_metrics['recon'] += losses['recon'].item()
            train_metrics['diversity'] += losses['diversity'].item()
            train_metrics['count'] += 1
            
            # Update EMA teacher periodically
            if batch_idx % 10 == 0:
                model.update_teacher()
            
            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'recon': f"{losses['recon'].item():.4f}",
                'div': f"{losses['diversity'].item():.4f}",
                'lr': f"{current_lr:.2e}"
            })
        
        # Log training averages
        history["train_loss"].append(train_metrics['total'] / train_metrics['count'])
        history["train_recon"].append(train_metrics['recon'] / train_metrics['count'])
        history["train_diversity"].append(train_metrics['diversity'] / train_metrics['count'])
        history["learning_rate"].append(current_lr)
        
        # ========== Validation ==========
        model.eval()
        val_metrics = {
            'total': 0.0,
            'recon': 0.0,
            'diversity': 0.0,
            'count': 0
        }
        val_acc_sum = 0.0
        val_acc_count = 0
        
        with torch.no_grad():
            for batch_idx, (batch, lengths) in enumerate(tqdm(test_loader, desc="Validating", leave=False)):
                batch, lengths = batch.to(device), lengths.to(device)
                
                # Compute validation loss
                losses = model.loss(batch, lengths=lengths)
                val_metrics['total'] += losses['total'].item()
                val_metrics['recon'] += losses['recon'].item()
                val_metrics['diversity'] += losses['diversity'].item()
                val_metrics['count'] += 1
                
                # Compute accuracy on subset of batches (expensive)
                if batch_idx < 5:  # Only first 5 batches
                    try:
                        samples = model.sample(
                            batch_size=min(batch.size(0), 8),  # Limit batch size
                            max_len=batch.size(1),
                            device=device
                        )
                        
                        # Convert samples to tensor
                        sample_tensors = []
                        for s in samples:
                            if isinstance(s, torch.Tensor):
                                s = s.cpu().tolist()
                            # Pad to match batch length
                            s_padded = s[:batch.size(1)] + [model.pad_token_id] * max(0, batch.size(1) - len(s))
                            sample_tensors.append(torch.tensor(s_padded, device=device, dtype=torch.long))
                        
                        if sample_tensors:
                            samples_tensor = torch.stack(sample_tensors)
                            batch_subset = batch[:len(samples_tensor)]
                            
                            # Compute accuracy only on non-padding tokens
                            valid_mask = batch_subset != model.pad_token_id
                            if valid_mask.sum() > 0:
                                correct = (batch_subset == samples_tensor) & valid_mask
                                acc = correct.float().sum() / valid_mask.float().sum()
                                val_acc_sum += acc.item()
                                val_acc_count += 1
                    except Exception as e:
                        print(f"\n⚠️ Accuracy computation failed: {e}")
                        continue
        
        # Log validation averages
        avg_val_loss = val_metrics['total'] / val_metrics['count']
        avg_val_recon = val_metrics['recon'] / val_metrics['count']
        avg_val_div = val_metrics['diversity'] / val_metrics['count']
        avg_val_acc = val_acc_sum / val_acc_count if val_acc_count > 0 else 0.0
        
        history["val_loss"].append(avg_val_loss)
        history["val_recon"].append(avg_val_recon)
        history["val_diversity"].append(avg_val_div)
        history["val_accuracy"].append(avg_val_acc)
        
        # Print epoch summary
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{epochs} Summary:")
        print(f"  Train - Loss: {history['train_loss'][-1]:.4f}, "
              f"Recon: {history['train_recon'][-1]:.4f}, "
              f"Div: {history['train_diversity'][-1]:.4f}")
        print(f"  Val   - Loss: {avg_val_loss:.4f}, "
              f"Recon: {avg_val_recon:.4f}, "
              f"Div: {avg_val_div:.4f}, "
              f"Acc: {avg_val_acc:.4f}")
        print(f"  LR: {current_lr:.2e}")
        print(f"{'='*70}\n")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': avg_val_loss,
                'config': model.config,
                'history': history
            }, "best_chemistrySELFIESmodel.pth")
            print(f"  ✓ Saved best model (val_loss: {best_val_loss:.4f})\n")
    
    return model, history

# ====================================================
# 4. Analysis Functions (FIXED)
# ====================================================
def generate_and_decode(model, tokenizer, num_samples=5, max_len=90, temperature=1.0):
    """Generate chemistry molecules using adaptive refinement."""
    model.eval()
    print(f"\n🧪 Generating {num_samples} SELFIES molecules...")
    print(f"   Temperature: {temperature}")
    print("="*70)
    
    # Temporarily set temperature
    original_temp = model.config.sampling_temperature
    model.config.sampling_temperature = temperature
    
    with torch.no_grad():
        samples = model.sample(batch_size=num_samples, max_len=max_len, device=device)
    
    # Restore original temperature
    model.config.sampling_temperature = original_temp
    
    decoded_samples = []
    for i, sample in enumerate(samples):
        # Handle both list and tensor
        if isinstance(sample, torch.Tensor):
            sample = sample.cpu().tolist()
        elif not isinstance(sample, list):
            sample = list(sample)
            
        # Find actual length (stop at pad/eos)
        actual_len = len(sample)
        for j, tok in enumerate(sample):
            if tok == tokenizer.pad_token_id or tok == tokenizer.eos_token_id:
                actual_len = j
                break
                
        decoded = tokenizer.decode(sample[:actual_len], skip_special_tokens=True)
        decoded_samples.append(decoded)
        print(f"{i+1}. (len={actual_len:2d}) {decoded}")
    
    print("="*70)
    return decoded_samples

def plot_training_curves(history, save_path='training_curves_chemistry.png'):
    """
    Plot training progression with proper error handling.
    """
    print("\n📊 Plotting training curves...")
    
    # Verify we have data
    if not history.get('train_loss') or len(history['train_loss']) == 0:
        print("⚠️ No training data to plot!")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Curves: Implicit Refinement Model', fontsize=16, fontweight='bold')
    
    epochs = len(history['train_loss'])
    x = list(range(1, epochs + 1))  # 1-indexed for better readability
    
    # 1. Total Loss
    axes[0, 0].plot(x, history['train_loss'], label='Train', alpha=0.8, linewidth=2, marker='o', markersize=4)
    if history.get('val_loss') and len(history['val_loss']) > 0:
        axes[0, 0].plot(x, history['val_loss'], label='Val', linewidth=2, marker='s', markersize=4)
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Loss', fontsize=11)
    axes[0, 0].set_title('Total Loss', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')
    
    # 2. Reconstruction Loss
    axes[0, 1].plot(x, history['train_recon'], label='Train', alpha=0.8, linewidth=2, marker='o', markersize=4, color='tab:orange')
    if history.get('val_recon') and len(history['val_recon']) > 0:
        axes[0, 1].plot(x, history['val_recon'], label='Val', linewidth=2, marker='s', markersize=4, color='tab:red')
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Loss', fontsize=11)
    axes[0, 1].set_title('Reconstruction Loss', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3, linestyle='--')
    
    # 3. Diversity Loss
    if history.get('train_diversity') and any(history['train_diversity']):
        axes[0, 2].plot(x, history['train_diversity'], label='Train', alpha=0.8, linewidth=2, marker='o', markersize=4, color='purple')
        if history.get('val_diversity') and len(history['val_diversity']) > 0:
            axes[0, 2].plot(x, history['val_diversity'], label='Val', linewidth=2, marker='s', markersize=4, color='magenta')
        axes[0, 2].legend(fontsize=10)
    else:
        axes[0, 2].text(0.5, 0.5, 'No Diversity Loss Data', 
                       ha='center', va='center', transform=axes[0, 2].transAxes, fontsize=12)
    axes[0, 2].set_xlabel('Epoch', fontsize=11)
    axes[0, 2].set_ylabel('Loss', fontsize=11)
    axes[0, 2].set_title('Diversity Loss', fontsize=12, fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3, linestyle='--')
    
    # 4. Validation Accuracy
    if history.get('val_accuracy') and any(history['val_accuracy']):
        axes[1, 0].plot(x, history['val_accuracy'], linewidth=2.5, color='green', marker='D', markersize=5)
        axes[1, 0].set_ylim(0, 1.05)
        axes[1, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='50% baseline')
        axes[1, 0].legend(fontsize=10)
    else:
        axes[1, 0].text(0.5, 0.5, 'No Accuracy Data', 
                       ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=12)
    axes[1, 0].set_xlabel('Epoch', fontsize=11)
    axes[1, 0].set_ylabel('Accuracy', fontsize=11)
    axes[1, 0].set_title('Validation Accuracy', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, linestyle='--')
    
    # 5. Learning Rate
    if history.get('learning_rate') and any(history['learning_rate']):
        axes[1, 1].plot(x, history['learning_rate'], color='darkorange', linewidth=2, marker='v', markersize=4)
        axes[1, 1].set_yscale('log')
    else:
        axes[1, 1].text(0.5, 0.5, 'No LR Data', 
                       ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].set_xlabel('Epoch', fontsize=11)
    axes[1, 1].set_ylabel('Learning Rate (log scale)', fontsize=11)
    axes[1, 1].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, linestyle='--', which='both')
    
    # 6. Loss Comparison (Train vs Val)
    if history.get('train_loss') and history.get('val_loss'):
        gap = [v - t for t, v in zip(history['train_loss'], history['val_loss'])]
        axes[1, 2].plot(x, gap, color='crimson', linewidth=2, marker='x', markersize=5)
        axes[1, 2].axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        axes[1, 2].fill_between(x, gap, 0, alpha=0.3, color='crimson')
    else:
        axes[1, 2].text(0.5, 0.5, 'Insufficient Data', 
                       ha='center', va='center', transform=axes[1, 2].transAxes, fontsize=12)
    axes[1, 2].set_xlabel('Epoch', fontsize=11)
    axes[1, 2].set_ylabel('Gap (Val - Train)', fontsize=11)
    axes[1, 2].set_title('Generalization Gap', fontsize=12, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Training curves saved to '{save_path}'")
    plt.close()

def analyze_trajectory(model, tokenizer, device, max_len=50):
    """
    Analyze a single refinement trajectory step-by-step.
    """
    print("\n🔍 Analyzing Refinement Trajectory...")
    print("="*70)
    
    model.eval()
    analysis = model.analyze_refinement_trajectory(
        max_len=max_len,
        device=device,
        seed=42
    )
    
    model.print_refinement_trajectory(analysis, tokenizer=tokenizer)
    
    return analysis

# ====================================================
# 5. Main Execution
# ====================================================
if __name__ == "__main__":
    print(f"🚀 Using device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # ========== Load Tokenizer ==========
    print("\n📦 Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gbyuvd/bionat-selfies-gen-tokenizer-wordlevel")
    
    if not hasattr(tokenizer, 'mask_token_id') or tokenizer.mask_token_id is None:
        print("⚠️  Adding [MASK] token to vocabulary...")
        tokenizer.add_special_tokens({'mask_token': '[MASK]'})
    
    print(f"✅ Special tokens: pad={tokenizer.pad_token_id}, "
          f"eos={tokenizer.eos_token_id}, mask={tokenizer.mask_token_id}")
    print(f"📤 Vocab size: {tokenizer.vocab_size}")
    
    # ========== Load Data ==========
    print("\n📊 Loading chemistry data...")
    if not os.path.exists("./data/test.csv"):
        print("❌ Data file not found! Please ensure ./data/test.csv exists")
        exit(1)
        
    df = pd.read_csv("./data/test.csv")
    texts = df["SELFIES"].astype(str).tolist() if "SELFIES" in df.columns else df.iloc[:, 0].astype(str).tolist()
    
    # Filter sequences
    texts = [t for t in texts if 10 <= len(tokenizer.encode(t)) <= 90]
    print(f"   Total sequences (filtered 10-90 tokens): {len(texts)}")
    
    if len(texts) < 100:
        print(f"⚠️  Warning: Only {len(texts)} sequences available. Consider using more data.")
    
    # Split data
    random.seed(seed)
    indices = list(range(len(texts)))
    random.shuffle(indices)
    split_idx = int(0.1 * len(indices))
    train_texts = [texts[i] for i in indices[split_idx:]]
    test_texts = [texts[i] for i in indices[:split_idx]]
    
    seq_len = 90
    train_ds = TokenDataset(train_texts, tokenizer, seq_len)
    test_ds = TokenDataset(test_texts, tokenizer, seq_len)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"   Train samples: {len(train_ds)}")
    print(f"   Test samples: {len(test_ds)}")
    
    # ========== Initialize Model ==========
    print("\n🔧 Initializing Implicit Refinement Model...")
    from model3 import ImplicitRefinementModel, ImplicitRefinementConfig 

    config = ImplicitRefinementConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=320,
        num_layers=4,
        num_heads=4,
        max_seq_len=seq_len,          
        max_refinement_steps=10,
        dropout=0.1,
        use_self_cond=True,
        stop_threshold=0.02,
        min_refine_uncertainty=0.1,
        ema_decay=0.995,
        diversity_weight=0.05,
        sampling_temperature=1.0,
        use_refine_gate=True,  # Enable internal refinement gate
        use_gradient_checkpointing=False  # Enable for larger models
    )

    model = ImplicitRefinementModel(config, tokenizer=tokenizer).to(device)
    model.init_teacher()  # Initialize EMA teacher
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model initialized")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1e6:.2f} MB (fp32)")
    
    # ========== Train ==========
    print("\n" + "="*70)
    print("🚀 Starting Adaptive Refinement Training")
    print("="*70 + "\n")
    
    model, history = train_model_v_plus_m(
        model, 
        train_loader, 
        test_loader, 
        epochs=1,
        lr=3e-4
    )
    
    # ========== Save Final Model ==========
    final_path = "chemistrySELFIESmodelfinal.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': history
    }, "chemistrySELFIESmodelfinal.pth")
    print(f"\n✅ Final model saved to '{final_path}'")
    
    # ========== Visualizations ==========
    plot_training_curves(history)
    
    # ========== Analyze Trajectory ==========
    analyze_trajectory(model, tokenizer, device, max_len=50)
    
    # ========== Generate Molecules ==========
    print("\n🧪 Generating molecules with different temperatures...")
    for temp in [0.7, 1.0, 1.3]:
        print(f"\n{'='*70}")
        print(f"Temperature: {temp}")
        print('='*70)
        generate_and_decode(model, tokenizer, num_samples=5, max_len=seq_len, temperature=temp)
    
    print("\n" + "="*70)
    print("🎉 Training and analysis complete!")
    print("="*70)
