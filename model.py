from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

@dataclass
class ImplicitRefinementConfig:
    vocab_size: int = 100
    hidden_size: int = 128
    num_layers: int = 3
    num_heads: int = 4
    max_seq_len: int = 20
    max_refinement_steps: int = 6
    dropout: float = 0.1
    use_self_cond: bool = True
    stop_threshold: float = 0.02
    min_refine_uncertainty: float = 0.1
    ema_decay: float = 0.995
    diversity_weight: float = 0.05
    sampling_temperature: float = 1.2
    use_refine_gate: bool = False
    use_gradient_checkpointing: bool = False  # New: memory optimization
    use_flash_attention: bool = False  # New: speed optimization


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        position = torch.arange(max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_seq_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, L, device):
        # No need for .to(device) - buffer is already on correct device
        return self.pe[:L].unsqueeze(0)


class AdaptivePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.base_pe = SinusoidalPositionalEmbedding(dim, max_seq_len)
        self.decay_logit = nn.Parameter(torch.zeros(max_seq_len))
        # Better initialization for decay
        nn.init.normal_(self.decay_logit, mean=0.0, std=0.02)

    def forward(self, x):
        B, L = x.shape[:2]
        pe = self.base_pe(L, x.device)
        decay = torch.sigmoid(self.decay_logit[:L])
        return pe * decay.view(1, -1, 1)  # More efficient than unsqueeze


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            config.hidden_size, config.num_heads, 
            batch_first=True, dropout=config.dropout
        )
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, 4 * config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(4 * config.hidden_size, config.hidden_size),
            nn.Dropout(config.dropout)  # Added output dropout
        )
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        
        # Better weight initialization
        self._init_weights()
    
    def _init_weights(self):
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        # Pre-norm once, reuse
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm, x_norm, x_norm, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ImplicitRefinementModel(nn.Module):
    def __init__(self, config: ImplicitRefinementConfig, tokenizer=None):
        super().__init__()
        self.config = config

        # Token ID setup (unchanged logic)
        if tokenizer is not None:
            self.pad_token_id = getattr(tokenizer, "pad_token_id", None)
            self.mask_token_id = getattr(tokenizer, "mask_token_id", None)
            self.eos_token_id = getattr(tokenizer, "eos_token_id", 
                                       getattr(tokenizer, "sep_token_id", None))
            if self.mask_token_id is None:
                raise ValueError("Tokenizer must define mask_token_id")
            if self.pad_token_id is None:
                print("[Warning] pad_token_id not set — using 0")
                self.pad_token_id = 0
            if self.eos_token_id is None:
                print("[Warning] No eos_token_id or sep_token_id — will use implicit stopping")
        else:
            self.pad_token_id = 0
            self.mask_token_id = config.vocab_size - 1
            self.eos_token_id = config.vocab_size - 2

        special_tokens = [self.pad_token_id, self.mask_token_id]
        if self.eos_token_id is not None:
            special_tokens.append(self.eos_token_id)
        if len(set(special_tokens)) != len(special_tokens):
            raise ValueError(f"Special token collision")

        # Core embeddings with better initialization
        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        
        self.pos_emb = AdaptivePositionalEmbedding(config.hidden_size, config.max_seq_len)
        self.time_emb = nn.Linear(1, config.hidden_size)
        nn.init.normal_(self.time_emb.weight, mean=0.0, std=0.02)

        if config.use_self_cond:
            self.self_cond_proj = nn.Linear(config.vocab_size, config.hidden_size)
            nn.init.xavier_uniform_(self.self_cond_proj.weight)
        else:
            self.self_cond_proj = None

        self.transformer = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        self.to_logits = nn.Linear(config.hidden_size, config.vocab_size)
        nn.init.normal_(self.to_logits.weight, mean=0.0, std=0.02)
        
        if config.use_refine_gate:
            self.to_refine_gate = nn.Linear(config.hidden_size, 1)
            nn.init.xavier_uniform_(self.to_refine_gate.weight)
            nn.init.constant_(self.to_refine_gate.bias, 2.0)
        else:
            self.to_refine_gate = None

        # EMA teacher
        self.teacher = None
        self.ema_decay = config.ema_decay
        self.register_buffer("ema_step", torch.tensor(0, dtype=torch.long), persistent=False)
        
        # Cache for normalized entropy denominator
        self.register_buffer("log_vocab_size", torch.tensor(math.log(config.vocab_size)), persistent=False)

    def init_teacher(self):
        if self.teacher is None:
            self.teacher = copy.deepcopy(self)
            self.teacher.teacher = None
            for p in self.teacher.parameters():
                p.requires_grad_(False)

    def update_teacher(self):
        if self.teacher is None:
            self.init_teacher()
            return
        self.ema_step += 1
        # Use in-place operations for efficiency
        decay = min(self.ema_decay, (self.ema_step - 1).float() / (self.ema_step + 1).float())
        with torch.no_grad():
            for t_param, s_param in zip(self.teacher.parameters(), self.parameters()):
                t_param.mul_(decay).add_(s_param, alpha=1 - decay)

    def _forward_transformer(self, x):
        """Helper for gradient checkpointing"""
        for block in self.transformer:
            if self.config.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, x_self_cond: Optional[torch.Tensor] = None):
        B, L = x_t.shape
        
        # Combine embeddings more efficiently
        x = self.token_emb(x_t)
        x = x + self.pos_emb(x_t)
        
        # Optimize time embedding computation
        time_fea = self.time_emb(t.view(-1, 1)).view(B, 1, -1)
        x = x + time_fea
        
        if x_self_cond is not None and self.self_cond_proj is not None:
            x = x + self.self_cond_proj(x_self_cond)
        
        x = self._forward_transformer(x)
        
        logits = self.to_logits(x)
        
        if self.config.use_refine_gate and self.to_refine_gate is not None:
            refine_logits = self.to_refine_gate(x).squeeze(-1)
            refine_gate = torch.sigmoid(refine_logits)
            return logits, refine_gate
        return logits

    def _get_uncertainty_from_teacher(self, x_t, t, x_self_cond):
        """Optimized uncertainty computation"""
        model = self.teacher if self.teacher is not None else self
        with torch.no_grad():
            if self.config.use_refine_gate:
                logits, _ = model(x_t, t, x_self_cond)
            else:
                logits = model(x_t, t, x_self_cond)
        
        # More numerically stable entropy computation
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy / self.log_vocab_size

    @torch.no_grad()
    def sample(self, batch_size: int, max_len: int, device: torch.device) -> List[List[int]]:
        x_t = torch.full((batch_size, max_len), self.mask_token_id, device=device, dtype=torch.long)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        x_self_cond = None

        # Pre-allocate for efficiency
        t_tensor = torch.empty(batch_size, device=device, dtype=torch.float)
        
        # Track previous state without cloning
        x_prev = x_t.clone()

        for step in range(self.config.max_refinement_steps):
            t_tensor.fill_(step)
            
            if self.config.use_refine_gate:
                logits, refine_gate = self(x_t, t_tensor, x_self_cond)
            else:
                logits = self(x_t, t_tensor, x_self_cond)
                refine_gate = None

            uncertainty = self._get_uncertainty_from_teacher(x_t, t_tensor, x_self_cond)
            
            # More efficient sampling
            probs = F.softmax(logits * (1.0 / self.config.sampling_temperature), dim=-1)
            pred_tokens = torch.multinomial(
                probs.view(-1, probs.size(-1)), 
                num_samples=1
            ).view(batch_size, max_len)

            if self.config.use_refine_gate:
                needs_refine = refine_gate > 0.5
            else:
                needs_refine = uncertainty > self.config.min_refine_uncertainty

            if self.eos_token_id is not None:
                eos_mask = x_t == self.eos_token_id
                for b in range(batch_size):
                    if finished[b]:
                        needs_refine[b].fill_(False)
                        continue
                    eos_pos = eos_mask[b].nonzero(as_tuple=True)[0]
                    if eos_pos.numel() > 0:
                        first_eos = eos_pos.min().item()
                        needs_refine[b, first_eos:] = False
                        finished[b] = True

            # In-place update where possible
            x_t = torch.where(needs_refine, pred_tokens, x_t)

            # More efficient change detection
            changed = x_t != x_prev
            if self.eos_token_id is not None:
                active_mask = ~finished.unsqueeze(1)
                change_ratio = (changed & active_mask).sum().float() / (active_mask.sum().float() + 1e-8)
            else:
                change_ratio = changed.float().mean()

            if change_ratio < self.config.stop_threshold or finished.all():
                print(f"✅ Stopped at step {step+1} (change: {change_ratio:.2%})")
                break

            x_prev = x_t.clone()  # Only clone when continuing
            
            if self.config.use_self_cond:
                x_self_cond = F.softmax(logits, dim=-1)

        # Convert to output format
        outputs = []
        for b in range(batch_size):
            seq = x_t[b].cpu().tolist()
            if self.eos_token_id is not None:
                try:
                    eos_idx = seq.index(self.eos_token_id)
                    seq = seq[:eos_idx + 1]
                except ValueError:
                    pass
            outputs.append(seq)
        return outputs

    def loss(self, x_0: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        B, L = x_0.shape
        device = x_0.device

        # Compute padding mask once
        if lengths is not None:
            positions = torch.arange(L, device=device).unsqueeze(0)
            padding_mask = positions < lengths.unsqueeze(1)
        else:
            padding_mask = x_0 != self.pad_token_id

        t = torch.randint(0, self.config.max_refinement_steps, (B,), device=device)
        mask_rate = 1.0 - (t.float() / self.config.max_refinement_steps)
        
        # Efficient masking
        rand_vals = torch.rand(B, L, device=device)
        mask = (rand_vals < mask_rate.unsqueeze(1)) & padding_mask
        x_t = torch.where(mask, self.mask_token_id, x_0)

        x_self_cond = None
        if self.config.use_self_cond:
            with torch.no_grad():
                t_prev = torch.clamp(t + 1, max=self.config.max_refinement_steps - 1)
                mask_rate_prev = 1.0 - (t_prev.float() / self.config.max_refinement_steps)
                rand_vals_prev = torch.rand(B, L, device=device)
                mask_prev = (rand_vals_prev < mask_rate_prev.unsqueeze(1)) & padding_mask
                x_t_prev = torch.where(mask_prev, self.mask_token_id, x_0)
                
                if self.config.use_refine_gate:
                    logits_init, _ = self(x_t_prev, t_prev.float())
                else:
                    logits_init = self(x_t_prev, t_prev.float())
                x_self_cond = F.softmax(logits_init / 1.5, dim=-1)

        if self.config.use_refine_gate:
            logits, refine_gate = self(x_t, t.float(), x_self_cond)
        else:
            logits = self(x_t, t.float(), x_self_cond)

        # More efficient loss computation
        mask_flat = mask.view(-1)
        recon_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            x_0.view(-1),
            reduction='none'
        )
        recon_loss = (recon_loss * mask_flat.float()).sum() / (mask_flat.float().sum() + 1e-8)

        # Optimized diversity loss
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum(dim=-1)
        diversity_loss = torch.relu(0.5 - entropy) * mask.float()
        diversity_loss = diversity_loss.sum() / (mask.float().sum() + 1e-8)

        total_loss = recon_loss + self.config.diversity_weight * diversity_loss
        return {"total": total_loss, "recon": recon_loss, "diversity": diversity_loss}
    
    @torch.no_grad()
    def analyze_refinement_trajectory(
        self,
        max_len: int,
        device: torch.device,
        prompt: Optional[torch.Tensor] = None,
        seed: Optional[int] = None
    ) -> dict:
        if seed is not None:
            torch.manual_seed(seed)
        
        B = 1
        if prompt is not None:
            x_t = prompt.to(device).unsqueeze(0)
            assert x_t.shape[1] == max_len
        else:
            x_t = torch.full((B, max_len), self.mask_token_id, device=device, dtype=torch.long)
        
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        x_self_cond = None
        trajectory = []

        # Initial state
        tokens_str = self._tokens_to_str(x_t[0])
        trajectory.append({
            'step': 0,
            'tokens': x_t[0].cpu().clone(),
            'tokens_str': tokens_str,
            'needs_refine': torch.ones(max_len, dtype=torch.bool),
            'mask_str': "↑" * max_len,
            'entropy': [1.0] * max_len,
            'change_ratio': 1.0,
            'finished': False,
            'refine_gate': [1.0] * max_len if self.config.use_refine_gate else None
        })

        x_prev = x_t.clone()
        stopped_early = False

        for step in range(self.config.max_refinement_steps):
            t = torch.full((B,), step, dtype=torch.float, device=device)
            
            if self.config.use_refine_gate:
                logits, refine_gate = self(x_t, t, x_self_cond)
            else:
                logits = self(x_t, t, x_self_cond)
                refine_gate = None

            uncertainty = self._get_uncertainty_from_teacher(x_t, t, x_self_cond)
            probs = F.softmax(logits * (1.0 / self.config.sampling_temperature), dim=-1)
            pred_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(B, max_len)

            if self.config.use_refine_gate:
                needs_refine = refine_gate > 0.5
                refine_gate_cpu = refine_gate[0].cpu().tolist()
            else:
                needs_refine = uncertainty > self.config.min_refine_uncertainty
                refine_gate_cpu = None

            if self.eos_token_id is not None:
                eos_mask = x_t == self.eos_token_id
                if not finished[0]:
                    eos_pos = eos_mask[0].nonzero(as_tuple=True)[0]
                    if eos_pos.numel() > 0:
                        first_eos = eos_pos.min().item()
                        needs_refine[0, first_eos:] = False
                        finished[0] = True
                else:
                    needs_refine[0] = False

            new_x_t = torch.where(needs_refine, pred_tokens, x_t)
            changed = new_x_t != x_prev
            
            if self.eos_token_id is not None:
                active = ~finished.unsqueeze(1).expand(-1, max_len)
                change_ratio = (changed & active).sum().float() / (active.sum().float() + 1e-8)
            else:
                change_ratio = changed.float().mean()

            tokens_str = self._tokens_to_str(new_x_t[0])
            mask_str = self._build_mask_str(needs_refine[0], new_x_t[0])
            trajectory.append({
                'step': step + 1,
                'tokens': new_x_t[0].cpu().clone(),
                'tokens_str': tokens_str,
                'needs_refine': needs_refine[0].cpu().clone(),
                'mask_str': mask_str,
                'entropy': uncertainty[0].cpu().tolist(),
                'change_ratio': change_ratio.item(),
                'finished': finished.item(),
                'refine_gate': refine_gate_cpu
            })

            if change_ratio < self.config.stop_threshold or finished.all():
                stopped_early = True
                x_t = new_x_t
                break

            x_t = new_x_t
            x_prev = x_t.clone()
            
            if self.config.use_self_cond:
                x_self_cond = F.softmax(logits, dim=-1)

        final_seq = x_t[0].cpu().tolist()
        if self.eos_token_id is not None:
            try:
                eos_idx = final_seq.index(self.eos_token_id)
                final_seq = final_seq[:eos_idx + 1]
            except ValueError:
                pass

        return {
            'steps': trajectory,
            'final_seq': final_seq,
            'stopped_early': stopped_early,
            'max_steps': self.config.max_refinement_steps
        }

    def _tokens_to_str(self, tokens: torch.Tensor) -> List[str]:
        """Optimized token to string conversion"""
        ids = tokens.cpu().tolist()
        strs = []
        for tid in ids:
            if tid == self.mask_token_id:
                strs.append("MASK")
            elif tid == self.pad_token_id:
                strs.append("PAD")
            elif tid == self.eos_token_id:
                strs.append("EOS")
            else:
                strs.append(f"[{tid}]")
        return strs

    def _build_mask_str(self, needs_refine: torch.Tensor, tokens: torch.Tensor) -> str:
        """Optimized mask string builder"""
        arrows = []
        for i in range(len(needs_refine)):
            if needs_refine[i]:
                arrows.append("↑")
            else:
                tid = tokens[i].item()
                if tid in (self.mask_token_id, self.pad_token_id):
                    arrows.append("·")
                else:
                    arrows.append(" ")
        return "".join(arrows)
    
    def print_refinement_trajectory(self, analysis: dict, tokenizer=None):
        """Unchanged visualization method"""
        steps = analysis['steps']
        mode = "INTERNAL GATE" if self.config.use_refine_gate else "UNCERTAINTY THRESHOLD"
        print(f"🔍 Refinement Trajectory ({mode})\n")
        
        for i, step_info in enumerate(steps):
            step = step_info['step']
            tokens = step_info['tokens_str']
            arrow_line = ""
            for j in range(len(tokens)):
                if step_info['needs_refine'][j]:
                    arrow_line += " ↑"
                else:
                    arrow_line += "  "
            token_display = " ".join([f"[{t:>4}]" for t in tokens])
            print(f"t={step}: {token_display}")
            print(f"       {arrow_line}")

            if step_info.get('refine_gate') is not None:
                prob_str = " ".join([f"{p:.2f}" for p in step_info['refine_gate']])
                print(f"       Gate: {prob_str}")

            if i == len(steps) - 1:
                cr = step_info['change_ratio']
                msg = "✅ Early stop" if analysis['stopped_early'] else "⏹️ Max steps"
                print(f"       {'  ' * len(tokens)} ← change_ratio={cr:.1%} → {msg}")

        final = analysis['final_seq']
        if tokenizer:
            print(f"\nFinal: {repr(tokenizer.decode(final, skip_special_tokens=False))}")
        else:
            print(f"\nFinal: {final}")


if __name__ == "__main__":
    class DummyTokenizer:
        pad_token_id = 0
        mask_token_id = 99
        eos_token_id = 98
        vocab_size = 100

    tokenizer = DummyTokenizer()

    for use_gate in [False, True]:
        print("\n" + "="*60)
        print(f"TESTING OPTIMIZED V: use_refine_gate = {use_gate}")
        print("="*60)

        config = ImplicitRefinementConfig(
            vocab_size=100,
            hidden_size=64,
            num_layers=2,
            max_seq_len=8,
            max_refinement_steps=3,
            stop_threshold=0.05,
            diversity_weight=0.1,
            sampling_temperature=1.0,
            use_refine_gate=use_gate,
            use_gradient_checkpointing=False  # Enable for training large models
        )

        model = ImplicitRefinementModel(config, tokenizer=tokenizer)
        model.eval()
        model.init_teacher()

        # Test forward
        B, L = 1, 5
        x_t = torch.full((B, L), model.mask_token_id, dtype=torch.long)
        t = torch.zeros(B)
        with torch.no_grad():
            if use_gate:
                logits, gate = model(x_t, t)
                print(f"✅ Internal gate: {gate[0].tolist()}")
            else:
                logits = model(x_t, t)

        # Test sampling
        samples = model.sample(1, 6, 'cpu')
        print(f"Sample: {samples[0]}")

        # Test trajectory
        analysis = model.analyze_refinement_trajectory(5, 'cpu', seed=1)
        model.print_refinement_trajectory(analysis)
