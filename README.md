# Imadarem V  | WIP
**Implicitly Adaptive Refinement Model — Version V**

A lightweight, iterative-refinement language model that learns to “fill-in-the-blanks” starting from a fully-masked sequence — now with an **internal, learnable refinement gate**. Works with any tokenizer that supplies `mask_token_id`, `pad_token_id`, and (optionally) `eos_token_id`.

## 1. High-level idea  
Instead of left-to-right generation, the model treats text generation as a **denoising** process:

1. Start with every token = `[MASK]`  
2. Run a small, shared transformer for ≤ K steps  
3. At each step, **only re-predict tokens the model itself deems uncertain**  
4. Freeze tokens once an `[EOS]` is sampled; stop early when < τ tokens change  

The training objective is a masked-language-modeling loss with a time-dependent corruption schedule:  
`mask_rate(t) = 1 − t / K`.

In Version V, refinement decisions are made **internally**:
- When `use_refine_gate=True`, a lightweight gate head predicts a per-token **refinement probability**.
- Tokens are updated **iff** `refine_gate > 0.5` — no external entropy threshold needed.
- The gate is trained end-to-end and initialized to **refine by default**.

```text
🔍 Refinement Trajectory 

t=0: [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK]
        ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
t=1: [MASK] [ [5]] [ [5]] [ [5]] [ [5]] [ [6]] [ [5]] [ [5]] [[28]] [ [6]] [ [5]] [ [5]] [ [5]] [ [5]] [ [6]] [ [7]]
          ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
t=2: [MASK] [ [9]] [ [7]] [ [5]] [ [5]] [[11]] [ [6]] [ [5]] [[11]] [ [9]] [ [8]] [ [5]] [ [7]] [ [7]] [ [9]] [[12]]
          ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
t=3: [MASK] [ [5]] [ [7]] [ [9]] [ [5]] [[10]] [ [5]] [[14]] [ [8]] [ [8]] [ EOS] [ [5]] [ [5]] [ [5]] [ [6]] [[12]]
          ↑   ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑  
t=4: [MASK] [ [5]] [ [9]] [ [5]] [[13]] [ [5]] [[13]] [ [5]] [ [8]] [[14]] [ EOS] [ [5]] [ [5]] [ [5]] [ [6]] [[12]]
          ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑            
                                        ← change_ratio=0.0% → ✅ Early stop

Final: '<mask> [C] [O] [C] [Ring2] [C] [Ring2] [C] [Branch1] [Branch2] </s>'
```



## 2. Architecture snapshot  
| Component | Purpose | Key hyper-params |
|-----------|---------|------------------|
| `TokenEmbedding` | learned input embeddings | `vocab_size`, `hidden_size` |
| `AdaptivePositionalEmbedding` | sinusoidal PE × learned per-position decay | `max_seq_len` |
| `TimeEmbedding` | scalar step → vector (1-layer MLP) | `hidden_size` |
| `Self-condition projection` | soft previous logits → residual input | optional |
| `Transformer blocks` | full self-attention (shared across steps) | `num_layers`, `num_heads`, `dropout` |
| `Refinement Gate` (**V-only**) | predicts per-token refine/no-refine | sigmoid head, bias-init to +2.0 |
| `Teacher (EMA)` | exponential moving average for stable uncertainty (used only when gate is off) | `ema_decay` |


## 3. Sampling modes  
Imadarem V supports **two refinement strategies**:

| Mode | Trigger | Controlled by |
|------|--------|---------------|
| **Uncertainty Threshold** | entropy > `min_refine_uncertainty` | `use_refine_gate=False` |
| **Internal Gate** (**default in V**) | `refine_gate > 0.5` | `use_refine_gate=True` |

Both respect `[EOS]` freezing and early stopping via `stop_threshold`.



## 4. Sampling hyper-parameters  
| Hyper-param | Meaning | Default |
|-------------|---------|---------|
| `max_refinement_steps` | hard cap on iterations | 6 |
| `sampling_temperature` | softmax temperature during sampling | 1.2 |
| `min_refine_uncertainty` | entropy threshold (gate mode ignores this) | 0.1 |
| `stop_threshold` | early stop if < % tokens change | 0.02 |
| `use_refine_gate` | enable internal learned gate | **True** |



## 5. Tokenizer contract  
Required special IDs (auto-detected):  
```python
tokenizer.mask_token_id   # must exist
tokenizer.pad_token_id    # fallback: 0
tokenizer.eos_token_id    # fallback: sep_token_id, else None
```
Collision check is performed at model init.



## 6. Typical config (quick start)  
```python
config = ImplicitRefinementConfig(
            vocab_size=100,
            hidden_size=64,
            num_layers=2,
            max_seq_len=8,
            max_refinement_steps=3,
            stop_threshold=0.05,
            diversity_weight=0.1,
            sampling_temperature=1.0,
            use_refine_gate=use_gate
        )

        model = ImplicitRefinementModel(config, tokenizer=tokenizer)
        model.eval()
        model.init_teacher()
```



## 7. Strengths & limitations  
✅ **Pros**  
- **Non-autoregressive** → fully parallel sampling  
- **Learned refinement policy** (no hand-tuned entropy thresholds)  
- **Early stopping** enables variable-length outputs  
- **EMA teacher** stabilizes uncertainty (when gate is off)  
- Compatible with **any subword or character tokenizer**  

❌ **Cons**  
- Output length capped by `max_seq_len`  
- No explicit mechanism for **long-range coverage** or **input conditioning** (e.g., prompts)  


## 8. Citations  
- Ranger21 Optimizer:  
```bibtex
@article{wright2021ranger21,
  title={Ranger21: a synergistic deep learning optimizer}, 
  author={Wright, Less and Demeure, Nestor},
  year={2021},
  journal={arXiv preprint arXiv:2106.13731},
}
```

> **Note**: Imadarem V unifies refinement control **inside the model**, eliminating the need for external meta-policies. The internal gate is lightweight, end-to-end trainable, and simplifies deployment.
