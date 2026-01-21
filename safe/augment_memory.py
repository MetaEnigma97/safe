import os
import torch
import random
import copy
import numpy as np
import pandas as pd
import datamol as dm
import safe as sf
from typing import List, Callable, Optional, Union, Dict, Any, Tuple
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast

# SAFE imports
# Ensure these are available in your environment
try:
    from safe.trainer.model import SAFEDoubleHeadsModel
    from safe.tokenizer import SAFETokenizer
    from safe.sample import SAFEDesign
except ImportError:
    pass # Assume they are available at runtime

class MemoryDataset(Dataset):
    def __init__(self, safe_strings: List[str]):
        self.safe_strings = safe_strings
    def __len__(self):
        return len(self.safe_strings)
    def __getitem__(self, idx):
        return self.safe_strings[idx]

class SafeAugmentedOptimizer:
    def __init__(
        self,
        model_path: str,
        lr: float = 5e-5,
        batch_size: int = 1024,
        max_length: int = 100,
        memory_size: int = 100,
        entropy_weight: float = 0.0,
        exploration_prob: float = 0.1, 
        augmentation_rounds: int = 0, # NEW: Enable Augmented Memory (Randomized SMILES)
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.memory_size = memory_size
        self.lr = lr
        self.batch_size = batch_size
        self.max_length = max_length
        self.entropy_weight = entropy_weight
        self.exploration_prob = exploration_prob
        self.augmentation_rounds = augmentation_rounds
        self.model_path = model_path
        
        print(f"Loading SAFE model from {model_path}...")
        self.designer = sf.SAFEDesign.load_default(verbose=False, model_dir=model_path)
        self.model = self.designer.model.to(self.device)
        self.tokenizer = self.designer.tokenizer
        
        # Backup original for exploration
        self.original_designer = sf.SAFEDesign.load_default(verbose=False, model_dir=model_path)
        self.original_model = self.original_designer.model.to(self.device)
        self.original_model.eval()
        for param in self.original_model.parameters():
            param.requires_grad = False
        
        # Memory structure: List of (score, smiles, safe_str)
        self.memory: List[Tuple[float, str, str]] = []
        self.seen_smiles = set()
        
        # Optimizer - Initialize here if you want persistence, 
        # but for short fine-tuning steps, re-init is often fine. 
        # We will keep it local to fine_tune to match typical "few-shot" logic,
        # but you can move it here if you want momentum to carry over steps.

    def update_memory(self, candidates: List[Tuple[float, str, str]]) -> int:
        """
        Update memory with new candidates.
        Corrected logic: Collect all valid candidates, merge with memory, sort, then prune.
        Returns: Number of NEW items that survived into the memory.
        """
        # Filter invalid
        valid_candidates = [c for c in candidates if c[1] is not None and c[2] is not None]
        if not valid_candidates:
            return 0

        # Identify potential new additions (not currently in memory)
        # Note: We allow updating if the same SMILES comes with a BETTER score (though rare in this setup)
        # For simplicity, we stick to the set check for novelty.
        
        current_memory_smiles = set(self.seen_smiles)
        
        # Merge candidates into a temporary list
        # We process candidates to ensure we only keep the best score for any duplicate SMILES within the batch
        batch_best = {}
        for score, smi, safe_str in valid_candidates:
            if smi not in batch_best:
                batch_best[smi] = (score, smi, safe_str)
            else:
                if score > batch_best[smi][0]:
                    batch_best[smi] = (score, smi, safe_str)
        
        new_items = list(batch_best.values())
        
        # Add new unique items to memory
        added_candidates = []
        for item in new_items:
            # If not seen, add. 
            # If seen, strictly we could update if score improves, but we'll skip for now to keep seen_smiles simple.
            if item[1] not in self.seen_smiles:
                self.memory.append(item)
                self.seen_smiles.add(item[1])
                added_candidates.append(item[1])

        # Sort Memory by score (descending)
        self.memory.sort(key=lambda x: x[0], reverse=True)
        
        # Prune if over size
        if len(self.memory) > self.memory_size:
            kept_memory = self.memory[:self.memory_size]
            
            # Identify what was removed to update seen_smiles
            # (Though seen_smiles typically tracks lifetime seen, for memory we usually track "currently in memory")
            # The reference implementation tracks currently in memory.
            
            self.memory = kept_memory
            
            # Rebuild seen set to match current memory exactly
            self.seen_smiles = set([x[1] for x in self.memory])
            
            # Calculate how many of the *newly added* candidates survived the cut
            survived_new_smiles = self.seen_smiles.intersection(set(added_candidates))
            return len(survived_new_smiles)
        else:
            return len(added_candidates)

    def _augment_safe_strings(self, smiles_list: List[str]) -> List[str]:
        """
        Generate randomized variations of molecules and encode them to SAFE.
        This mimics the 'Augmented Memory' approach.
        """
        augmented_data = []
        for smi in smiles_list:
            try:
                mol = dm.to_mol(smi)
                if mol is None: continue
                
                # Generate 'augmentation_rounds' randomized SMILES
                for _ in range(self.augmentation_rounds):
                    # Randomize SMILES
                    rand_smi = dm.to_smiles(mol, canonical=False, randomize=True)
                    if rand_smi:
                        rand_mol = dm.to_mol(rand_smi)
                        # Attempt to encode non-canonically if SAFE supports it, 
                        # or just rely on randomized SMILES input producing slightly different tokens 
                        # if the tokenizer is sensitive to it.
                        # Note: sf.encode(canonical=True) might force them back to same string.
                        # We try canonical=False to allow variation.
                        safe_str = sf.encode(rand_mol, canonical=False) 
                        augmented_data.append(safe_str)
            except Exception:
                continue
        return augmented_data

    def mode_collapse_guard(self):
        """
        Check if memory has collapsed to a single solution (low diversity).
        If so, clear memory to force exploration.
        """
        if len(self.memory) < self.memory_size // 2:
            return

        scores = [x[0] for x in self.memory]
        unique_scores = set(scores)
        
        # If all scores are identical (and not just 1 item), assume collapse
        if len(unique_scores) == 1 and len(scores) > 1:
            print("!!! Mode Collapse Detected - Purging Memory !!!")
            self.memory = []
            self.seen_smiles = set()

    def fine_tune_on_memory(self, epochs: int = 5, train_batch_size: int = 32):
        if len(self.memory) < 4: return

        # 1. Prepare Training Data
        # Always include the canonical strings currently in memory
        train_strs = [item[2] for item in self.memory]
        
        # 2. Data Augmentation (Reference Feature)
        if self.augmentation_rounds > 0:
            memory_smiles = [item[1] for item in self.memory]
            augmented_strs = self._augment_safe_strings(memory_smiles)
            train_strs.extend(augmented_strs)

        self.model.train()
        # Using a fresh optimizer for the fine-tuning step (expert iteration style)
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        
        dataset = MemoryDataset(train_strs)
        # Drop last to avoid very small batches which can cause unstable gradients
        loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, drop_last=len(dataset) > train_batch_size)
        
        for _ in range(epochs):
            for batch_strs in loader:
                # Tokenization logic compatible with SAFE
                if hasattr(self.tokenizer, "tokenizer"):
                    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=self.tokenizer.tokenizer)
                    if hasattr(self.tokenizer, "pad_token"): hf_tokenizer.pad_token = self.tokenizer.pad_token
                    if hasattr(self.tokenizer, "eos_token"): hf_tokenizer.eos_token = self.tokenizer.eos_token
                    if hasattr(self.tokenizer, "bos_token"): hf_tokenizer.bos_token = self.tokenizer.bos_token
                else:
                    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=self.tokenizer)

                if hf_tokenizer.pad_token is None:
                    if hf_tokenizer.eos_token is not None:
                        hf_tokenizer.pad_token = hf_tokenizer.eos_token
                    else:
                        hf_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

                inputs = hf_tokenizer(
                    batch_strs, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=self.max_length,
                    add_special_tokens=True
                )
                
                input_ids = inputs["input_ids"].to(self.device)
                labels = input_ids.clone()
                if hasattr(hf_tokenizer, "pad_token_id"):
                    labels[labels == hf_tokenizer.pad_token_id] = -100
                
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                
                # Entropy Regularization
                if self.entropy_weight > 0:
                    logits = outputs.logits
                    probs = F.softmax(logits, dim=-1)
                    log_probs = F.log_softmax(logits, dim=-1)
                    # Calculate entropy
                    entropy = -torch.sum(probs * log_probs, dim=-1).mean()
                    loss = loss - (self.entropy_weight * entropy)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
        
        self.model.eval()

    def generate_batch(self, mode: str, input_data: Any = None) -> Tuple[List[str], bool]:
        is_exploring = random.random() < self.exploration_prob
        current_designer = self.original_designer if is_exploring else self.designer
        temp = 1.2 if is_exploring else 1.0 
        
        gen_kwargs = {
            "max_length": self.max_length,
            "do_sample": True,
            "temperature": temp
        }
        
        try:
            if mode == "random":
                generated_smiles = current_designer.de_novo_generation(
                    sanitize=True, n_samples_per_trial=self.batch_size, n_trials=5, **gen_kwargs
                )
            elif mode == "motif":
                if not input_data: raise ValueError("Input motif required")
                generated_smiles = current_designer.motif_extension(
                    sanitize=True, motif=input_data, n_samples_per_trial=self.batch_size, n_trials=5, **gen_kwargs
                )
            elif mode == "linker":
                if not input_data: raise ValueError("Input fragments required")
                generated_smiles = current_designer.linker_generation(
                    *input_data, sanitize=True, n_samples_per_trial=self.batch_size, n_trials=5, **gen_kwargs
                )
            elif mode == "scaffold":
                if not input_data: raise ValueError("Input scaffold required")
                generated_smiles = current_designer.scaffold_decoration(
                    sanitize=True, scaffold=input_data, n_samples_per_trial=self.batch_size, n_trials=5, **gen_kwargs
                )
            elif mode == "substructure":
                if not input_data: raise ValueError("Input core required")
                generated_smiles = current_designer.substructure_generation(
                    sanitize=True, core=input_data, n_samples_per_trial=self.batch_size, n_trials=5, **gen_kwargs
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")
        except Exception:
            return [], is_exploring

        return [s for s in generated_smiles if s is not None], is_exploring

    def run(
        self,
        mode: str,
        score_fn: Callable[[List[str]], Dict[str, List[float]]],
        save_path: str,
        input_data: Any = None,
        target_score: float = 2.0,
        epochs: int = 50,
        save_freq: int = 5,
        ft_epochs_per_step: int = 5,
        train_batch_size: int = 256
    ):
        os.makedirs(save_path, exist_ok=True)
        print(f"Starting | Mode: {mode} | Batch: {self.batch_size} | LR: {self.lr} | Augment: {self.augmentation_rounds}")
        
        history = {
            "epoch": [], 
            "mean_reward": [], 
            "validity": [], 
            "best_memory_score": [], 
            "memory_size": [], 
            "is_exploring": [],
            "new_in_memory_pct": [],
            "perfect_score_pct": []
        }
        
        pbar = tqdm(range(1, epochs + 1))
        
        for epoch in pbar:
            # 1. Mode Collapse Check
            self.mode_collapse_guard()

            # 2. Generation
            raw_smiles_batch, is_exploring = self.generate_batch(mode, input_data)
            
            candidates = []
            valid_smiles_list = []
            valid_safe_strs = []
            
            # 3. Validation & Encoding
            for smi in raw_smiles_batch:
                if smi is None: continue
                try:
                    mol = dm.to_mol(smi)
                    if mol is not None:
                        safe_str = sf.encode(mol, canonical=True)
                        valid_smiles_list.append(smi)
                        valid_safe_strs.append(safe_str)
                except:
                    continue
            
            # 4. Scoring
            total_generated = len(raw_smiles_batch)
            validity = len(valid_smiles_list) / total_generated if total_generated > 0 else 0.0
            
            mean_reward = 0.0
            perfect_count = 0
            
            if valid_smiles_list:
                try:
                    scores_dict = score_fn(valid_smiles_list)
                    aggregated_rewards = [0.0] * len(valid_smiles_list)
                    for _, scores in scores_dict.items():
                        if len(scores) == len(valid_smiles_list):
                            for i in range(len(valid_smiles_list)):
                                aggregated_rewards[i] += scores[i]
                    
                    for i, (smi, safe_str, score) in enumerate(zip(valid_smiles_list, valid_safe_strs, aggregated_rewards)):
                        candidates.append((score, smi, safe_str))
                        if score >= target_score:
                            perfect_count += 1
                            
                    mean_reward = np.mean(aggregated_rewards)
                except Exception:
                    pass
            
            perfect_score_pct = perfect_count / len(valid_smiles_list) if valid_smiles_list else 0.0

            # 5. Update Memory
            new_items_count = self.update_memory(candidates)
            new_in_memory_pct = new_items_count / len(valid_smiles_list) if valid_smiles_list else 0.0
            
            best_mem_score = self.memory[0][0] if self.memory else 0.0
            
            # 6. Logging
            history["epoch"].append(epoch)
            history["mean_reward"].append(mean_reward)
            history["validity"].append(validity)
            history["best_memory_score"].append(best_mem_score)
            history["memory_size"].append(len(self.memory))
            history["is_exploring"].append(is_exploring)
            history["new_in_memory_pct"].append(new_in_memory_pct)
            history["perfect_score_pct"].append(perfect_score_pct)
            
            # 7. Fine-Tuning (Experience Replay)
            if len(self.memory) >= 4:
                self.fine_tune_on_memory(epochs=ft_epochs_per_step, train_batch_size=train_batch_size)
            
            explore_tag = "[EXP]" if is_exploring else "     "
            pbar.set_description(
                f"{explore_tag} E:{epoch} | Rwd:{mean_reward:.2f} | Perfect:{perfect_score_pct:.1%} | NewMem:{new_in_memory_pct:.1%} | Best:{best_mem_score:.2f}"
            )

            if epoch % save_freq == 0:
                ckpt_path = os.path.join(save_path, f"checkpoint_{epoch}")
                self.model.save_pretrained(ckpt_path)
                self.tokenizer.save_pretrained(ckpt_path)
                pd.DataFrame(history).to_csv(os.path.join(save_path, "training_metrics.csv"), index=False)
                pd.DataFrame(self.memory, columns=["score", "smiles", "safe"]).to_csv(
                    os.path.join(save_path, f"memory_{epoch}.csv"), index=False
                )

        final_path = os.path.join(save_path, "final_model")
        self.model.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)
        pd.DataFrame(history).to_csv(os.path.join(save_path, "final_metrics.csv"), index=False)
        if self.memory:
            pd.DataFrame(self.memory, columns=["score", "smiles", "safe"]).to_csv(
                os.path.join(save_path, "final_memory.csv"), index=False
            )
        print(f"Done. Results saved to {save_path}")
