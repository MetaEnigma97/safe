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
from safe.trainer.model import SAFEDoubleHeadsModel
from safe.tokenizer import SAFETokenizer
from safe.sample import SAFEDesign

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
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.memory_size = memory_size
        self.lr = lr
        self.batch_size = batch_size
        self.max_length = max_length
        self.entropy_weight = entropy_weight
        self.exploration_prob = exploration_prob
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
        
        self.memory: List[Tuple[float, str, str]] = []
        self.seen_smiles = set()

    def update_memory(self, candidates: List[Tuple[float, str, str]]) -> int:
        """
        Update memory and return the number of NEW items added.
        """
        added_count = 0
        for score, smi, safe_str in candidates:
            # Check if unique and valid
            if smi not in self.seen_smiles and safe_str is not None:
                # Basic check: only add if memory not full OR score is better than worst in memory
                if len(self.memory) < self.memory_size:
                    self.memory.append((score, smi, safe_str))
                    self.seen_smiles.add(smi)
                    added_count += 1
                else:
                    # Memory is full, check if better than worst
                    # Assuming memory is sorted descending, last is worst
                    worst_score = self.memory[-1][0]
                    if score > worst_score:
                        self.memory.append((score, smi, safe_str))
                        self.seen_smiles.add(smi)
                        added_count += 1
        
        # Re-sort and Prune
        self.memory.sort(key=lambda x: x[0], reverse=True)
        
        if len(self.memory) > self.memory_size:
            removed = self.memory[self.memory_size:]
            for _, smi, _ in removed:
                if smi in self.seen_smiles:
                    self.seen_smiles.remove(smi)
            self.memory = self.memory[:self.memory_size]
            
        return added_count

    def fine_tune_on_memory(self, epochs: int = 5, train_batch_size: int = 32):
        if len(self.memory) < 4: return

        self.model.train()
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        
        train_data = [item[2] for item in self.memory]
        dataset = MemoryDataset(train_data)
        loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)
        
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
                nll_loss = outputs.loss
                
                loss = nll_loss
                if self.entropy_weight > 0:
                    logits = outputs.logits
                    probs = F.softmax(logits, dim=-1)
                    log_probs = F.log_softmax(logits, dim=-1)
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
        target_score: float = 2.0, # NEW: The score considered "Perfect"
        epochs: int = 50,
        save_freq: int = 5,
        ft_epochs_per_step: int = 5,
        train_batch_size: int = 256
    ):
        os.makedirs(save_path, exist_ok=True)
        print(f"Starting | Mode: {mode} | Batch: {self.batch_size} | LR: {self.lr}")
        
        history = {
            "epoch": [], 
            "mean_reward": [], 
            "validity": [], 
            "best_memory_score": [], 
            "memory_size": [], 
            "is_exploring": [],
            "new_in_memory_pct": [], # NEW: % of batch added to memory
            "perfect_score_pct": []  # NEW: % of batch hitting target score
        }
        
        pbar = tqdm(range(1, epochs + 1))
        
        for epoch in pbar:
            raw_smiles_batch, is_exploring = self.generate_batch(mode, input_data)
            
            candidates = []
            valid_smiles_list = []
            valid_safe_strs = []
            
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
            
            # --- Metrics Calculation ---
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
                        if score >= target_score: # Count perfect scores
                            perfect_count += 1
                            
                    mean_reward = np.mean(aggregated_rewards)
                except Exception:
                    pass
            
            perfect_score_pct = perfect_count / len(valid_smiles_list) if valid_smiles_list else 0.0

            # Update Memory & Count Freshness
            new_items_count = self.update_memory(candidates)
            
            # Calculate freshness relative to valid molecules generated
            new_in_memory_pct = new_items_count / len(valid_smiles_list) if valid_smiles_list else 0.0
            
            best_mem_score = self.memory[0][0] if self.memory else 0.0
            
            # History
            history["epoch"].append(epoch)
            history["mean_reward"].append(mean_reward)
            history["validity"].append(validity)
            history["best_memory_score"].append(best_mem_score)
            history["memory_size"].append(len(self.memory))
            history["is_exploring"].append(is_exploring)
            history["new_in_memory_pct"].append(new_in_memory_pct)
            history["perfect_score_pct"].append(perfect_score_pct)
            
            if len(self.memory) >= 4:
                self.fine_tune_on_memory(epochs=ft_epochs_per_step, train_batch_size=train_batch_size)
            
            explore_tag = "[EXP]" if is_exploring else "     "
            # Updated Pbar Description
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
