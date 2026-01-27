import os
import torch
import random
import numpy as np
import pandas as pd
import datamol as dm
import safe as sf
from typing import List, Callable, Dict, Any, Tuple, Optional
from tqdm.auto import tqdm
from torch.optim import AdamW
import torch.nn.functional as F
from collections import defaultdict
from transformers import PreTrainedTokenizerFast
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

# SAFE imports
try:
    from safe.trainer.model import SAFEDoubleHeadsModel
    from safe.tokenizer import SAFETokenizer
    from safe.sample import SAFEDesign
except ImportError:
    pass

class Inception:
    def __init__(self, memory_size: int = 100, batch_size: int = 32):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory: List[Tuple[str, float, float]] = []

    def add(self, safe_strings: List[str], scores: List[float], prior_log_probs: List[float]):
        for s, sc, plp in zip(safe_strings, scores, prior_log_probs):
            # Only add high quality samples to memory
            if sc > 0.8: 
                self.memory.append((s, sc, plp))
        
        # Keep only unique SMILES in memory to maintain diversity
        unique_mem = {}
        for s, sc, plp in self.memory:
            if s not in unique_mem or sc > unique_mem[s][0]:
                unique_mem[s] = (sc, plp)
        
        self.memory = [(s, sc, plp) for s, (sc, plp) in unique_mem.items()]
        self.memory.sort(key=lambda x: x[1], reverse=True)
        self.memory = self.memory[:self.memory_size]

    def sample(self) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        if not self.memory:
            return [], None, None
        k = min(len(self.memory), self.batch_size)
        indices = np.random.choice(len(self.memory), k, replace=False)
        batch = [self.memory[i] for i in indices]
        
        safe_strs = [item[0] for item in batch]
        scores = torch.tensor([item[1] for item in batch], dtype=torch.float32)
        prior_log_probs = torch.tensor([item[2] for item in batch], dtype=torch.float32)
        return safe_strs, scores, prior_log_probs

class DiversityFilter:
    def __init__(self, bucket_size: int = 500, penalty_factor: float = 0.5):
        self.bucket_size = bucket_size
        self.penalty_factor = penalty_factor
        self.scaffold_counter = defaultdict(int)
        self.seen_smiles = set() 
    
    def reset(self):
        self.scaffold_counter.clear()
        self.seen_smiles.clear()

    def calculate_penalty(self, smiles: str) -> Tuple[float, bool]:
        if smiles in self.seen_smiles:
            return 0.0, False 
        self.seen_smiles.add(smiles)

        scaffold = "generic"
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
                if scaffold_mol:
                    scaffold = Chem.MolToSmiles(scaffold_mol)
        except:
            pass
        
        self.scaffold_counter[scaffold] += 1
        if self.scaffold_counter[scaffold] > self.bucket_size:
            return self.penalty_factor, True 
        return 1.0, True

class SafeReinventOptimizer:
    def __init__(
        self,
        model_path: str,
        lr: float = 2e-6,
        batch_size: int = 256,
        max_length: int = 100,
        sigma: float = 5.0,
        bucket_size: int = 500,
        memory_size: int = 100,
        penalty_factor: float = 0.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.max_length = max_length
        self.sigma = sigma
        self.model_path = model_path
        
        print(f"Loading SAFE Agent from {model_path}...")
        self.agent_designer = sf.SAFEDesign.load_default(verbose=False, model_dir=model_path)
        self.agent_model = self.agent_designer.model.to(self.device)
        self.agent_model.train() 
        
        print(f"Loading SAFE Prior from {model_path}...")
        self.prior_designer = sf.SAFEDesign.load_default(verbose=False, model_dir=model_path)
        self.prior_model = self.prior_designer.model.to(self.device)
        self.prior_model.eval()
        for param in self.prior_model.parameters():
            param.requires_grad = False
            
        self.tokenizer = self.agent_designer.tokenizer
        self.optimizer = AdamW(self.agent_model.parameters(), lr=self.lr, weight_decay=1e-2)
        self.diversity_filter = DiversityFilter(bucket_size=bucket_size, penalty_factor=penalty_factor)
        self.inception = Inception(memory_size=memory_size, batch_size=32)

    def _get_log_probs(self, model, input_ids, attention_mask):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        log_probs = F.log_softmax(shift_logits, dim=-1)
        target_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        shift_mask = attention_mask[..., 1:].contiguous()
        
        seq_sum_log_probs = (target_log_probs * shift_mask).sum(dim=1)
        seq_lengths = shift_mask.sum(dim=1)
        seq_lengths = torch.clamp(seq_lengths, min=1.0)
        
        return seq_sum_log_probs / seq_lengths

    def train_step(self, smiles_list: List[str], rewards: List[float], raw_scores: List[float]):
        if not smiles_list: return 0.0, 0.0, 0.0
        
        safe_strings = []
        valid_indices = []
        for i, smi in enumerate(smiles_list):
            try:
                mol = dm.to_mol(smi)
                if mol:
                    encoded_str = sf.encode(mol, canonical=True)
                    if encoded_str:
                        safe_strings.append(encoded_str)
                        valid_indices.append(i)
            except: continue
        
        if not safe_strings: return 0.0, 0.0, 0.0
        
        train_rewards = torch.tensor([float(rewards[i]) for i in valid_indices], device=self.device, dtype=torch.float32)
        memory_scores = [float(raw_scores[i]) for i in valid_indices]

        if hasattr(self.tokenizer, "tokenizer"):
             hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=self.tokenizer.tokenizer)
             if hasattr(self.tokenizer, "pad_token"): hf_tokenizer.pad_token = self.tokenizer.pad_token
             if hasattr(self.tokenizer, "eos_token"): hf_tokenizer.eos_token = self.tokenizer.eos_token
        else:
             hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=self.tokenizer)
        if hf_tokenizer.pad_token is None:
            hf_tokenizer.pad_token = hf_tokenizer.eos_token if hf_tokenizer.eos_token else '[PAD]'

        train_batch_size = 32
        total_samples = len(safe_strings)
        num_batches = (total_samples + train_batch_size - 1) // train_batch_size
        
        total_loss = 0.0
        total_agent_logp = 0.0
        total_prior_logp = 0.0
        
        indices = list(range(total_samples))
        random.shuffle(indices)
        
        for i in range(num_batches):
            batch_indices = indices[i * train_batch_size : (i + 1) * train_batch_size]
            batch_strs = [safe_strings[k] for k in batch_indices]
            batch_rewards = train_rewards[batch_indices]
            
            inputs = hf_tokenizer(batch_strs, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            agent_log_probs = self._get_log_probs(self.agent_model, input_ids, attention_mask)
            with torch.no_grad():
                prior_log_probs = self._get_log_probs(self.prior_model, input_ids, attention_mask)
            
            augmented_target = prior_log_probs + (self.sigma * batch_rewards)
            augmented_target = torch.clamp(augmented_target, max=0.0) 
            
            loss = (agent_log_probs - augmented_target).pow(2).mean()

            current_batch_raw_scores = [memory_scores[k] for k in batch_indices]
            self.inception.add(batch_strs, current_batch_raw_scores, prior_log_probs.cpu().tolist())
            
            mem_strs, mem_scores, mem_prior_log_probs = self.inception.sample()
            if mem_strs:
                mem_inputs = hf_tokenizer(mem_strs, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
                mem_ids = mem_inputs["input_ids"].to(self.device)
                mem_mask = mem_inputs["attention_mask"].to(self.device)
                
                mem_agent_log_probs = self._get_log_probs(self.agent_model, mem_ids, mem_mask)
                mem_scores = mem_scores.to(self.device)
                mem_prior_log_probs = mem_prior_log_probs.to(self.device)
                
                mem_target = mem_prior_log_probs + (self.sigma * mem_scores)
                mem_target = torch.clamp(mem_target, max=0.0)
                
                mem_loss = (mem_agent_log_probs - mem_target).pow(2).mean()
                loss = 0.5 * loss + 0.5 * mem_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent_model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_agent_logp += agent_log_probs.mean().item()
            total_prior_logp += prior_log_probs.mean().item()
            
            del inputs, input_ids, attention_mask, agent_log_probs, prior_log_probs
            torch.cuda.empty_cache()
            
        return (total_loss / num_batches), (total_agent_logp / num_batches), (total_prior_logp / num_batches)

    # Modified generate_batch to accept temperature
    def generate_batch(self, mode: str, temperature: float = 1.0, input_data: Any = None) -> List[str]:
        micro_batch = 64
        total_samples = self.batch_size
        all_smiles = []
        num_chunks = (total_samples + micro_batch - 1) // micro_batch
        
        # Use dynamic temperature
        gen_kwargs = {
            "max_length": self.max_length, 
            "do_sample": True, 
            "temperature": temperature, 
            "top_p": 0.95
        }

        for _ in range(num_chunks):
            current_n = min(micro_batch, total_samples - len(all_smiles))
            if current_n <= 0: break
            try:
                batch_res = []
                if mode == "random": batch_res = self.agent_designer.de_novo_generation(sanitize=True, n_samples_per_trial=current_n, n_trials=1, **gen_kwargs)
                elif mode == "motif": batch_res = self.agent_designer.motif_extension(sanitize=True, motif=input_data, n_samples_per_trial=current_n, n_trials=1, **gen_kwargs)
                elif mode == "linker": batch_res = self.agent_designer.linker_generation(*input_data, sanitize=True, n_samples_per_trial=current_n, n_trials=1, **gen_kwargs)
                elif mode == "scaffold": batch_res = self.agent_designer.scaffold_decoration(sanitize=True, scaffold=input_data, n_samples_per_trial=current_n, n_trials=1, **gen_kwargs)
                elif mode == "substructure": batch_res = self.agent_designer.substructure_generation(sanitize=True, core=input_data, n_samples_per_trial=current_n, n_trials=1, **gen_kwargs)
                if batch_res: all_smiles.extend([s for s in batch_res if s])
            except Exception: pass
            torch.cuda.empty_cache()
        return all_smiles

    def run(self, mode: str, score_fn: Callable, save_path: str, input_data: Any = None, target_score: float = 2.0, epochs: int = 50, save_freq: int = 5):
        os.makedirs(save_path, exist_ok=True)
        print(f"Starting REINVENT Optimization | Mode: {mode} | Batch: {self.batch_size} | Sigma: {self.sigma}")
        self.diversity_filter.reset()
        history = defaultdict(list)
        pbar = tqdm(range(1, epochs + 1))
        
        # Initial Temperature
        current_temp = 1.0
        
        for epoch in pbar:
            # Generate with current temperature
            raw_smiles = self.generate_batch(mode, current_temp, input_data)
            valid_smiles = [s for s in raw_smiles if s and dm.to_mol(s)]
            validity = len(valid_smiles) / len(raw_smiles) if raw_smiles else 0.0
            
            total_rewards = []
            raw_scores_list = []
            final_smiles = []
            new_count = 0
            
            if valid_smiles:
                try:
                    scores_dict = score_fn(valid_smiles)
                    raw_scores = np.zeros(len(valid_smiles))
                    for v in scores_dict.values():
                        if len(v) == len(valid_smiles): raw_scores += np.array(v)
                    
                    for i, smi in enumerate(valid_smiles):
                        penalty, is_new = self.diversity_filter.calculate_penalty(smi)
                        if is_new: new_count += 1
                        
                        final_s = raw_scores[i] * penalty
                        final_smiles.append(smi)
                        total_rewards.append(final_s)
                        raw_scores_list.append(raw_scores[i])
                except Exception as e: print(f"Score Error: {e}")
            
            loss, avg_agent_logp, avg_prior_logp = self.train_step(final_smiles, total_rewards, raw_scores_list)
            
            mean_rwd = np.mean(total_rewards) if total_rewards else 0.0
            mean_raw = np.mean(raw_scores_list) if raw_scores_list else 0.0
            
            # --- Dynamic Exploration Logic ---
            # Calculate gap between Raw Score and Reward (Penalized Score)
            # Large gap => Many molecules are being penalized => High repetition
            repetition_gap = mean_raw - mean_rwd
            
            # Adjust Temperature based on Repetition Gap
            if repetition_gap > 0.1: 
                # Repetition is high, heat up to force exploration
                current_temp = min(current_temp + 0.1, 1.5)
            elif repetition_gap < 0.05:
                # Diversity is good, cool down to exploit
                current_temp = max(current_temp - 0.05, 1.0)
            
            history["epoch"].append(epoch)
            history["mean_score"].append(mean_rwd)
            history["mean_raw_score"].append(mean_raw)
            history["mean_loss"].append(loss)
            history["validity"].append(validity)
            history["temperature"].append(current_temp) # Log temp
            
            # Updated Pbar: Shows Temp (T)
            pbar.set_description(f"Raw:{mean_raw:.2f}|Rwd:{mean_rwd:.2f}|Gap:{repetition_gap:.2f}|T:{current_temp:.1f}")
            
            if epoch % save_freq == 0:
                self.agent_model.save_pretrained(os.path.join(save_path, f"checkpoint_{epoch}"))
                self.tokenizer.save_pretrained(os.path.join(save_path, f"checkpoint_{epoch}"))
                pd.DataFrame(history).to_csv(os.path.join(save_path, "metrics.csv"), index=False)

        self.agent_model.save_pretrained(os.path.join(save_path, "final_model"))
        self.tokenizer.save_pretrained(os.path.join(save_path, "final_model"))
        pd.DataFrame(history).to_csv(os.path.join(save_path, "final_metrics.csv"), index=False)
        print("Done.")
