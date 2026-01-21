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
from safe.trainer.model import SAFEDoubleHeadsModel
from safe.tokenizer import SAFETokenizer
from safe.sample import SAFEDesign

class DiversityFilter:
    def __init__(self, bucket_size: int = 500): # Increased default bucket size
        self.bucket_size = bucket_size
        self.scaffold_counter = defaultdict(int)
        self.seen_smiles = set() 
    
    def reset(self):
        self.scaffold_counter.clear()
        self.seen_smiles.clear()

    def calculate_penalty(self, smiles: str) -> Tuple[float, bool]:
        # 1. Uniqueness check
        if smiles in self.seen_smiles:
            is_new = False
            pass 
        else:
            is_new = True
            self.seen_smiles.add(smiles)

        # 2. Scaffold calculation
        scaffold = "generic"
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
            if scaffold_mol:
                scaffold = Chem.MolToSmiles(scaffold_mol)
        # 3. Bucket check
        self.scaffold_counter[scaffold] += 1
        
        # If bucket is full, penalty is 0.0
        if self.scaffold_counter[scaffold] > self.bucket_size:
            return 0.0, is_new
            
        return 1.0, is_new


class SafeReinventOptimizer:
    def __init__(
        self,
        model_path: str,
        lr: float = 1e-5,
        batch_size: int = 1024,
        max_length: int = 80,
        sigma: float = 60.0,
        bucket_size: int = 500, # Increased
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
        self.optimizer = AdamW(self.agent_model.parameters(), lr=self.lr)
        self.diversity_filter = DiversityFilter(bucket_size=bucket_size)

    def _get_log_probs(self, model, input_ids, attention_mask):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits 
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        log_probs = F.log_softmax(shift_logits, dim=-1)
        target_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        shift_mask = attention_mask[..., 1:].contiguous()
        seq_log_probs = (target_log_probs * shift_mask).sum(dim=1)
        
        return seq_log_probs

    def train_step(self, smiles_list: List[str], rewards: List[float]):
        if not smiles_list: return 0.0
            
        safe_strings = []
        valid_indices = []
        
        for i, smi in enumerate(smiles_list):
            try:
                mol = dm.to_mol(smi)
                if mol:
                    try:
                        encoded_str = sf.encode(mol, canonical=True)
                        if encoded_str:
                            safe_strings.append(encoded_str)
                            valid_indices.append(i)
                    except Exception:
                        continue
            except Exception:
                continue
        
        if not safe_strings: return 0.0
            
        valid_rewards = torch.tensor(
            [float(rewards[i]) for i in valid_indices], 
            device=self.device,
            dtype=torch.float32
        )
        
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
            safe_strings,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        agent_log_probs = self._get_log_probs(self.agent_model, input_ids, attention_mask)
        with torch.no_grad():
            prior_log_probs = self._get_log_probs(self.prior_model, input_ids, attention_mask)
            
        score_term = self.sigma * valid_rewards
        loss = (agent_log_probs - prior_log_probs - score_term).pow(2).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent_model.parameters(), 1.0)
        self.optimizer.step()
        
        loss_val = loss.item()
        
        del inputs, input_ids, attention_mask, agent_log_probs, prior_log_probs, loss, score_term, valid_rewards
        torch.cuda.empty_cache() 
        
        return loss_val

    def generate_batch(self, mode: str, input_data: Any = None) -> List[str]:
        micro_batch = 64
        total_samples = self.batch_size
        all_smiles = []
        num_chunks = (total_samples + micro_batch - 1) // micro_batch
        
        gen_kwargs = {
            "max_length": self.max_length,
            "do_sample": True,
            "temperature": 1.0, 
            "top_p": 0.95
        }

        for _ in range(num_chunks):
            current_n = min(micro_batch, total_samples - len(all_smiles))
            if current_n <= 0: break

            try:
                batch_res = []
                if mode == "random":
                    batch_res = self.agent_designer.de_novo_generation(
                        sanitize=True, n_samples_per_trial=current_n, n_trials=1, **gen_kwargs
                    )
                elif mode == "motif":
                    batch_res = self.agent_designer.motif_extension(
                        sanitize=True, motif=input_data, n_samples_per_trial=current_n, n_trials=1, **gen_kwargs
                    )
                elif mode == "linker":
                    batch_res = self.agent_designer.linker_generation(
                        *input_data, sanitize=True, n_samples_per_trial=current_n, n_trials=1, **gen_kwargs
                    )
                elif mode == "scaffold":
                    batch_res = self.agent_designer.scaffold_decoration(
                        sanitize=True, scaffold=input_data, n_samples_per_trial=current_n, n_trials=1, **gen_kwargs
                    )
                elif mode == "substructure":
                    batch_res = self.agent_designer.substructure_generation(
                        sanitize=True, core=input_data, n_samples_per_trial=current_n, n_trials=1, **gen_kwargs
                    )
                
                if batch_res:
                    all_smiles.extend([s for s in batch_res if s])
            except Exception:
                pass
            
            torch.cuda.empty_cache()

        return all_smiles

    def run(
        self,
        mode: str,
        score_fn: Callable[[List[str]], Dict[str, List[float]]],
        save_path: str,
        input_data: Any = None,
        target_score: float = 2.0,
        epochs: int = 50,
        save_freq: int = 5
    ):
        os.makedirs(save_path, exist_ok=True)
        print(f"Starting REINVENT Optimization | Mode: {mode} | Batch: {self.batch_size}")
        
        self.diversity_filter.reset()
        
        history = {
            "epoch": [], "mean_score": [], "mean_loss": [], "validity": [], 
            "new_molecules_pct": [], "perfect_score_pct": []
        }
        
        pbar = tqdm(range(1, epochs + 1))
        
        for epoch in pbar:
            # 1. Generate
            raw_smiles = self.generate_batch(mode, input_data)
            
            # Debug: Check generation diversity
            if raw_smiles:
                unique_gen = len(set(raw_smiles))
                if unique_gen < 5:
                    print(f"\nWARNING: Mode Collapse? Only {unique_gen} unique SMILES in batch of {len(raw_smiles)}")
            
            # 2. Validate
            valid_smiles = []
            for s in raw_smiles:
                if s and dm.to_mol(s): valid_smiles.append(s)
            
            total_gen = len(raw_smiles)
            validity = len(valid_smiles) / total_gen if total_gen > 0 else 0.0
            
            total_rewards = []
            final_smiles = []
            new_count = 0
            perfect_count = 0
            
            if valid_smiles:
                scores_dict = score_fn(valid_smiles)
                
                raw_scores = np.zeros(len(valid_smiles))
                for k, v in scores_dict.items():
                    if len(v) == len(valid_smiles):
                        raw_scores += np.array(v)
                
                for i, smi in enumerate(valid_smiles):
                    penalty, is_new = self.diversity_filter.calculate_penalty(smi)
                    
                    if is_new: new_count += 1
                    
                    final_s = raw_scores[i] * penalty
                    if raw_scores[i] >= target_score: perfect_count += 1
                    
                    final_smiles.append(smi)
                    total_rewards.append(final_s)
                        
            # 3. Train
            loss = self.train_step(final_smiles, total_rewards)
            
            # 4. Stats
            mean_rwd = np.mean(total_rewards) if total_rewards else 0.0
            new_pct = new_count / len(valid_smiles) if valid_smiles else 0.0
            perfect_pct = perfect_count / len(valid_smiles) if valid_smiles else 0.0
            
            history["epoch"].append(epoch)
            history["mean_score"].append(mean_rwd)
            history["mean_loss"].append(loss)
            history["validity"].append(validity)
            history["new_molecules_pct"].append(new_pct)
            history["perfect_score_pct"].append(perfect_pct)
            
            pbar.set_description(
                f"Rwd: {mean_rwd:.2f} | Loss: {loss:.2f} | Val: {validity:.0%} | New: {new_pct:.0%} | Perf: {perfect_pct:.0%}"
            )
            
            if epoch % save_freq == 0:
                ckpt_path = os.path.join(save_path, f"checkpoint_{epoch}")
                self.agent_model.save_pretrained(ckpt_path)
                self.tokenizer.save_pretrained(ckpt_path)
                pd.DataFrame(history).to_csv(os.path.join(save_path, "metrics.csv"), index=False)

        final_path = os.path.join(save_path, "final_model")
        self.agent_model.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)
        pd.DataFrame(history).to_csv(os.path.join(save_path, "final_metrics.csv"), index=False)
        print("Done.")
