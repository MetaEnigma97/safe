# safe/augment_memory.py
# Augmented memory optimizer with memory pruning strategies (diversity/age/novelty)
# Depends: rdkit, datamol, safe, transformers, torch, numpy, pandas

import os
import random
import time
from typing import List, Tuple, Callable, Optional, Any, Dict

import numpy as np
import pandas as pd
import datamol as dm
import safe as sf

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from transformers import PreTrainedTokenizerFast
from tqdm.auto import tqdm

# rdkit for fingerprints / tanimoto
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

# SAFE imports (may raise at static time; assume available at runtime)
try:
    from safe.trainer.model import SAFEDoubleHeadsModel
    from safe.tokenizer import SAFETokenizer
    from safe.sample import SAFEDesign
except ImportError:
    SAFEDesign = None

# ----------------------------
# Utilities: fingerprinting
# ----------------------------
def mol_from_smiles(smiles: str):
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None

def morgan_fp_bits(mol, radius: int = 2, n_bits: int = 2048):
    if mol is None:
        return None
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    except Exception:
        return None

def tanimoto_sim(fp1, fp2):
    if fp1 is None or fp2 is None:
        return 0.0
    try:
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception:
        return 0.0

# ----------------------------
# Dataset / collate
# ----------------------------
class MemoryDataset(Dataset):
    def __init__(self, entries: List[Tuple[str, float]]):
        self.entries = entries
    def __len__(self):
        return len(self.entries)
    def __getitem__(self, idx):
        return self.entries[idx]

def collate_strings_and_weights(batch: List[Tuple[str, float]]):
    strings = [b[0] for b in batch]
    weights = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    return strings, weights

# ----------------------------
# Main optimizer
# ----------------------------
class SafeAugmentedOptimizer:
    def __init__(
        self,
        model_path: str,
        lr: float = 1e-5,
        batch_size: int = 1024,
        max_length: int = 100,
        memory_size: int = 200,
        entropy_weight: float = 0.0,
        exploration_prob: float = 0.1,
        augmentation_rounds: int = 0,
        device: Optional[str] = None,
        kl_weight: float = 0.1,
        weight_temp: float = 1.0,
        prioritized_replay: bool = True,
        fp_radius: int = 2,
        fp_bits: int = 2048,
        admission_by_threshold: bool = True,
        absolute_score_threshold: Optional[float] = None,
        allow_memory_growth: bool = False,
        unique_sampling: bool = False,
        double_loop_augment: bool = False,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.memory_size = int(memory_size)
        self.lr = lr
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.entropy_weight = entropy_weight
        self.exploration_prob = exploration_prob
        self.augmentation_rounds = int(augmentation_rounds)
        self.model_path = model_path
        self.kl_weight = float(kl_weight)
        self.weight_temp = float(weight_temp)
        self.prioritized_replay = bool(prioritized_replay)

        # fingerprint params
        self.fp_radius = fp_radius
        self.fp_bits = fp_bits

        self.admission_by_threshold = bool(admission_by_threshold)
        self.absolute_score_threshold = None if absolute_score_threshold is None else float(absolute_score_threshold)
        self.allow_memory_growth = bool(allow_memory_growth)

        # extra options
        self.unique_sampling = bool(unique_sampling)
        self.double_loop_augment = bool(double_loop_augment)

        print(f"[SafeAugmentedOptimizer] Loading SAFE designer from {model_path} ...")
        self.designer = sf.SAFEDesign.load_default(verbose=False, model_dir=model_path)
        self.model = self.designer.model.to(self.device)
        self.tokenizer = self.designer.tokenizer

        # frozen original for KL regularization & exploration and prior likelihoods
        self.original_designer = sf.SAFEDesign.load_default(verbose=False, model_dir=model_path)
        self.original_model = self.original_designer.model.to(self.device)
        self.original_model.eval()
        for p in self.original_model.parameters():
            p.requires_grad = False

        # HF tokenizer wrapper once
        if hasattr(self.tokenizer, "tokenizer"):
            self.hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=self.tokenizer.tokenizer)
        else:
            self.hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=self.tokenizer)
        if self.hf_tokenizer.pad_token is None:
            if self.hf_tokenizer.eos_token is not None:
                self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token
            else:
                self.hf_tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # Memory: dict smiles -> dict with keys: score, smiles, safe, added_at, fp
        self.memory_index: Dict[str, Dict[str, Any]] = {}
        self.memory_list: List[Dict[str, Any]] = []

        print(f"[SafeAugmentedOptimizer] Initialized on {self.device} | mem_size={self.memory_size} | augment_rounds={self.augmentation_rounds} | unique_sampling={self.unique_sampling} | double_loop_augment={self.double_loop_augment}")

    # ----------------------------
    # Internal helpers for memory
    # ----------------------------
    def _make_entry(self, score: float, smiles: str, safe_str: str, epoch_or_time: Optional[float] = None):
        # compute fingerprint
        fp = None
        try:
            mol = mol_from_smiles(smiles)
            fp = morgan_fp_bits(mol, radius=self.fp_radius, n_bits=self.fp_bits)
        except Exception:
            fp = None
        return {"score": float(score), "smiles": smiles, "safe": safe_str, "added_at": epoch_or_time or time.time(), "fp": fp}

    def _rebuild_memory_list(self):
        # Rebuild memory_list from memory_index and enforce capacity
        self.memory_list = sorted(self.memory_index.values(), key=lambda x: x["score"], reverse=True)
        if len(self.memory_list) > self.memory_size:
            survivors = self.memory_list[: self.memory_size]
            self.memory_index = {item["smiles"]: item for item in survivors}
            self.memory_list = survivors

    def prune_memory_diversity(self, tanimoto_threshold: float = 0.8):
        kept = []
        for item in sorted(self.memory_list, key=lambda x: x["score"], reverse=True):
            fp = item.get("fp")
            too_similar = False
            for k in kept:
                if k["fp"] is None or fp is None:
                    continue
                if tanimoto_sim(k["fp"], fp) >= tanimoto_threshold:
                    too_similar = True
                    break
            if not too_similar:
                kept.append(item)
            if len(kept) >= self.memory_size:
                break
        self.memory_list = kept[: self.memory_size]
        self.memory_index = {item["smiles"]: item for item in self.memory_list}

    def prune_memory_age(self, keep_recent: int = None):
        kr = keep_recent or self.memory_size
        items = sorted(self.memory_list, key=lambda x: x["added_at"], reverse=True)
        kept = items[:kr]
        self.memory_list = kept
        self.memory_index = {item["smiles"]: item for item in kept}

    def prune_memory_keep_top(self):
        self._rebuild_memory_list()

    def try_novelty_replace(self, candidate_item: Dict[str, Any], novelty_threshold: float = 0.6, min_score_delta: float = 0.0) -> bool:
        self._rebuild_memory_list()
        if len(self.memory_list) < self.memory_size:
            self.memory_index[candidate_item["smiles"]] = candidate_item
            self._rebuild_memory_list()
            return True

        candidate_fp = candidate_item.get("fp")
        max_sim = 0.0
        most_similar_item = None
        for mem in self.memory_list:
            sim = tanimoto_sim(candidate_fp, mem.get("fp"))
            if sim > max_sim:
                max_sim = sim
                most_similar_item = mem

        lowest_score_item = self.memory_list[-1]

        if max_sim < novelty_threshold and candidate_item["score"] >= (lowest_score_item["score"] - 1e-12 + min_score_delta):
            sims = []
            for mem in self.memory_list:
                sim_val = tanimoto_sim(candidate_fp, mem.get("fp"))
                sims.append((sim_val, mem))
            sims_pos = [(s, m) for s, m in sims if s > 0.0]
            if sims_pos:
                victim = min(sims_pos, key=lambda x: (x[1]["score"], -x[0]))[1]
            else:
                victim = lowest_score_item

            victim_smi = victim.get("smiles")
            if victim_smi in self.memory_index:
                try:
                    del self.memory_index[victim_smi]
                except KeyError:
                    self._rebuild_memory_list()
                    if len(self.memory_list) == 0:
                        return False
                    victim2 = self.memory_list[-1]
                    if victim2["smiles"] in self.memory_index:
                        del self.memory_index[victim2["smiles"]]
                    else:
                        return False
            else:
                self._rebuild_memory_list()
                if len(self.memory_list) == 0:
                    return False
                victim2 = self.memory_list[-1]
                if victim2["smiles"] in self.memory_index:
                    del self.memory_index[victim2["smiles"]]
                else:
                    return False

            self.memory_index[candidate_item["smiles"]] = candidate_item
            self._rebuild_memory_list()
            return True

        return False

    def prune_memory(self, strategy: str = "keep_top_score", tanimoto_threshold: float = 0.8, keep_recent: Optional[int] = None):
        if strategy == "keep_top_score":
            self.prune_memory_keep_top()
        elif strategy == "age_based":
            self.prune_memory_age(keep_recent=keep_recent)
        elif strategy == "diversity":
            self.prune_memory_diversity(tanimoto_threshold=tanimoto_threshold)
        else:
            self.prune_memory_keep_top()

    # ----------------------------
    # Memory update APIs
    # ----------------------------
    def update_memory(self, candidates: List[Tuple[float, str, str]], epoch_or_time: Optional[float] = None,
                      prune_strategy: str = "keep_top_score", tanimoto_threshold: float = 0.8,
                      novelty_threshold: float = 0.6, min_score_delta: float = 0.0) -> int:
        if not candidates:
            return 0

        self._rebuild_memory_list()

        added_or_updated = 0
        current_min = None
        if self.memory_list:
            current_min = self.memory_list[-1]["score"]

        def admission_threshold():
            if self.absolute_score_threshold is not None:
                return self.absolute_score_threshold
            return current_min

        thr = admission_threshold()

        for score, smi, safe_str in candidates:
            if smi is None or safe_str is None:
                continue

            if self.admission_by_threshold:
                if thr is not None:
                    if float(score) <= thr + float(min_score_delta):
                        continue

            entry = self._make_entry(score, smi, safe_str, epoch_or_time=epoch_or_time)
            existing = self.memory_index.get(smi)

            if existing is None:
                if self.allow_memory_growth:
                    self.memory_index[smi] = entry
                    added_or_updated += 1
                    self._rebuild_memory_list()
                    if self.memory_list:
                        current_min = self.memory_list[-1]["score"]
                        thr = admission_threshold()
                    continue

                if len(self.memory_list) < self.memory_size:
                    self.memory_index[smi] = entry
                    added_or_updated += 1
                    self._rebuild_memory_list()
                    if self.memory_list:
                        current_min = self.memory_list[-1]["score"]
                        thr = admission_threshold()
                else:
                    if prune_strategy == "novelty_replace":
                        replaced = self.try_novelty_replace(entry, novelty_threshold=novelty_threshold, min_score_delta=min_score_delta)
                        if replaced:
                            added_or_updated += 1
                            self._rebuild_memory_list()
                            if self.memory_list:
                                current_min = self.memory_list[-1]["score"]
                                thr = admission_threshold()
                            continue
                        else:
                            self._rebuild_memory_list()
                            lowest_score_item = self.memory_list[-1] if self.memory_list else None
                            if lowest_score_item and entry["score"] > lowest_score_item["score"] + min_score_delta:
                                if lowest_score_item["smiles"] in self.memory_index:
                                    del self.memory_index[lowest_score_item["smiles"]]
                                    self.memory_index[smi] = entry
                                    added_or_updated += 1
                                    self._rebuild_memory_list()
                                    if self.memory_list:
                                        current_min = self.memory_list[-1]["score"]
                                        thr = admission_threshold()
                    else:
                        self._rebuild_memory_list()
                        lowest_score_item = self.memory_list[-1] if self.memory_list else None
                        if lowest_score_item and entry["score"] > lowest_score_item["score"] + min_score_delta:
                            if lowest_score_item["smiles"] in self.memory_index:
                                del self.memory_index[lowest_score_item["smiles"]]
                                self.memory_index[smi] = entry
                                added_or_updated += 1
                                self._rebuild_memory_list()
                                if self.memory_list:
                                    current_min = self.memory_list[-1]["score"]
                                    thr = admission_threshold()
            else:
                if entry["score"] > existing["score"]:
                    self.memory_index[smi] = entry
                    added_or_updated += 1
                    self._rebuild_memory_list()
                    if self.memory_list:
                        current_min = self.memory_list[-1]["score"]
                        thr = admission_threshold()

        if not self.allow_memory_growth:
            if prune_strategy == "diversity":
                self.prune_memory_diversity(tanimoto_threshold=tanimoto_threshold)
            elif prune_strategy == "age_based":
                self.prune_memory_age(keep_recent=None)
            else:
                self.prune_memory_keep_top()

        return added_or_updated

    # ----------------------------
    # Augmentation & fine-tune (kept similar to previous design)
    # ----------------------------
    def _augment_safe_strings(self, smiles_with_scores: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        augmented = []
        for smi, score in smiles_with_scores:
            try:
                mol = dm.to_mol(smi)
                if mol is None:
                    continue
                for _ in range(self.augmentation_rounds):
                    rand_smi = dm.to_smiles(mol, canonical=False, randomize=True)
                    if not rand_smi:
                        continue
                    rand_mol = dm.to_mol(rand_smi)
                    if rand_mol is None:
                        continue
                    try:
                        safe_str = sf.encode(rand_mol, canonical=False)
                    except Exception:
                        try:
                            safe_str = sf.encode(rand_mol, canonical=True)
                        except Exception:
                            safe_str = None
                    if safe_str:
                        augmented.append((safe_str, float(score)))
            except Exception:
                continue
        return augmented

    def _build_train_entries_and_weights(self):
        self._rebuild_memory_list()
        if not self.memory_list:
            return []
        base_entries = [(item["safe"], float(item["score"])) for item in self.memory_list]
        if self.augmentation_rounds > 0:
            smiles_scores = [(item["smiles"], item["score"]) for item in self.memory_list]
            base_entries.extend(self._augment_safe_strings(smiles_scores))

        if len(base_entries) == 0:
            return []
        scores = np.array([s for _, s in base_entries], dtype=float)
        baseline = np.mean(scores)
        centered = scores - baseline
        scaled = centered / max(self.weight_temp, 1e-6)
        exps = np.exp(scaled - np.max(scaled))
        weights = exps / (exps.sum() + 1e-12)
        mean_w = np.mean(weights)
        if not np.isfinite(mean_w) or mean_w <= 0:
            weights = np.ones_like(weights)
            mean_w = 1.0
        weights = weights / mean_w
        weights = np.clip(weights, 1e-6, 1e6)
        entries_with_weights = [(base_entries[i][0], float(weights[i])) for i in range(len(base_entries))]
        return entries_with_weights

    def fine_tune_on_memory(self, epochs: int = 5, train_batch_size: int = 32):
        entries_with_weights = self._build_train_entries_and_weights()
        if not entries_with_weights:
            return
        dataset = MemoryDataset(entries_with_weights)
        if self.prioritized_replay:
            sample_weights = [w for _, w in entries_with_weights]
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
            loader = DataLoader(dataset, batch_size=train_batch_size, sampler=sampler, collate_fn=collate_strings_and_weights, drop_last=len(dataset) > train_batch_size)
        else:
            loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, collate_fn=collate_strings_and_weights, drop_last=len(dataset) > train_batch_size)

        self.model.train()
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        ce_loss = CrossEntropyLoss(reduction="none")

        for ep in range(epochs):
            epoch_loss = 0.0
            epoch_kl = 0.0
            cnt = 0
            for batch_strings, batch_weights in loader:
                try:
                    inputs = self.hf_tokenizer(batch_strings, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length, add_special_tokens=True)
                    input_ids = inputs["input_ids"].to(self.device)
                    labels = input_ids.clone().to(self.device)
                    pad_id = self.hf_tokenizer.pad_token_id
                    if pad_id is not None:
                        labels[labels == pad_id] = -100

                    outputs = self.model(input_ids=input_ids, labels=labels)
                    logits = outputs.logits
                    vocab_size = logits.size(-1)
                    logits_flat = logits.view(-1, vocab_size)
                    labels_flat = labels.view(-1)
                    losses_flat = ce_loss(logits_flat, labels_flat)
                    losses = losses_flat.view(logits.size(0), logits.size(1))
                    mask = (labels != -100).float()
                    token_counts = mask.sum(dim=1).clamp(min=1.0)
                    per_example_loss = (losses * mask).sum(dim=1) / token_counts
                    batch_weights = batch_weights.to(self.device)
                    batch_weights = torch.clamp(batch_weights, min=1e-6, max=1e6)
                    weighted_loss = (per_example_loss * batch_weights).mean()

                    kl_term = 0.0
                    if self.kl_weight and self.kl_weight > 0:
                        with torch.no_grad():
                            orig_logits = self.original_model(input_ids=input_ids).logits
                        cur_log_prob = F.log_softmax(logits, dim=-1)
                        orig_prob = F.softmax(orig_logits, dim=-1)
                        kl_per_token = (orig_prob * (torch.log(orig_prob + 1e-12) - cur_log_prob)).sum(dim=-1)
                        kl_per_seq = (kl_per_token * mask).sum(dim=1) / token_counts
                        kl_term = kl_per_seq.mean()

                    entropy_term = 0.0
                    if self.entropy_weight and self.entropy_weight > 0:
                        probs = F.softmax(logits, dim=-1)
                        log_probs = F.log_softmax(logits, dim=-1)
                        entropy = -torch.sum(probs * log_probs, dim=-1)
                        entropy = (entropy * mask).sum(dim=1) / token_counts
                        entropy_term = entropy.mean()

                    loss = weighted_loss + self.kl_weight * kl_term - self.entropy_weight * entropy_term
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += weighted_loss.item()
                    epoch_kl += float(kl_term) if isinstance(kl_term, torch.Tensor) else 0.0
                    cnt += 1
                except Exception as e:
                    print(f"[FT] batch training error: {e}")
                    continue
            if cnt > 0:
                avg_loss = epoch_loss / cnt
                avg_kl = epoch_kl / cnt
                print(f"[FT] epoch={ep+1}/{epochs} avg_weighted_loss={avg_loss:.5f} avg_kl={avg_kl:.5f}")

        self.model.eval()

    # ----------------------------
    # Additional utilities
    # ----------------------------
    def sample_unique(self, sequences: List[str]) -> List[str]:
        seen = set()
        unique = []
        for s in sequences:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        return unique

    def get_prior_likelihoods(self, safe_strings: List[str]) -> List[float]:
        """
        Compute mean negative log-likelihood per token for each safe_string using the frozen original_model.
        Returns list of floats (mean NLL per sequence). Uses tokenizer wrapper.
        """
        if not safe_strings:
            return []
        try:
            with torch.no_grad():
                inputs = self.hf_tokenizer(safe_strings, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length, add_special_tokens=True)
                input_ids = inputs["input_ids"].to(self.device)
                labels = input_ids.clone().to(self.device)
                pad_id = self.hf_tokenizer.pad_token_id
                if pad_id is not None:
                    labels[labels == pad_id] = -100
                outputs = self.original_model(input_ids=input_ids).logits
                vocab_size = outputs.size(-1)
                logits_flat = outputs.view(-1, vocab_size)
                labels_flat = labels.view(-1)
                ce = CrossEntropyLoss(reduction="none")
                losses_flat = ce(logits_flat, labels_flat)  # per-token
                losses = losses_flat.view(outputs.size(0), outputs.size(1))
                mask = (labels != -100).float()
                token_counts = mask.sum(dim=1).clamp(min=1.0)
                per_seq_nll = (losses * mask).sum(dim=1) / token_counts
                return [float(x) for x in per_seq_nll.cpu().numpy()]
        except Exception as e:
            print(f"[get_prior_likelihoods] error: {e}")
            return [0.0 for _ in safe_strings]

    def augmented_memory_replay(self) -> Tuple[List[str], List[float], List[float]]:
        """
        Randomize (augment) the SMILES in memory and return randomized_smiles_list,
        corresponding scores, and prior_likelihoods (NLL) computed on randomized safe strings.
        """
        self._rebuild_memory_list()
        if not self.memory_list:
            return [], [], []
        smiles = [it["smiles"] for it in self.memory_list]
        scores = [it["score"] for it in self.memory_list]
        randomized = []
        for s in smiles:
            try:
                mol = dm.to_mol(s)
                if mol is None:
                    randomized.append(s)
                    continue
                rand = dm.to_smiles(mol, canonical=False, randomize=True)
                randomized.append(rand if rand else s)
            except Exception:
                randomized.append(s)
        # compute safe strings for randomized smiles
        randomized_safe = []
        for s in randomized:
            try:
                mol = dm.to_mol(s)
                if mol is None:
                    randomized_safe.append(s)
                    continue
                try:
                    safe_str = sf.encode(mol, canonical=False)
                except Exception:
                    safe_str = sf.encode(mol, canonical=True)
                randomized_safe.append(safe_str if safe_str else s)
            except Exception:
                randomized_safe.append(s)
        prior_likes = self.get_prior_likelihoods(randomized_safe)
        return randomized, scores, prior_likes

    def batch_tanimoto_search(self, candidate_fp):
        """
        Return (max_sim, best_memory_item) for candidate_fp vs memory. Use RDKit BulkTanimotoSimilarity if available.
        """
        self._rebuild_memory_list()
        fps = [it.get("fp") for it in self.memory_list]
        if not fps or candidate_fp is None:
            return 0.0, None
        try:
            # BulkTanimotoSimilarity exists in DataStructs
            sims = DataStructs.BulkTanimotoSimilarity(candidate_fp, fps)
            max_idx = int(np.argmax(sims))
            return float(sims[max_idx]), self.memory_list[max_idx]
        except Exception:
            # fallback to loop
            max_sim = 0.0
            best = None
            for mem in self.memory_list:
                sim = tanimoto_sim(candidate_fp, mem.get("fp"))
                if sim > max_sim:
                    max_sim = sim
                    best = mem
            return max_sim, best

    def export_memory(self, out_path: str):
        """
        Export memory to CSV. out_path can be a directory or a filename.
        """
        self._rebuild_memory_list()
        df = pd.DataFrame([(it["score"], it["smiles"], it["safe"], it["added_at"]) for it in self.memory_list],
                          columns=["score", "smiles", "safe", "added_at"])
        if os.path.isdir(out_path):
            out_file = os.path.join(out_path, "memory_export.csv")
        else:
            out_file = out_path
        df.to_csv(out_file, index=False)
        return out_file

    def import_memory(self, in_csv: str):
        """
        Load memory CSV (columns: score,smiles,safe,added_at) and rebuild memory_index/list.
        """
        df = pd.read_csv(in_csv)
        self.memory_index = {}
        for _, row in df.iterrows():
            try:
                score = float(row["score"])
                smi = str(row["smiles"])
                safe_str = str(row["safe"])
                added_at = float(row.get("added_at", time.time()))
                entry = self._make_entry(score, smi, safe_str, epoch_or_time=added_at)
                self.memory_index[smi] = entry
            except Exception:
                continue
        self._rebuild_memory_list()

    # ----------------------------
    # Generation (kept similar)
    # ----------------------------
    def generate_batch(self, mode: str, input_data: Any = None, n_samples: Optional[int] = None, n_trials: Optional[int] = 1):
        is_exploring = random.random() < self.exploration_prob
        current_designer = self.original_designer if is_exploring else self.designer
        temp = 1.2 if is_exploring else 1.0
        gen_kwargs = {"max_length": self.max_length, "do_sample": True, "temperature": temp}
        n_samples = int(n_samples or self.batch_size)
        n_trials = int(n_trials or 1)
        try:
            current_designer.model.eval()
            current_designer.tokenizer
            with torch.no_grad():
                if mode == "random":
                    generated = current_designer.de_novo_generation(sanitize=True, n_samples_per_trial=n_samples, n_trials=n_trials, **gen_kwargs)
                elif mode == "motif":
                    if not input_data: raise ValueError("Input motif required")
                    generated = current_designer.motif_extension(sanitize=True, motif=input_data, n_samples_per_trial=n_samples, n_trials=n_trials, **gen_kwargs)
                elif mode == "linker":
                    if not input_data: raise ValueError("Input fragments required")
                    generated = current_designer.linker_generation(*input_data, sanitize=True, n_samples_per_trial=n_samples, n_trials=n_trials, **gen_kwargs)
                elif mode == "scaffold":
                    if not input_data: raise ValueError("Input scaffold required")
                    generated = current_designer.scaffold_decoration(sanitize=True, scaffold=input_data, n_samples_per_trial=n_samples, n_trials=n_trials, **gen_kwargs)
                elif mode == "substructure":
                    if not input_data: raise ValueError("Input core required")
                    generated = current_designer.substructure_generation(sanitize=True, core=input_data, n_samples_per_trial=n_samples, n_trials=n_trials, **gen_kwargs)
                else:
                    raise ValueError(f"Unknown mode: {mode}")
        except Exception as e:
            print(f"[SafeAugmentedOptimizer] generation error: {e}")
            return [], is_exploring

        results = [s for s in generated if s is not None]
        if self.unique_sampling:
            results = self.sample_unique(results)
        return results, is_exploring

    # ----------------------------
    # Run loop
    # ----------------------------
    def run(
        self,
        mode: str,
        score_fn: Callable[[List[str]], dict],
        save_path: str,
        input_data: Any = None,
        target_score: float = 2.0,
        epochs: int = 50,
        save_freq: int = 5,
        ft_epochs_per_step: int = 5,
        train_batch_size: int = 32,
        gen_n_trials: int = 1,
        gen_n_samples: Optional[int] = None,
        mol_type: Optional[str] = None,
        checker: Optional[Callable[[str], bool]] = None,
        invalid_score_penalty: float = -1.0,
        memory_prune_strategy: str = "keep_top_score",
        tanimoto_threshold: float = 0.8,
        novelty_threshold: float = 0.6,
        min_score_delta: float = 0.0,
    ):
        os.makedirs(save_path, exist_ok=True)
        print(f"[SafeAugmentedOptimizer] Run start: mode={mode}, mem_prune={memory_prune_strategy}, tanimoto={tanimoto_threshold}, novelty={novelty_threshold}")
        self.absolute_score_threshold = 0.7 * target_score if not self.absolute_score_threshold else self.absolute_score_threshold
        history = {"epoch": [], "mean_reward": [], "validity": [], "best_memory_score": [], "memory_size": [], "is_exploring": [], "new_in_memory_pct": [], "perfect_score_pct": [], "prior_mean_nll": []}
        pbar = tqdm(range(1, epochs + 1))
        for epoch in pbar:
            if len(self.memory_list) >= max(4, self.memory_size // 2):
                smiles = [it["smiles"] for it in self.memory_list]
                unique_frac = len(set(smiles)) / len(smiles) if len(smiles) > 0 else 0.0
                if unique_frac < 0.2:
                    print("[SafeAugmentedOptimizer] Mode collapse detected; purging memory.")
                    self.memory_index = {}
                    self.memory_list = []

            raw_batch, is_exploring = self.generate_batch(mode, input_data=input_data, n_samples=gen_n_samples, n_trials=gen_n_trials)
            candidates = []
            valid_smiles = []
            valid_safe = []
            for smi in raw_batch:
                if smi is None: continue
                try:
                    mol = dm.to_mol(smi)
                    if mol is not None:
                        safe_str = sf.encode(mol, canonical=True)
                        valid_smiles.append(smi)
                        valid_safe.append(safe_str)
                except Exception:
                    continue

            total_generated = len(raw_batch)
            validity = (len(valid_smiles) / total_generated) if total_generated > 0 else 0.0
            mean_reward = 0.0
            perfect_count = 0
            prior_mean_nll = 0.0
            if valid_smiles:
                try:
                    scores_dict = score_fn(valid_smiles)
                    agg_rewards = [0.0] * len(valid_smiles)
                    for _, scores in scores_dict.items():
                        if len(scores) == len(valid_smiles):
                            for i in range(len(valid_smiles)):
                                agg_rewards[i] += float(scores[i])
                    # apply checker penalty if provided
                    for smi, safe_str, raw_score in zip(valid_smiles, valid_safe, agg_rewards):
                        score = float(raw_score)
                        if checker is not None:
                            ok = bool(checker(smi))
                            if not ok:
                                score = score * float(invalid_score_penalty)
                        candidates.append((float(score), smi, safe_str))
                        if score >= target_score:
                            perfect_count += 1
                    mean_reward = float(np.mean(agg_rewards))
                except Exception as e:
                    print(f"[SafeAugmentedOptimizer] scoring error: {e}")

                # compute prior-likelihoods (NLL) on safe strings for monitoring
                try:
                    prior_likes = self.get_prior_likelihoods(valid_safe)
                    prior_mean_nll = float(np.mean(prior_likes)) if prior_likes else 0.0
                except Exception as e:
                    prior_mean_nll = 0.0
                    print(f"[SafeAugmentedOptimizer] prior likelihood compute error: {e}")

            perfect_score_pct = (perfect_count / len(valid_smiles)) if valid_smiles else 0.0

            new_count = self.update_memory(candidates, epoch_or_time=epoch,
                                           prune_strategy=memory_prune_strategy,
                                           tanimoto_threshold=tanimoto_threshold,
                                           novelty_threshold=novelty_threshold,
                                           min_score_delta=min_score_delta)

            new_in_memory_pct = (new_count / len(valid_smiles)) if valid_smiles else 0.0
            best_mem_score = self.memory_list[0]["score"] if self.memory_list else 0.0

            if len(self.memory_list) > 0 and epoch % max(1, save_freq) == 0:
                tops = [(round(it["score"],3), it["smiles"]) for it in self.memory_list[:5]]
                print(f"[MEM TOP] {tops}")

            history["epoch"].append(epoch)
            history["mean_reward"].append(mean_reward)
            history["validity"].append(validity)
            history["best_memory_score"].append(best_mem_score)
            history["memory_size"].append(len(self.memory_list))
            history["is_exploring"].append(is_exploring)
            history["new_in_memory_pct"].append(new_in_memory_pct)
            history["perfect_score_pct"].append(perfect_score_pct)
            history["prior_mean_nll"].append(prior_mean_nll)

            if len(self.memory_list) >= 1:
                try:
                    self.fine_tune_on_memory(epochs=ft_epochs_per_step, train_batch_size=train_batch_size)
                    if self.double_loop_augment and self.augmentation_rounds > 0:
                        # optional double-loop: use augmented_memory_replay and fine-tune again
                        randomized, scores, prior_likes = self.augmented_memory_replay()
                        # build entries and quick weights for replay
                        replay_entries = []
                        for s, sc in zip(randomized, scores):
                            try:
                                mol = dm.to_mol(s)
                                if mol is None:
                                    continue
                                try:
                                    safe_s = sf.encode(mol, canonical=False)
                                except Exception:
                                    safe_s = sf.encode(mol, canonical=True)
                                if safe_s:
                                    replay_entries.append((safe_s, float(sc)))
                            except Exception:
                                continue
                        # temporarily set memory_list to replay_entries then fine-tune once
                        if replay_entries:
                            # backup
                            backup_index = self.memory_index.copy()
                            backup_list = self.memory_list.copy()
                            # clear and set
                            self.memory_index = {}
                            for s, sc in replay_entries:
                                ent = self._make_entry(sc, s, s, epoch_or_time=time.time())
                                self.memory_index[s] = ent
                            self._rebuild_memory_list()
                            # fine-tune on augmented replay set
                            try:
                                self.fine_tune_on_memory(epochs=1, train_batch_size=train_batch_size)
                            except Exception as e:
                                print(f"[SafeAugmentedOptimizer] double-loop fine-tune error: {e}")
                            # restore
                            self.memory_index = backup_index
                            self.memory_list = backup_list
                except Exception as e:
                    print(f"[SafeAugmentedOptimizer] fine-tune error (continuing): {e}")

            explore_tag = "[EXP]" if is_exploring else "     "
            pbar.set_description(f"{explore_tag} E:{epoch} | Rwd:{mean_reward:.2f} | Perfect:{perfect_score_pct:.1%} | NewMem:{new_in_memory_pct:.1%} | Best:{best_mem_score:.2f}")

            if epoch % save_freq == 0:
                ckpt_dir = os.path.join(save_path, f"checkpoint_{epoch}")
                os.makedirs(ckpt_dir, exist_ok=True)
                try:
                    self.model.save_pretrained(ckpt_dir)
                    self.tokenizer.save_pretrained(ckpt_dir)
                except Exception as e:
                    print(f"[SafeAugmentedOptimizer] save failed: {e}")
                pd.DataFrame(history).to_csv(os.path.join(save_path, "training_metrics.csv"), index=False)
                if self.memory_list:
                    df_mem = pd.DataFrame([(it["score"], it["smiles"], it["safe"], it["added_at"]) for it in self.memory_list],
                                           columns=["score","smiles","safe","added_at"])
                    df_mem.to_csv(os.path.join(save_path, f"memory_{epoch}.csv"), index=False)

        final_dir = os.path.join(save_path, "final_model")
        os.makedirs(final_dir, exist_ok=True)
        try:
            self.model.save_pretrained(final_dir)
            self.tokenizer.save_pretrained(final_dir)
        except Exception as e:
            print(f"[SafeAugmentedOptimizer] final save failed: {e}")
        pd.DataFrame(history).to_csv(os.path.join(save_path, "final_metrics.csv"), index=False)
        if self.memory_list:
            df_mem = pd.DataFrame([(it["score"], it["smiles"], it["safe"], it["added_at"]) for it in self.memory_list],
                                   columns=["score","smiles","safe","added_at"])
            df_mem.to_csv(os.path.join(save_path, "final_memory.csv"), index=False)
        print(f"[SafeAugmentedOptimizer] Done. Results saved to {save_path}")
