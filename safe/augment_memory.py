# safe/augment_memory.py
# Augmented memory optimizer with memory pruning strategies (diversity/age/novelty)
# Depends: rdkit, datamol, safe, transformers, torch, numpy, pandas

import os
import random
import time
import math
from typing import List, Tuple, Callable, Optional, Any, Dict
from collections import defaultdict

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
    pass

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
        # If not None, use this absolute threshold instead of current memory min
        self.absolute_score_threshold = None if absolute_score_threshold is None else float(absolute_score_threshold)
        # If True, memory can grow beyond memory_size (no capacity-based pruning). If False, capacity is enforced.
        self.allow_memory_growth = bool(allow_memory_growth)

        print(f"[SafeAugmentedOptimizer] Loading SAFE designer from {model_path} ...")
        self.designer = sf.SAFEDesign.load_default(verbose=False, model_dir=model_path)
        self.model = self.designer.model.to(self.device)
        self.tokenizer = self.designer.tokenizer

        # frozen original for KL regularization & exploration
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
        # Sorted list by score for easy access (list of dicts)
        self.memory_list: List[Dict[str, Any]] = []

        print(f"[SafeAugmentedOptimizer] Initialized on {self.device} | mem_size={self.memory_size} | augment_rounds={self.augmentation_rounds}")

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
        self.memory_list = sorted(self.memory_index.values(), key=lambda x: x["score"], reverse=True)
        if len(self.memory_list) > self.memory_size:
            survivors = self.memory_list[: self.memory_size]
            self.memory_index = {item["smiles"]: item for item in survivors}
            self.memory_list = survivors

    def prune_memory_diversity(self, tanimoto_threshold: float = 0.8):
        """
        Greedy diversity pruning: keep highest-score representative first,
        skip items that are too similar (>= tanimoto_threshold) to already kept representatives.
        """
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
        # rebuild index/list
        self.memory_list = kept[: self.memory_size]
        self.memory_index = {item["smiles"]: item for item in self.memory_list}

    def prune_memory_age(self, keep_recent: int = None):
        """
        Age-based pruning: keep most recent `keep_recent` items (by added_at),
        or keep most recent memory_size if keep_recent is None.
        """
        kr = keep_recent or self.memory_size
        items = sorted(self.memory_list, key=lambda x: x["added_at"], reverse=True)
        kept = items[:kr]
        self.memory_list = kept
        self.memory_index = {item["smiles"]: item for item in kept}

    def prune_memory_keep_top(self):
        """
        Keep top-scoring items (default simple behavior)
        """
        self._rebuild_memory_list()
        # _rebuild_memory_list already truncated to memory_size

    def try_novelty_replace(self, candidate_item: Dict[str, Any], novelty_threshold: float = 0.6, min_score_delta: float = 0.0) -> bool:
        """
        If memory is full and candidate is novel enough (max sim < novelty_threshold)
        and candidate.score is not much lower than memory minimum, replace a similar low-score member.

        Return True if replacement happened (candidate added to memory), False otherwise.

        Fixes applied:
        - ensure memory_list is rebuilt from memory_index at entry to avoid stale snapshot issues
        - check existence of victim key in memory_index before deletion; if missing, rebuild and pick a fresh victim
        - handle graceful fallback (no KeyError)
        """
        # Ensure we operate on up-to-date memory_list
        self._rebuild_memory_list()

        if len(self.memory_list) < self.memory_size:
            # not full -> simply add
            self.memory_index[candidate_item["smiles"]] = candidate_item
            self._rebuild_memory_list()
            return True

        # compute max similarity of candidate to memory
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
            # pick victim: prefer the most similar low-score item if any, else the lowest_score_item
            victim = None
            # find candidates with sim > 0.0 and pick one with lowest score
            sims = []
            for mem in self.memory_list:
                sim_val = tanimoto_sim(candidate_fp, mem.get("fp"))
                sims.append((sim_val, mem))
            # filter sims > 0 and sort by score ascending
            sims_pos = [(s, m) for s, m in sims if s > 0.0]
            if sims_pos:
                # pick the mem among sims_pos with smallest score
                victim = min(sims_pos, key=lambda x: (x[1]["score"], -x[0]))[1]
            else:
                victim = lowest_score_item

            # attempt safe deletion: ensure victim key is present
            victim_smi = victim.get("smiles")
            if victim_smi in self.memory_index:
                try:
                    del self.memory_index[victim_smi]
                except KeyError:
                    # race condition: someone else removed it; rebuild and fallback
                    self._rebuild_memory_list()
                    if len(self.memory_list) == 0:
                        return False
                    victim2 = self.memory_list[-1]
                    if victim2["smiles"] in self.memory_index:
                        del self.memory_index[victim2["smiles"]]
                    else:
                        return False
            else:
                # victim not in index (stale). rebuild and remove current lowest if possible
                self._rebuild_memory_list()
                if len(self.memory_list) == 0:
                    return False
                victim2 = self.memory_list[-1]
                if victim2["smiles"] in self.memory_index:
                    del self.memory_index[victim2["smiles"]]
                else:
                    return False

            # now insert candidate
            self.memory_index[candidate_item["smiles"]] = candidate_item
            # rebuild list to keep invariants
            self._rebuild_memory_list()
            return True

        return False

    def prune_memory(self, strategy: str = "keep_top_score", tanimoto_threshold: float = 0.8, keep_recent: Optional[int] = None):
        """
        Public pruning function: pick strategy in {"keep_top_score","age_based","diversity"}
        """
        if strategy == "keep_top_score":
            self.prune_memory_keep_top()
        elif strategy == "age_based":
            self.prune_memory_age(keep_recent=keep_recent)
        elif strategy == "diversity":
            self.prune_memory_diversity(tanimoto_threshold=tanimoto_threshold)
        else:
            # fallback to top score
            self.prune_memory_keep_top()

    # ----------------------------
    # Memory update APIs
    # ----------------------------
    def update_memory(self, candidates: List[Tuple[float, str, str]], epoch_or_time: Optional[float] = None,
                      prune_strategy: str = "keep_top_score", tanimoto_threshold: float = 0.8,
                      novelty_threshold: float = 0.6, min_score_delta: float = 0.0) -> int:
        """
        Add/merge candidate list into memory.
        candidates: list of (score, smiles, safe_str)
        prune_strategy: used for post-merge pruning
        novelty_threshold/min_score_delta: used if prune_strategy == 'novelty_replace' when trying instant replace
        Returns number of actually added / replaced items.

        Fixes applied:
        - rebuild memory_list at start to ensure consistent snapshot
        - rely on try_novelty_replace (which itself rebuilds and deletes safely)
        """
        if not candidates:
            return 0

        # Ensure memory_list reflects memory_index before processing (fix for KeyError)
        self._rebuild_memory_list()

        added_or_updated = 0

        current_min = None
        if self.memory_list:
            current_min = self.memory_list[-1]["score"]

        # compute admission threshold function
        def admission_threshold():
            if self.absolute_score_threshold is not None:
                return self.absolute_score_threshold
            return current_min  # may be None

        thr = admission_threshold()

        for score, smi, safe_str in candidates:
            if smi is None or safe_str is None:
                continue

            # If threshold admission is enabled, check it first
            if self.admission_by_threshold:
                # If there's no threshold (memory empty and no absolute threshold), allow
                if thr is not None:
                    # candidate must strictly exceed threshold + min_score_delta
                    if float(score) <= thr + float(min_score_delta):
                        # reject candidate by threshold
                        continue
                # else thr is None -> memory empty and no absolute threshold -> accept

            entry = self._make_entry(score, smi, safe_str, epoch_or_time=epoch_or_time)
            existing = self.memory_index.get(smi)

            # If allow_memory_growth, we will add any candidate that passed admission (no capacity check)
            if existing is None:
                if self.allow_memory_growth:
                    # Just add/append
                    self.memory_index[smi] = entry
                    added_or_updated += 1
                    # update list snapshot
                    self._rebuild_memory_list()
                    # recompute current_min and threshold for subsequent candidates
                    if self.memory_list:
                        current_min = self.memory_list[-1]["score"]
                        thr = admission_threshold()
                    continue

                # otherwise, same capacity-aware logic as before
                if len(self.memory_list) < self.memory_size:
                    # memory not full, add
                    self.memory_index[smi] = entry
                    added_or_updated += 1
                    self._rebuild_memory_list()
                    # update current_min / thr
                    if self.memory_list:
                        current_min = self.memory_list[-1]["score"]
                        thr = admission_threshold()
                else:
                    # memory full: try novelty_replace if configured
                    if prune_strategy == "novelty_replace":
                        replaced = self.try_novelty_replace(entry, novelty_threshold=novelty_threshold, min_score_delta=min_score_delta)
                        if replaced:
                            added_or_updated += 1
                            # update current_min / thr
                            self._rebuild_memory_list()
                            if self.memory_list:
                                current_min = self.memory_list[-1]["score"]
                                thr = admission_threshold()
                            continue
                        else:
                            # fallback: if candidate.score > lowest + min_score_delta, replace lowest
                            self._rebuild_memory_list()
                            lowest_score_item = self.memory_list[-1] if self.memory_list else None
                            if lowest_score_item and entry["score"] > lowest_score_item["score"] + min_score_delta:
                                # safe deletion
                                if lowest_score_item["smiles"] in self.memory_index:
                                    del self.memory_index[lowest_score_item["smiles"]]
                                    self.memory_index[smi] = entry
                                    added_or_updated += 1
                                    self._rebuild_memory_list()
                                    # update thr
                                    if self.memory_list:
                                        current_min = self.memory_list[-1]["score"]
                                        thr = admission_threshold()
                            # else skip
                    else:
                        # default: replace lowest if candidate is better by min_score_delta
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
                                # victim missing; skip
                                pass
            else:
                # existing in memory: update only if score better
                if entry["score"] > existing["score"]:
                    self.memory_index[smi] = entry
                    added_or_updated += 1
                    self._rebuild_memory_list()
                    if self.memory_list:
                        current_min = self.memory_list[-1]["score"]
                        thr = admission_threshold()

        # After merging candidates, apply global pruning only if we do NOT allow unlimited growth
        if not self.allow_memory_growth:
            if prune_strategy == "diversity":
                self.prune_memory_diversity(tanimoto_threshold=tanimoto_threshold)
            elif prune_strategy == "age_based":
                self.prune_memory_age(keep_recent=None)
            else:
                self.prune_memory_keep_top()
        else:
            # If allow_memory_growth True but user still wants to cap later, they can call prune_memory manually.
            pass

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
                        safe_str = sf.encode(rand_mol, canonical=True)
                    if safe_str:
                        augmented.append((safe_str, float(score)))
            except Exception:
                continue
        return augmented

    def _build_train_entries_and_weights(self):
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
        weights = weights / np.mean(weights)
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
            if cnt > 0:
                avg_loss = epoch_loss / cnt
                avg_kl = epoch_kl / cnt
                print(f"[FT] epoch={ep+1}/{epochs} avg_weighted_loss={avg_loss:.5f} avg_kl={avg_kl:.5f}")

        self.model.eval()

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
        return [s for s in generated if s is not None], is_exploring

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
        history = defaultdict(list)
        pbar = tqdm(range(1, epochs + 1))
        for epoch in pbar:
            # collapse guard
            if len(self.memory_list) >= max(4, self.memory_size // 2):
                smiles = [it["smiles"] for it in self.memory_list]
                if len(set(smiles)) == 10 and len(scores) > 10:
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

            perfect_score_pct = (perfect_count / len(valid_smiles)) if valid_smiles else 0.0

            # update memory with chosen prune strategy & replacement policy
            new_count = self.update_memory(candidates, epoch_or_time=epoch,
                                           prune_strategy=memory_prune_strategy,
                                           tanimoto_threshold=tanimoto_threshold,
                                           novelty_threshold=novelty_threshold,
                                           min_score_delta=min_score_delta)

            new_in_memory_pct = (new_count / len(valid_smiles)) if valid_smiles else 0.0
            best_mem_score = self.memory_list[0]["score"] if self.memory_list else 0.0

            # debug top
            if len(self.memory_list) > 0 and epoch % max(1, save_freq) == 0:
                tops = [(round(it["score"],3), it["smiles"]) for it in self.memory_list[-5:]]
                print(f"[MEM TOP] {tops}")

            # logging
            history["epoch"].append(epoch)
            history["mean_reward"].append(mean_reward)
            history["validity"].append(validity)
            history["best_memory_score"].append(best_mem_score)
            history["memory_size"].append(len(self.memory_list))
            history["is_exploring"].append(is_exploring)
            history["new_in_memory_pct"].append(new_in_memory_pct)
            history["perfect_score_pct"].append(perfect_score_pct)

            # fine-tune
            if len(self.memory_list) >= 1:
                self.fine_tune_on_memory(epochs=ft_epochs_per_step, train_batch_size=train_batch_size)
 
            explore_tag = "[EXP]" if is_exploring else "     "
            pbar.set_description(f"{explore_tag} E:{epoch} | Rwd:{mean_reward:.2f} | Perfect:{perfect_score_pct:.1%} | NewMem:{new_in_memory_pct:.1%} | Best:{best_mem_score:.2f}")

            # periodic save
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

        # final save
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
