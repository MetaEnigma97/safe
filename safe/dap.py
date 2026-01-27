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

# SAFE imports (may raise if package layout differs)
try:
    from safe.trainer.model import SAFEDoubleHeadsModel
    from safe.tokenizer import SAFETokenizer
    from safe.sample import SAFEDesign
except ImportError:
    SAFEDesign = None

def safe_encode_or_none(smi: str) -> Optional[str]:
    try:
        mol = dm.to_mol(smi)
        if mol is None:
            return None
        return sf.encode(mol, canonical=True)
    except Exception:
        return None

class Inception:
    def __init__(self, memory_size: int = 100, batch_size: int = 32, admission_threshold: Optional[float] = None):
        self.memory_size = int(memory_size)
        self.batch_size = int(batch_size)
        self.admission_threshold = None if admission_threshold is None else float(admission_threshold)
        self.memory: List[Tuple[str, float, float]] = []

    def add(self, safe_strings: List[str], scores: List[float], prior_log_probs: List[float]):
        if not safe_strings:
            return
        entries = []
        for s, sc, plp in zip(safe_strings, scores, prior_log_probs):
            if s is None:
                continue
            if self.admission_threshold is not None and sc < self.admission_threshold:
                continue
            entries.append((s, float(sc), float(plp)))
        if not entries:
            return
        uniq: Dict[str, Tuple[float, float]] = {item[0]: (item[1], item[2]) for item in self.memory}
        for s, sc, plp in entries:
            prev = uniq.get(s)
            if prev is None or sc > prev[0]:
                uniq[s] = (sc, plp)
        mem_list = [(s, v[0], v[1]) for s, v in uniq.items()]
        mem_list.sort(key=lambda x: x[1], reverse=True)
        self.memory = mem_list[: self.memory_size]

    def sample(self) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        if not self.memory:
            return [], torch.tensor([], dtype=torch.float32), torch.tensor([], dtype=torch.float32)
        k = min(len(self.memory), self.batch_size)
        idxs = np.random.choice(len(self.memory), size=k, replace=False)
        batch = [self.memory[i] for i in idxs]
        safe_strs = [b[0] for b in batch]
        scores = torch.tensor([b[1] for b in batch], dtype=torch.float32)
        prior_lp = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        return safe_strs, scores, prior_lp

class DiversityFilter:
    def __init__(self, bucket_size: int = 500, penalty_factor: float = 0.5):
        self.bucket_size = int(bucket_size)
        self.penalty_factor = float(penalty_factor)
        self.scaffold_counter: Dict[str, int] = defaultdict(int)
        self.seen_smiles: set = set()

    def reset(self):
        self.scaffold_counter.clear()
        self.seen_smiles.clear()

    def calculate_penalty(self, smiles: str) -> Tuple[float, bool]:
        if not smiles:
            return 0.0, False
        if smiles in self.seen_smiles:
            return (self.penalty_factor, False)
        self.seen_smiles.add(smiles)
        scaffold = "generic"
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
                if scaffold_mol:
                    scaffold = Chem.MolToSmiles(scaffold_mol)
        except Exception:
            scaffold = "generic"
        self.scaffold_counter[scaffold] += 1
        if self.scaffold_counter[scaffold] > self.bucket_size:
            return (self.penalty_factor, True)
        return (1.0, True)

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
        inception_admission_threshold: Optional[float] = None,
        reward_standardize: bool = False,
        reward_clip: Optional[float] = None,
        hf_tokenizer_max_length: Optional[int] = None,
        train_micro_batch_size: int = 128,
        gradient_accumulation_steps: int = 1,
        use_fp16: bool = True,
    ):
        self.device = device
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.sigma = float(sigma)
        self.model_path = model_path
        self.reward_standardize = bool(reward_standardize)
        self.reward_clip = None if reward_clip is None else float(reward_clip)
        self.hf_tokenizer_max_length = hf_tokenizer_max_length or self.max_length

        # training memory parameters
        self.train_micro_batch_size = int(train_micro_batch_size)
        self.gradient_accumulation_steps = int(max(1, gradient_accumulation_steps))
        self.use_fp16 = bool(use_fp16)

        print(f"[SafeReinventOptimizer] Loading SAFE Agent from {model_path}...")
        self.agent_designer = sf.SAFEDesign.load_default(verbose=False, model_dir=model_path)
        self.agent_model = self.agent_designer.model.to(self.device)
        self.agent_model.train()

        print(f"[SafeReinventOptimizer] Loading SAFE Prior from {model_path}...")
        self.prior_designer = sf.SAFEDesign.load_default(verbose=False, model_dir=model_path)
        self.prior_model = self.prior_designer.model.to(self.device)
        self.prior_model.eval()
        for param in self.prior_model.parameters():
            param.requires_grad = False

        self.tokenizer = self.agent_designer.tokenizer
        if hasattr(self.tokenizer, "tokenizer"):
            self.hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=self.tokenizer.tokenizer)
        else:
            self.hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=self.tokenizer)
        if self.hf_tokenizer.pad_token is None:
            if getattr(self.hf_tokenizer, "eos_token", None) is not None:
                self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token
            else:
                self.hf_tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.optimizer = AdamW(self.agent_model.parameters(), lr=self.lr, weight_decay=1e-2)
        self.diversity_filter = DiversityFilter(bucket_size=bucket_size, penalty_factor=penalty_factor)
        self.inception = Inception(memory_size=memory_size, batch_size=32, admission_threshold=inception_admission_threshold)

        # fp16 scaler if using mixed precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_fp16 and (self.device.startswith("cuda")))

    def _get_log_probs(self, model, input_ids, attention_mask):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        log_probs = F.log_softmax(shift_logits, dim=-1)
        target_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        shift_mask = attention_mask[..., 1:].contiguous().float()
        seq_sum_log_probs = (target_log_probs * shift_mask).sum(dim=1)
        seq_lengths = shift_mask.sum(dim=1).clamp(min=1.0)
        return seq_sum_log_probs / seq_lengths

    def _prepare_safe_strings(self, smiles_list: List[str]) -> Tuple[List[str], List[int]]:
        safe_strings = []
        valid_indices = []
        for i, smi in enumerate(smiles_list):
            s = safe_encode_or_none(smi)
            if s:
                safe_strings.append(s)
                valid_indices.append(i)
        return safe_strings, valid_indices

    def train_step(self, smiles_list: List[str], rewards: List[float], raw_scores: List[float]):
        if not smiles_list:
            return 0.0, 0.0, 0.0

        safe_strings, valid_indices = self._prepare_safe_strings(smiles_list)
        if not safe_strings:
            return 0.0, 0.0, 0.0

        device = self.device
        all_augmented_targets = []
        all_agent_logps = []
        all_prior_logps = []

        # prepare per-valid rewards and raw_scores
        batch_rewards_full = np.array([float(rewards[i]) for i in valid_indices], dtype=float)
        batch_raw_scores = [float(raw_scores[i]) for i in valid_indices]

        # optional standardize and clip (we will convert per chunk to tensor)
        if self.reward_standardize and batch_rewards_full.size > 1:
            mean_r = batch_rewards_full.mean()
            std_r = batch_rewards_full.std()
            batch_rewards_full = (batch_rewards_full - mean_r) / (std_r + 1e-8)
        if self.reward_clip is not None:
            batch_rewards_full = np.clip(batch_rewards_full, -self.reward_clip, self.reward_clip)

        # We'll process safe_strings in micro-batches and accumulate gradients
        N = len(safe_strings)
        micro = int(self.train_micro_batch_size)
        accum_steps = int(self.gradient_accumulation_steps)
        assert micro > 0 and accum_steps > 0

        self.optimizer.zero_grad()
        total_loss_value = 0.0
        total_agent_logp = 0.0
        total_prior_logp = 0.0
        chunks = 0

        # We'll also collect prior log probs for inception update
        inception_prior_list = []
        inception_safe_list = []
        inception_scores_list = []

        for start in range(0, N, micro):
            end = min(start + micro, N)
            chunk_safe = safe_strings[start:end]
            chunk_rewards = batch_rewards_full[start:end]
            chunk_raw_scores = batch_raw_scores[start:end]

            # tokenize chunk
            inputs = self.hf_tokenizer(chunk_safe, return_tensors="pt", padding=True, truncation=True, max_length=self.hf_tokenizer_max_length, add_special_tokens=True)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            # forward agent (with autocast if FP16 enabled)
            with torch.cuda.amp.autocast(enabled=(self.use_fp16 and device.startswith("cuda"))):
                agent_logps = self._get_log_probs(self.agent_model, input_ids=input_ids, attention_mask=attention_mask)

            # forward prior in no_grad to save memory
            with torch.no_grad():
                prior_logps = self._get_log_probs(self.prior_model, input_ids=input_ids, attention_mask=attention_mask)

            # prepare tensors
            agent_logps = agent_logps.to(device)
            prior_logps = prior_logps.to(device)
            chunk_rewards_t = torch.tensor(chunk_rewards, device=device, dtype=torch.float32)

            # augmented target and clamp
            augmented_target = prior_logps + (self.sigma * chunk_rewards_t)
            augmented_target = torch.clamp(augmented_target, max=0.0)

            # compute loss for chunk
            # use mse; scale by 1/accum_steps for gradient accumulation
            chunk_loss = F.mse_loss(agent_logps, augmented_target, reduction="mean") / accum_steps

            # backward (use scaler if FP16)
            if self.use_fp16 and device.startswith("cuda"):
                self.scaler.scale(chunk_loss).backward()
            else:
                chunk_loss.backward()

            # accumulate stats
            total_loss_value += float(chunk_loss.item()) * accum_steps  # scale back
            total_agent_logp += float(agent_logps.mean().detach().cpu().item())
            total_prior_logp += float(prior_logps.mean().detach().cpu().item())
            chunks += 1

            # collect for inception buffer (use CPU lists)
            inception_safe_list.extend(chunk_safe)
            inception_scores_list.extend([float(s) for s in chunk_raw_scores])
            inception_prior_list.extend([float(p) for p in prior_logps.detach().cpu().numpy().tolist()])

            # cleanup chunk-level tensors
            del inputs, input_ids, attention_mask, agent_logps, prior_logps, chunk_rewards_t, augmented_target, chunk_loss
            torch.cuda.empty_cache()

            # step optimizer every accum_steps micro-batches
            if (chunks % accum_steps) == 0:
                if self.use_fp16 and device.startswith("cuda"):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

        # After all chunks, ensure optimizer step if not already done
        if (chunks % accum_steps) != 0:
            if self.use_fp16 and device.startswith("cuda"):
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()

        # update inception with accumulated CPU lists
        try:
            if inception_safe_list:
                self.inception.add(inception_safe_list, inception_scores_list, inception_prior_list)
        except Exception:
            pass

        avg_loss = total_loss_value / max(1, chunks)
        avg_agent_logp = total_agent_logp / max(1, chunks)
        avg_prior_logp = total_prior_logp / max(1, chunks)
        return float(avg_loss), float(avg_agent_logp), float(avg_prior_logp)

    def generate_batch(self, mode: str, temperature: float = 1.0, input_data: Any = None) -> List[str]:
        micro_batch = 64
        total_samples = self.batch_size
        all_smiles: List[str] = []
        # ensure generation does not track gradients and model is in eval
        self.agent_model.eval()
        try:
            with torch.no_grad():
                while len(all_smiles) < total_samples:
                    current_n = min(micro_batch, total_samples - len(all_smiles))
                    try:
                        if mode == "random":
                            batch_res = self.agent_designer.de_novo_generation(sanitize=True, n_samples_per_trial=current_n, n_trials=1, max_length=self.max_length, do_sample=True, temperature=temperature, top_p=0.95)
                        elif mode == "motif":
                            batch_res = self.agent_designer.motif_extension(sanitize=True, motif=input_data, n_samples_per_trial=current_n, n_trials=1, max_length=self.max_length, do_sample=True, temperature=temperature, top_p=0.95)
                        elif mode == "linker":
                            batch_res = self.agent_designer.linker_generation(*input_data, sanitize=True, n_samples_per_trial=current_n, n_trials=1, max_length=self.max_length, do_sample=True, temperature=temperature, top_p=0.95)
                        elif mode == "scaffold":
                            batch_res = self.agent_designer.scaffold_decoration(sanitize=True, scaffold=input_data, n_samples_per_trial=current_n, n_trials=1, max_length=self.max_length, do_sample=True, temperature=temperature, top_p=0.95)
                        elif mode == "substructure":
                            batch_res = self.agent_designer.substructure_generation(sanitize=True, core=input_data, n_samples_per_trial=current_n, n_trials=1, max_length=self.max_length, do_sample=True, temperature=temperature, top_p=0.95)
                        else:
                            batch_res = []
                        if batch_res:
                            all_smiles.extend([s for s in batch_res if s])
                    except Exception:
                        # ignore generation errors
                        pass
        finally:
            # restore train mode
            self.agent_model.train()
        return all_smiles

    def run(self, mode: str, score_fn: Callable, save_path: str, input_data: Any = None, target_score: float = 2.0, epochs: int = 50, save_freq: int = 5):
        os.makedirs(save_path, exist_ok=True)
        print(f"[SafeReinventOptimizer] Starting optimization | Mode: {mode} | Batch: {self.batch_size} | Sigma: {self.sigma}")
        self.diversity_filter.reset()
        history = defaultdict(list)
        pbar = tqdm(range(1, epochs + 1))
        current_temp = 1.0

        for epoch in pbar:
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
                    if isinstance(scores_dict, dict):
                        for v in scores_dict.values():
                            if len(v) == len(valid_smiles):
                                raw_scores += np.array(v)
                    elif isinstance(scores_dict, (list, tuple, np.ndarray)):
                        raw_scores = np.array(scores_dict)
                    else:
                        raw_scores = np.zeros(len(valid_smiles))
                except Exception as e:
                    print(f"[SafeReinventOptimizer] Scoring error: {e}")
                    raw_scores = np.zeros(len(valid_smiles))

                for i, smi in enumerate(valid_smiles):
                    penalty, is_new = self.diversity_filter.calculate_penalty(smi)
                    if is_new:
                        new_count += 1
                    final_score = float(raw_scores[i]) * float(penalty)
                    final_smiles.append(smi)
                    total_rewards.append(final_score)
                    raw_scores_list.append(float(raw_scores[i]))

            loss, avg_agent_logp, avg_prior_logp = self.train_step(final_smiles, total_rewards, raw_scores_list)

            mean_rwd = float(np.mean(total_rewards)) if total_rewards else 0.0
            mean_raw = float(np.mean(raw_scores_list)) if raw_scores_list else 0.0
            std_rwd = float(np.std(total_rewards)) if total_rewards else 0.0

            repetition_gap = mean_raw - mean_rwd
            if repetition_gap > 0.1:
                current_temp = min(current_temp + 0.1, 1.5)
            elif repetition_gap < 0.05:
                current_temp = max(current_temp - 0.05, 1.0)

            history["epoch"].append(epoch)
            history["mean_score"].append(mean_rwd)
            history["mean_raw_score"].append(mean_raw)
            history["mean_loss"].append(loss)
            history["validity"].append(validity)
            history["temperature"].append(current_temp)
            history["avg_agent_logp"].append(avg_agent_logp)
            history["avg_prior_logp"].append(avg_prior_logp)
            history["inception_size"].append(len(self.inception.memory))
            history["new_count"].append(new_count)
            history["reward_std"].append(std_rwd)

            pbar.set_description(f"Raw:{mean_raw:.3f}|Rwd:{mean_rwd:.3f}|Gap:{repetition_gap:.3f}|T:{current_temp:.2f}|L:{loss:.4f}")

            if epoch % save_freq == 0:
                try:
                    self.agent_model.save_pretrained(os.path.join(save_path, f"checkpoint_{epoch}"))
                    self.tokenizer.save_pretrained(os.path.join(save_path, f"checkpoint_{epoch}"))
                except Exception as e:
                    print(f"[SafeReinventOptimizer] save failed: {e}")
                pd.DataFrame(history).to_csv(os.path.join(save_path, "metrics.csv"), index=False)

        try:
            self.agent_model.save_pretrained(os.path.join(save_path, "final_model"))
            self.tokenizer.save_pretrained(os.path.join(save_path, "final_model"))
        except Exception as e:
            print(f"[SafeReinventOptimizer] final save failed: {e}")
        pd.DataFrame(history).to_csv(os.path.join(save_path, "final_metrics.csv"), index=False)
        print("[SafeReinventOptimizer] Done.")
