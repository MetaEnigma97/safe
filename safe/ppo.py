import os
import random
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from typing import List, Callable, Dict, Any
from tqdm.auto import tqdm
import datamol as dm

# Import SAFE components
from safe.trainer.model import SAFEDoubleHeadsModel
from safe.tokenizer import SAFETokenizer
import safe as sf
from transformers import PreTrainedTokenizerFast

class SafePPOTrainer:
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lr: float = 1e-5,
        kl_coef: float = 0.1,
        batch_size: int = 16,
        max_length: int = 128,
    ):
        """
        PPO Trainer for SAFE models.
        
        Args:
            model_path: Path or HuggingFace ID of the SAFE model.
            device: Device to run training on.
            lr: Learning rate.
            kl_coef: Coefficient for KL divergence penalty.
            batch_size: Batch size for training.
            max_length: Maximum generation length.
        """
        self.device = device
        self.kl_coef = kl_coef
        self.batch_size = batch_size
        self.max_length = max_length

        # 1. Load the SAFE Tokenizer Wrapper
        if os.path.isdir(tokenizer_path):
            safe_tokenizer_wrapper = SAFETokenizer.load(tokenizer_path)
        else:
            safe_tokenizer_wrapper = SAFETokenizer.from_pretrained(tokenizer_path)

        # 2. Convert to Hugging Face PreTrainedTokenizerFast to make it callable
        # The 'SAFETokenizer' class stores the actual tokenizer in the `.tokenizer` attribute
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=safe_tokenizer_wrapper.tokenizer
        )
        
        # 3. Explicitly set special tokens on the HF wrapper so padding works correctly
        self.tokenizer.pad_token = safe_tokenizer_wrapper.tokenizer.pad_token
        self.tokenizer.bos_token = safe_tokenizer_wrapper.tokenizer.bos_token
        self.tokenizer.eos_token = safe_tokenizer_wrapper.tokenizer.eos_token
        
        # Load Policy Model (Actor)
        self.model = SAFEDoubleHeadsModel.from_pretrained(model_path).to(self.device)
        self.model.train()
        
        # Load Reference Model (Frozen)
        self.ref_model = SAFEDoubleHeadsModel.from_pretrained(model_path).to(self.device)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.optimizer = AdamW(self.model.parameters(), lr=lr)

    def _get_prompts(self, mode: str, data: List[str] = None, n_prompts: int = 16) -> List[str]:
        """
        Prepare prompts based on the generation mode.
        """
        if mode == "random":
            # Return empty strings; tokenizer will add BOS token automatically
            return [""] * n_prompts
        
        if mode in ["scaffold", "linker", "motif", "substructure"]:
            if not data:
                raise ValueError(f"Data (scaffolds/fragments) must be provided for mode '{mode}'")
            # Sample random prompts from the provided data
            return random.choices(data, k=n_prompts)
        
        raise ValueError(f"Unknown generation mode: {mode}")

    def score_sequences(self, sequences: List[str], score_fns: List[Callable[[str], float]]) -> tuple[torch.Tensor, float]:
        """
        Modified: Now returns (rewards_tensor, validity_rate)
        """
        rewards = []
        valid_count = 0
        
        for seq in sequences:
            smiles = None
            try:
                # 尝试解码
                mol = sf.decode(seq, as_mol=True)
                if mol is not None:
                    smiles = dm.to_smiles(mol)
                    valid_count += 1
            except Exception:
                smiles = None

            if smiles is None:
                # 建议：如果是刚开始训练，惩罚不要太大，或者检查模型加载是否正确
                rewards.append(0) 
                continue
            
            total_score = 0.0
            for fn in score_fns:
                try:
                    total_score += fn(dm.to_mol(smiles))
                except:
                    total_score += 0.0
            rewards.append(total_score)
            
        validity_rate = valid_count / len(sequences) if sequences else 0.0
        return torch.tensor(rewards, device=self.device), validity_rate


    def train_step(self, prompts: List[str], score_fns: List[Callable]):
        """
        Perform a single PPO training step.
        """
        # 1. Tokenize Prompts
        cleaned_prompts = [str(p) for p in prompts]
        inputs = self.tokenizer(cleaned_prompts, return_tensors="pt", padding=True).to(self.device)
        
        # 2. Generation (Rollout)
        with torch.no_grad():
            gen_output = self.model.generate(
                **inputs,
                max_length=self.max_length,
                do_sample=True, # 确保开启采样
                top_k=10,       # 可以尝试减小 top_k (如 10) 让生成更保守/准确
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True 
            )
        generated_ids = gen_output.sequences
        generated_strs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        # 3. Compute Rewards & Validity
        rewards, validity = self.score_sequences(generated_strs, score_fns)
        
        # 4. Compute Loss
        outputs = self.model(input_ids=generated_ids, labels=generated_ids)
        logits = outputs.logits
        with torch.no_grad():
            ref_outputs = self.ref_model(input_ids=generated_ids)
            ref_logits = ref_outputs.logits
            
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        ref_log_probs = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
        
        target_ids = generated_ids[:, 1:]
        token_log_probs = torch.gather(log_probs, 2, target_ids.unsqueeze(-1)).squeeze(-1)
        ref_token_log_probs = torch.gather(ref_log_probs, 2, target_ids.unsqueeze(-1)).squeeze(-1)
        
        mask = (target_ids != self.tokenizer.pad_token_id)
        kl_div = token_log_probs - ref_token_log_probs
        rewards_expanded = rewards.view(-1, 1).expand_as(token_log_probs)
        
        # 4. Advantage 计算
        advantage = rewards_expanded - (self.kl_coef * kl_div)

        # === 关键修改：Advantage Normalization ===
        # 标准化 Advantage 可以防止负值过大导致梯度爆炸，也能让正负样本更平衡
        if mask.sum() > 1:
            adv_mean = (advantage * mask).sum() / mask.sum()
            adv_std = ((advantage - adv_mean) ** 2 * mask).sum() / mask.sum()
            advantage = (advantage - adv_mean) / (torch.sqrt(adv_std) + 1e-8)
        
        # Loss Calculation
        loss = - (advantage * token_log_probs * mask).sum() / mask.sum()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # 返回 validity 以便监控
        return loss.item(), rewards.mean().item(), validity

    def train(
        self,
        mode: str,
        score_fns: List[Callable],
        save_path: str,
        epochs: int = 10,
        save_freq: int = 5,
        train_data: List[str] = None,
        steps_per_epoch: int = 50
    ):
        """
        Main training loop.
        """
        os.makedirs(save_path, exist_ok=True)
        
        print(f"Starting PPO Training in '{mode}' mode...")
        print(f"Targeting {len(score_fns)} score functions.")
        
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            epoch_reward = 0.0
            
            pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}/{epochs}")
            for _ in pbar:
                prompts = self._get_prompts(mode, train_data, self.batch_size)
                loss, avg_reward, validity = self.train_step(prompts, score_fns)
                epoch_loss += loss
                epoch_reward += avg_reward
                pbar.set_postfix({"loss": f"{loss:.2f}", "reward": f"{avg_reward:.2f}", "validity":f"{validity:.2f}"})
            
            avg_epoch_loss = epoch_loss / steps_per_epoch
            avg_epoch_reward = epoch_reward / steps_per_epoch
            
            print(f"Epoch {epoch} summary: Loss={avg_epoch_loss:.4f}, Avg Reward={avg_epoch_reward:.4f}")
            
            if epoch % save_freq == 0:
                ckpt_path = os.path.join(save_path, f"checkpoint-{epoch}")
                self.model.save_pretrained(ckpt_path)
                self.tokenizer.save_pretrained(ckpt_path)
                print(f"Saved model to {ckpt_path}")

        final_path = os.path.join(save_path, "final_model")
        self.model.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)
        print(f"Training complete. Model saved to {final_path}")
