import torch
import torch.nn.functional as F
import random
import datamol as dm
from typing import List, Optional, Union, Callable, Dict, Any
from tqdm.auto import tqdm
from loguru import logger
from torch.nn.utils.rnn import pad_sequence

import safe as sf
from safe.sample import SAFEDesign

class SafePPO(SAFEDesign):
    """
    Trainer for Proximal Policy Optimization (PPO) on SAFE models.
    """

    def train(
        self,
        task: str,
        reward_fns: List[Callable[[Any], float]],
        inputs: Optional[List[Any]] = None,
        n_steps: int = 100,
        batch_size: int = 16,
        learning_rate: float = 1e-5,
        ppo_epochs: int = 4,
        clip_epsilon: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Run PPO training loop.

        Args:
            task: Task type ("random", "motif_extension", "scaffold_decoration", "substructure_generation", "linker_generation").
            reward_fns: List of reward functions (callable taking a molecule/SMILES and returning float).
            inputs: List of inputs for the task (e.g., scaffolds). Not required for "random".
            n_steps: Total number of collection steps.
            batch_size: Batch size for collection and update.
            learning_rate: Optimizer learning rate.
            ppo_epochs: Number of optimization epochs per collected batch.
            clip_epsilon: PPO clip parameter.
            gamma: Discount factor.
            gae_lambda: GAE smoothing parameter.
            entropy_coef: Coefficient for entropy loss.
            value_coef: Coefficient for value loss.
            max_grad_norm: Gradient clipping norm.
            device: Device to run training on.
            **kwargs: Arguments passed to the generation method (e.g., max_length, do_sample).
        """
        
        if device is None:
            device = self.model.device
        else:
            self.model.to(device)
            
        # Ensure we have a value head (scalar regression)
        # We reuse the `multiple_choice_head` from SAFEDoubleHeadsModel, ensuring it outputs 1 value.
        if self.model.config.num_labels != 1:
            logger.info("Initializing Value Head (num_labels=1)")
            self.model.config.num_labels = 1
            from safe.trainer.model import PropertyHead
            self.model.multiple_choice_head = PropertyHead(self.model.config).to(device)
            
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        for step in tqdm(range(n_steps), desc="PPO Training"):
            # --- 1. Experience Collection (Rollout) ---
            
            # Sample prompts
            prompts = self._sample_prompts(task, inputs, batch_size)
            
            rollouts = []
            
            # For "random", prompt is None or empty.
            if task == "random":
                 generated_texts = self._generate(n_samples=batch_size, safe_prefix=None, **kwargs)
                 for txt in generated_texts:
                     rollouts.append({"prompt": "", "text": txt})
            else:
                 # Generate for each prompt
                 for prompt in prompts:
                     if prompt is None: continue
                     try:
                         # Generate 1 sample per prompt
                         gen_list = self._generate(n_samples=1, safe_prefix=prompt, **kwargs)
                         if gen_list:
                             rollouts.append({"prompt": prompt, "text": gen_list[0]})
                     except Exception as e:
                         pass
            
            if not rollouts:
                logger.warning("No rollouts collected in this step.")
                continue

            # Compute Rewards
            valid_rollouts = []
            for item in rollouts:
                mol = sf.decode(item["text"], as_mol=True)
                if mol is None:
                    reward = 0.0 
                else:
                    try:
                        reward = sum(fn(mol) for fn in reward_fns)
                    except Exception:
                        reward = 0.0
                
                # Tokenize to get IDs
                prompt_ids = self.tokenizer.encode(item["prompt"]) if item["prompt"] else []
                full_ids = self.tokenizer.encode(item["text"])
                
                # Check if generation actually happened
                if len(full_ids) <= len(prompt_ids):
                    continue
                
                valid_rollouts.append({
                    "prompt_ids": prompt_ids,
                    "full_ids": full_ids,
                    "reward": reward
                })

            if not valid_rollouts:
                continue
                
            # --- 2. PPO Update ---
            
            batch_full_ids = [torch.tensor(x["full_ids"], dtype=torch.long) for x in valid_rollouts]
            batch_rewards = torch.tensor([x["reward"] for x in valid_rollouts], dtype=torch.float, device=device)
            
            padded_ids = pad_sequence(batch_full_ids, batch_first=True, padding_value=self.tokenizer.get_pretrained().pad_token_id).to(device)
            attention_mask = (padded_ids != self.tokenizer.get_pretrained().pad_token_id).long()
            
            # Create action mask: 1 for generated tokens, 0 for prompt tokens
            action_mask = torch.zeros_like(padded_ids)
            for i, item in enumerate(valid_rollouts):
                p_len = len(item["prompt_ids"])
                f_len = len(item["full_ids"])
                if f_len > p_len:
                    action_mask[i, p_len:f_len] = 1
                
            # Get Old Log Probs and Values (no grad)
            with torch.no_grad():
                outputs = self.model(input_ids=padded_ids, attention_mask=attention_mask, output_hidden_states=True)
                logits = outputs.logits # (B, L, V)
                
                # Shift logits right to match tokens (next token prediction)
                # logits[:, :-1] predicts input_ids[:, 1:]
                shift_logits = logits[:, :-1, :]
                shift_labels = padded_ids[:, 1:]
                
                log_probs = F.log_softmax(shift_logits, dim=-1)
                
                # Gather log probs of actions taken
                old_token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
                
            # PPO Epochs
            for _ in range(ppo_epochs):
                 # Forward pass
                 outputs = self.model(input_ids=padded_ids, attention_mask=attention_mask)
                 logits = outputs.logits
                 new_values = outputs.mc_logits.squeeze(-1) # (B)
                 
                 shift_logits = logits[:, :-1, :]
                 new_log_probs = F.log_softmax(shift_logits, dim=-1)
                 new_token_log_probs = new_log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
                 
                 mask = action_mask[:, 1:]
                 
                 # Ratio
                 ratio = torch.exp(new_token_log_probs - old_token_log_probs)
                 
                 # Advantages = Reward - Value
                 # Using REINFORCE-like Advantage with baseline for whole sequence
                 advantages = (batch_rewards - new_values.detach()).unsqueeze(1) # (B, 1)
                 token_advantages = advantages.expand_as(ratio) # (B, L-1)
                 
                 # Policy Loss
                 surr1 = ratio * token_advantages
                 surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * token_advantages
                 policy_loss = -torch.min(surr1, surr2)
                 
                 # Mask policy loss (only calculate for generated tokens)
                 policy_loss = (policy_loss * mask).sum() / (mask.sum() + 1e-8)
                 
                 # Value Loss
                 value_loss = F.mse_loss(new_values, batch_rewards)
                 
                 # Total Loss
                 loss = policy_loss + value_coef * value_loss
                 
                 optimizer.zero_grad()
                 loss.backward()
                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                 optimizer.step()
                 
            # Cleanup
            del batch_full_ids, padded_ids, outputs
            torch.cuda.empty_cache()

    def _sample_prompts(self, task, inputs, batch_size):
        if task == "random":
             return [None] * batch_size
             
        if not inputs:
             return [None] * batch_size
             
        # Sample inputs
        batch_inputs = random.choices(inputs, k=batch_size)
        prompts = []
        
        for inp in batch_inputs:
             try:
                 encoded = self._get_safe_prefix(task, inp)
                 prompts.append(encoded)
             except Exception:
                 prompts.append(None)
        return prompts

    def _get_safe_prefix(self, task, inp):
         if task in ["motif_extension", "scaffold_decoration"]:
              encoded = self.safe_encoder.encoder(inp, canonical=False, randomize=True, allow_empty=True)
              if task == "motif_extension" and encoded.count("(") == encoded.count(")"):
                   encoded = encoded.rstrip(".") + "."
              return encoded
         elif task == "linker_generation":
              if isinstance(inp, (list, tuple)) and len(inp) == 2:
                   side_chains = sf.utils.compute_side_chains(inp[0], inp[1])
                   encoded = self.safe_encoder.encoder(side_chains, canonical=False, randomize=False)
                   return encoded
              elif isinstance(inp, str):
                   return inp
         elif task == "substructure_generation":
              return self.safe_encoder.encoder(inp, canonical=False, randomize=True, allow_empty=True)
              
         return str(inp)
