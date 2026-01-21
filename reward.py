from typing import List, Dict, Tuple, Optional, Literal
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs
import sys
sys.path.append('/AI4S/Users/jwli')
from mol_prediction.predictors import PropertyPredictor


class RewardCalculator:
    def __init__(
        self,
        rewards: Dict[str, Tuple[float, float]],
        substructure: Optional[List[str]] = None,
        match_mode: Literal["any", "all"] = "any",
        fp_radius: int = 2,
        fp_nbits: int = 2048,
    ):
        """
        初始化 RewardCalculator

        - rewards: 每个属性的阈值范围 (low, high)，例如 {"mw": (0, 300), "lumo": (-2, 100)}
        - substructure: SMARTS 列表，例如 ['C(F)(F)F', '[*]O[*]']
        - match_mode: 'any' 表示匹配任意一个即可，'all' 表示必须全部匹配
        - fp_radius, fp_nbits: 指纹参数，用于相似性计算（Morgan 指纹）
        """
        self.rewards = rewards
        self.predictor = PropertyPredictor()

        # 结构匹配配置
        self.match_mode = match_mode
        self.fp_radius = fp_radius
        self.fp_nbits = fp_nbits

        self.substructure_smarts = substructure or []
        self.patts = [Chem.MolFromSmarts(s) for s in self.substructure_smarts if s]
        # 过滤无效 SMARTS
        self.patts = [p for p in self.patts if p is not None]

    # ---------------------------
    # 通用阈值评分：范围内 1，范围外线性衰减
    # ---------------------------
    def _score_with_threshold(self, value: float, low: float, high: float) -> float:
        if low <= value <= high:
            return 1.0
        # 线性衰减：距离越远分数越低
        if value < low:
            denom = abs(low) if low != 0 else 1.0
            return max(0.0, 0.5 - (low - value) / denom)
        else:  # value > high
            denom = abs(high) if high != 0 else 1.0
            return max(0.0, 0.5 - (value - high) / denom)

    # ---------------------------
    # 指纹与相似性
    # ---------------------------
    def _morgan_fp(self, mol: Chem.Mol):
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=self.fp_radius, nBits=self.fp_nbits)

    def _tanimoto(self, fp1, fp2) -> float:
        return DataStructs.TanimotoSimilarity(fp1, fp2)

    # ---------------------------
    # 结构匹配 + 相似性补偿
    # ---------------------------
    def reward_structure(self, smiles_list: List[str]) -> List[float]:
        """
        - 满足匹配条件（any/all）返回 1
        - 不满足时，返回与子结构的相似性分数：
          - any: 取与任一子结构的最大相似性
          - all: 取与所有子结构的最小相似性（更严格）
        """
        scores: List[float] = []
        # 预先计算子结构指纹
        patt_fps = []
        for p in self.patts:
            # SMARTS 转 Mol 指纹（注意：SMARTS 不是完整分子，但 RDKit 仍可生成模式指纹）
            try:
                patt_fps.append(self._morgan_fp(p))
            except Exception:
                patt_fps.append(None)

        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or not self.patts:
                scores.append(0)
                continue

            # 结构匹配判断
            matches = [mol.HasSubstructMatch(p) for p in self.patts]
            if self.match_mode == "any":
                if any(matches):
                    scores.append(1.0)
                    continue
            else:  # "all"
                if all(matches):
                    scores.append(1.0)
                    continue

            # 不满足匹配时，计算相似性补偿
            try:
                mol_fp = self._morgan_fp(mol)
                sims = []
                for fp in patt_fps:
                    if fp is None:
                        sims.append(0)
                    else:
                        sims.append(self._tanimoto(mol_fp, fp))
                if not sims:
                    scores.append(0)
                    continue

                if self.match_mode == "any":
                    sim_score = max(sims)
                else:  # "all"
                    sim_score = min(sims)

                sim_score = max(0, float(sim_score))
                scores.append(sim_score)
            except Exception:
                scores.append(0)

        return scores

    def reward_mw(self, smiles_list: List[str]) -> List[float]:
        mw_values = self.predictor.compute_molecular_weights(smiles_list)
        low, high = self.rewards.get("mw", (0.0, 300.0))
        return [self._score_with_threshold(mw, low, high) for mw in mw_values]

    def reward_homo(self, smiles_list: List[str]) -> List[float]:
        homo_values = self.predictor.predict_homo(smiles_list)
        if "homo" in self.rewards:
            low, high = self.rewards["homo"]
            return [self._score_with_threshold(val, low, high) for val in homo_values]
        return homo_values

    def reward_lumo(self, smiles_list: List[str]) -> List[float]:
        lumo_values = self.predictor.predict_lumo(smiles_list)
        low, high = self.rewards.get("lumo", (-2.0, 100.0))
        return [self._score_with_threshold(val, low, high) for val in lumo_values]

    def reward_ox(self, smiles_list: List[str]) -> List[float]:
        ox_values = self.predictor.predict_ox(smiles_list)
        low, high = self.rewards.get("ox", (93.0, 120.0))
        return [self._score_with_threshold(val, low, high) for val in ox_values]

    def reward_ch(self, smiles_list: List[str]) -> List[float]:
        ch_values = self.predictor.predict_ch_be(smiles_list)
        low, high = self.rewards.get("ch", (93.0, 120.0))
        return [self._score_with_threshold(val, low, high) for val in ch_values]

    def calculate(self, smiles_list: List[str]) -> Dict[str, List[float]]:
        """
        返回每个启用 reward 的分数列表（与输入长度一致）
        """
        results: Dict[str, List[float]] = {}
        if "structure" in self.rewards or self.patts:
            results["structure"] = self.reward_structure(smiles_list)
        if "mw" in self.rewards:
            results["mw"] = self.reward_mw(smiles_list)
        if "homo" in self.rewards:
            results["homo"] = self.reward_homo(smiles_list)
        if "lumo" in self.rewards:
            results["lumo"] = self.reward_lumo(smiles_list)
        if "ox" in self.rewards:
            results["ox"] = self.reward_ox(smiles_list)
        if "ch" in self.rewards:
            results["ch"] = self.reward_ch(smiles_list)
        return results
