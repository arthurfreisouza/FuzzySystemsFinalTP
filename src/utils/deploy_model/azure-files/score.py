# score.py — Azure ML Online Endpoint Scoring Script
# Requer: torch, scikit-learn, joblib, numpy
from __future__ import annotations
import os
import json
import numpy as np
import torch
import torch.nn as nn
import joblib
from pathlib import Path


class ScatterANFIS(nn.Module):
    """ANFIS Sugeno 1st-order, scatter partition — idêntica ao notebook."""

    def __init__(self, centers, sigmas, mf_type: str = "gauss"):
        super().__init__()
        if not isinstance(centers, torch.Tensor):
            centers = torch.tensor(centers, dtype=torch.float32)
        if not isinstance(sigmas, torch.Tensor):
            sigmas = torch.tensor(sigmas, dtype=torch.float32)
        self.R, self.N = centers.shape
        self.mf_type = mf_type
        self.centers = nn.Parameter(centers.clone())
        self.sigmas = nn.Parameter(sigmas.clone())
        if mf_type == "gbell":
            self.b_param = nn.Parameter(torch.ones(self.R, self.N) * 2.0)
        self.W_cons = nn.Parameter(torch.zeros(self.R, self.N + 1))

    def _fuzzify(self, x):
        x_exp = x.unsqueeze(1)
        c_exp = self.centers.unsqueeze(0)
        s_exp = self.sigmas.clamp(min=1e-4).unsqueeze(0)
        if self.mf_type == "gauss":
            mu = torch.exp(-0.5 * ((x_exp - c_exp) / s_exp) ** 2)
        else:
            b_exp = self.b_param.clamp(min=0.5).unsqueeze(0)
            mu = 1.0 / (1.0 + torch.abs((x_exp - c_exp) / s_exp) ** (2.0 * b_exp))
        return mu.prod(dim=2)

    def forward(self, x):
        w = self._fuzzify(x)
        w_norm = w / w.sum(dim=1, keepdim=True).clamp(min=1e-8)
        x_aug = torch.cat([x, torch.ones(x.size(0), 1, device=x.device)], dim=1)
        f = torch.einsum("bi,ri->br", x_aug, self.W_cons)
        return (w_norm * f).sum(dim=1, keepdim=True)


model = None
scaler = None
feat_sel = None
selected_indices = None
config = None


def init():
    global model, scaler, feat_sel, selected_indices, config
    model_dir = Path(os.environ["AZUREML_MODEL_DIR"]) / "model_artifacts"
    with open(model_dir / "model_config.json") as f:
        config = json.load(f)
    scaler           = joblib.load(model_dir / "scaler.pkl")
    feat_sel         = joblib.load(model_dir / "feature_selector.pkl")
    selected_indices = feat_sel["selected_indices"]
    centers = torch.tensor(feat_sel["centers"], dtype=torch.float32)
    sigmas  = torch.tensor(feat_sel["sigmas"],  dtype=torch.float32)
    model = ScatterANFIS(centers, sigmas, mf_type=config["mf_type"])
    model.load_state_dict(
        torch.load(model_dir / "anfis_model.pt", map_location="cpu")
    )
    model.eval()


def run(raw_data: str) -> str:
    # Entrada: {"data": [[f1, f2, ..., f30], ...]}  (30 features originais, não normalizadas)
    data = np.array(json.loads(raw_data)["data"], dtype=np.float32)
    # 1. Normalizar todas as 30 features (scaler foi ajustado em 30 features)
    X_sc  = scaler.transform(data).astype(np.float32)
    # 2. Selecionar o subconjunto de features
    X_sel = X_sc[:, selected_indices]
    X_t   = torch.tensor(X_sel)
    with torch.no_grad():
        logits = model(X_t).numpy().ravel()
    preds  = (logits >= config["threshold"]).astype(int).tolist()
    labels = ["Malignant" if p == 0 else "Benign" for p in preds]
    return json.dumps({"predictions": preds, "labels": labels})
