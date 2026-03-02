from pathlib import Path
import json

import numpy as np
import torch

from src.actformer.token_eval import (
    compute_topk_jaccard_similarity,
    evaluate_token_replacement,
    position_accuracy_curve,
)


def test_compute_topk_jaccard_similarity_simple():
    a = torch.tensor([[[1, 2, 3], [4, 5, 6]]])
    b = torch.tensor([[[2, 3, 7], [4, 8, 9]]])
    out = compute_topk_jaccard_similarity(a, b)
    # pos0: |{2,3}| / |{1,2,3,7}| = 2/4 = 0.5
    # pos1: |{4}| / |{4,5,6,8,9}| = 1/5 = 0.2
    assert out.shape == (1, 2)
    assert torch.allclose(out, torch.tensor([[0.5, 0.2]]), atol=1e-6)


def test_position_accuracy_curve():
    matches = torch.tensor([[True, False, True], [False, True, True]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    correct, total = position_accuracy_curve(matches, mask)
    assert correct.tolist() == [1, 1, 1]
    assert total.tolist() == [2, 2, 1]


class _FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=16):
        encoded = []
        for t in texts:
            vals = [((ord(c) % 20) + 1) for c in t][:max_length]
            if not vals:
                vals = [1]
            encoded.append(vals)
        max_len = max(len(x) for x in encoded)
        ids = []
        masks = []
        for x in encoded:
            pad = [0] * (max_len - len(x))
            ids.append(x + pad)
            masks.append([1] * len(x) + [0] * len(pad))
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }

    def decode(self, token_ids, clean_up_tokenization_spaces=False):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return "|".join(str(int(x)) for x in token_ids)


class _FakeLayer(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states, **kwargs):
        return (self.proj(hidden_states),)


class _FakeInner(torch.nn.Module):
    def __init__(self, hidden_size: int, n_layers: int):
        super().__init__()
        self.layers = torch.nn.ModuleList([_FakeLayer(hidden_size) for _ in range(n_layers)])


class _FakeLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 64, hidden_size: int = 8, n_layers: int = 4):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.model = _FakeInner(hidden_size, n_layers)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        self.config = type("Cfg", (), {"vocab_size": vocab_size})()

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, return_dict=True, use_cache=False):
        h = self.embed(input_ids)
        hidden_states = [h]
        for layer in self.model.layers:
            h = layer(h, attention_mask=attention_mask)[0]
            hidden_states.append(h)
        logits = self.lm_head(h)
        if return_dict:
            return type("Out", (), {"logits": logits, "hidden_states": tuple(hidden_states)})()
        return logits, tuple(hidden_states)


class _FakeActFormer(torch.nn.Module):
    def forward(self, x=None, mask=None, **kwargs):
        # Slight perturbation to ensure differences appear.
        mu = x + 0.01
        return mu, None, None


def test_evaluate_token_replacement_outputs(tmp_path, monkeypatch):
    import src.actformer.token_eval as te

    monkeypatch.setattr(te, "AutoTokenizer", type("AT", (), {"from_pretrained": staticmethod(lambda *a, **k: _FakeTokenizer())}))
    monkeypatch.setattr(te, "AutoModelForCausalLM", type("AM", (), {"from_pretrained": staticmethod(lambda *a, **k: _FakeLM())}))
    monkeypatch.setattr(te, "AutoConfig", type("AC", (), {"from_pretrained": staticmethod(lambda *a, **k: type("C", (), {"hidden_size": 8})())}))
    monkeypatch.setattr(
        te,
        "_load_eval_docs",
        lambda config, split, max_docs: [
            {"doc_id": 1, "label": 0, "split": split, "text": "abcd"},
            {"doc_id": 2, "label": 0, "split": split, "text": "wxyz"},
        ],
    )
    monkeypatch.setattr(te, "_load_actformer", lambda config, hidden_size, device: (_FakeActFormer().to(device), Path("fake.pt")))
    monkeypatch.setattr(
        te,
        "_load_normalization_stats",
        lambda config, layer_index, hidden_size: (np.zeros(hidden_size, dtype=np.float32), np.ones(hidden_size, dtype=np.float32)),
    )

    (tmp_path / "best_layer.json").write_text('{"layer_index": 0}')
    config = {
        "seed": 42,
        "model": {"base_model_name": "fake", "cache_dir": None, "layer_index": 1},
        "layer_search": {"layer_search_output": str(tmp_path / "best_layer.json")},
        "actformer": {"d_model": 8, "n_layers": 2, "n_heads": 2, "ff_mult": 2, "loss_type": "mse", "causal": True},
        "token_eval": {
            "split": "test",
            "max_docs": 2,
            "max_seq_len": 8,
            "batch_size": 2,
            "topk_values": [5],
            "save_examples": 10,
            "output_dir": str(tmp_path / "token_eval_out"),
        },
    }

    out = evaluate_token_replacement(config)
    assert out["num_docs"] == 2
    assert out["replacement_depths"] == [1]
    assert out.get("best_layer") == 0
    assert out.get("evaluation_mode") == "free_running_argmax"

    out_dir = tmp_path / "token_eval_out"
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "position_metrics.json").exists()
    assert (out_dir / "errors.jsonl").exists()
    assert (out_dir / "examples.md").exists()

    summary = json.loads((out_dir / "summary.json").read_text())
    assert len(summary["replacement_depths"]) == 1
    assert list(summary["depth_metrics"].keys()) == ["1"]


def test_main_accepts_actformer_run_dir(tmp_path, monkeypatch):
    import src.actformer.token_eval as te

    run_dir = tmp_path / "run_20260301_162139"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "best.pt").write_bytes(b"")
    (run_dir / "config.yaml").write_text("token_eval:\\n  output_dir: " + str(tmp_path / "out") + "\\n")

    captured: dict[str, object] = {}

    def _fake_eval(cfg):
        captured["cfg"] = cfg
        return {"num_docs": 1, "replacement_depths": [1]}

    monkeypatch.setattr(te, "evaluate_token_replacement", _fake_eval)
    monkeypatch.setattr(
        te.argparse.ArgumentParser,
        "parse_args",
        lambda self: type(
            "Args",
            (),
            {"config": "configs/default.yaml", "actformer_run_dir": str(run_dir)},
        )(),
    )

    te.main()
    cfg = captured["cfg"]
    assert cfg["token_eval"]["actformer_checkpoint"] == str(run_dir / "best.pt")
