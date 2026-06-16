"""OCR 인식 어댑터 (1차: CRNN + CTC, torch-only → [B, T, NC+1] softmax 확률).

dataset.names = charset(문자 리스트, NC=문자수). 출력 채널 = NC+1(마지막=CTC blank).
softmax 베이킹(참조 코어 paddle-rec 동형) → 출력 확률[0,1]; argmax+CTC-collapse 는 VSC 가 수행.
(PaddleOCR det+rec+cls 풀스택은 paddle 의존 — 동일 어댑터 뒤 엔진 교체로 확장.)
heavy lib(torch)는 메서드 내부 lazy import.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..registry import register_trainer
from .base import BaseTrainer


@register_trainer("ocr", "crnn_ctc", "parseq")
class OcrCrnnTrainer(BaseTrainer):

    def prepare_data(self) -> Any:
        from ..data.readers.ocr import read_ocr_labels
        rows = read_ocr_labels(self.cfg.dataset.root)
        if not rows:
            raise ValueError("OCR labels.txt(이미지<TAB>텍스트) 0행 — dataset.root 확인")
        return rows

    def _model(self, n_classes: int, c: int):
        from torch import nn

        class _CRNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.cnn = nn.Sequential(
                    nn.Conv2d(c, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),    # H/2 W/2
                    nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),  # H/4 W/4
                    nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(True),
                    nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True),
                    nn.MaxPool2d((2, 1), (2, 1)),                                    # H/8 W/4
                )
                self.pool = nn.AdaptiveAvgPool2d((1, None))   # H→1, W(=T) 유지
                self.rnn = nn.LSTM(256, 256, num_layers=2, bidirectional=True, batch_first=True)
                self.fc = nn.Linear(512, n_classes)

            def forward(self, x):
                f = self.pool(self.cnn(x)).squeeze(2).permute(0, 2, 1)   # [B,T,256]
                f, _ = self.rnn(f)
                return self.fc(f)                                        # [B,T,n_classes]

        return _CRNN()

    def train(self, data: Any) -> Path:
        import cv2
        import numpy as np
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset

        t, e = self.cfg.train, self.cfg.export
        dev = f"cuda:{t.gpus[0]}" if (t.gpus and torch.cuda.is_available()) else "cpu"
        H, W, C = e.input.h, e.input.w, e.input.c
        names = self.names
        ch2idx = {ch: i for i, ch in enumerate(names)}
        blank = len(names)                                  # CTC blank = NC
        rows = [(p, txt) for p, txt in data if all(c in ch2idx for c in txt)]
        if not rows:
            raise ValueError("charset(dataset.names)로 인코딩 가능한 텍스트 행 0개")

        class OcrDS(Dataset):
            def __len__(self): return len(rows)
            def __getitem__(self, i):
                ip, txt = rows[i]
                img = cv2.imread(ip)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if C == 3 else \
                    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., None]
                img = cv2.resize(img, (W, H)).reshape(H, W, C)
                x = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)
                tgt = torch.tensor([ch2idx[c] for c in txt], dtype=torch.long)
                return torch.from_numpy(x), tgt

        def _collate(batch):
            xs = torch.stack([b[0] for b in batch])
            tgts = torch.cat([b[1] for b in batch])
            tlens = torch.tensor([len(b[1]) for b in batch], dtype=torch.long)
            return xs, tgts, tlens

        loader = DataLoader(OcrDS(), batch_size=t.batch, shuffle=True, num_workers=0,
                            collate_fn=_collate)
        model = self._model(len(names) + 1, C).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=t.optimizer.lr,
                                weight_decay=t.optimizer.weight_decay)
        ctc = nn.CTCLoss(blank=blank, zero_infinity=True)
        model.train()
        loss = torch.tensor(0.0)
        for ep in range(t.epochs):
            for xs, tgts, tlens in loader:
                xs = xs.to(dev)
                logits = model(xs)                              # [B,T,n_classes]
                logp = logits.log_softmax(2).permute(1, 0, 2)   # [T,B,n_classes]
                in_len = torch.full((xs.size(0),), logits.size(1), dtype=torch.long)
                opt.zero_grad()
                loss = ctc(logp, tgts, in_len, tlens)
                loss.backward()
                opt.step()
            print(f"[ocr] epoch {ep+1}/{t.epochs} loss={float(loss):.4f}")
        ckpt = self.out_dir / "best.pt"
        torch.save({"state_dict": model.state_dict(), "num_classes": len(names)}, ckpt)
        return ckpt

    def export_to_vsc(self, ckpt: Path) -> dict[str, Path]:
        import torch
        from torch import nn
        from torch.nn import functional as F

        from ..export import VscExporter, introspect_output_shape

        e = self.cfg.export
        sd = torch.load(str(ckpt), map_location="cpu")
        base = self._model(len(self.names) + 1, e.input.c)
        base.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd)
        base.eval()

        class _OcrExport(nn.Module):   # softmax 베이킹(charset 축) → [B,T,NC+1] 확률
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, x): return F.softmax(self.m(x), dim=-1)

        model = _OcrExport(base).eval()
        dst = self.out_dir / "model.onnx"
        dummy = torch.randn(1, e.input.c, e.input.h, e.input.w)
        dyn = ({e.io_names.input: {0: "B"}, e.io_names.output: {0: "B", 1: "T"}}
               if e.dynamic_axes else None)
        torch.onnx.export(model, dummy, str(dst), opset_version=e.opset,
                          input_names=[e.io_names.input], output_names=[e.io_names.output],
                          dynamic_axes=dyn)
        out_shape = introspect_output_shape(dst)      # [1, T, NC+1]
        return VscExporter(self.cfg).write(dst, out_shape, weights_name="model.onnx")
