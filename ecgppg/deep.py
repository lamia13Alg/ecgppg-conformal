"""Deep-learning population baseline: a small two-branch 1-D CNN on the same beat representation
(64-sample ECG window, 32-sample PPG window, 7 rhythm/pulse scalars). Hyper-parameters are fixed
a priori (5 epochs, Adam 1e-3, batch 512, class-weighted loss); no early stopping on test data."""
import numpy as np


def _net():
    import torch.nn as nn
    import torch

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.e = nn.Sequential(nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
                                   nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(8))
            self.p = nn.Sequential(nn.Conv1d(1, 8, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
                                   nn.Conv1d(8, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(4))
            self.s = nn.Sequential(nn.Linear(7, 16), nn.ReLU())
            self.h = nn.Sequential(nn.Linear(256 + 64 + 16, 64), nn.ReLU(), nn.Dropout(0.2),
                                   nn.Linear(64, 1))

        def forward(self, xe, xp, xs):
            a = self.e(xe.unsqueeze(1)).flatten(1)
            b = self.p(xp.unsqueeze(1)).flatten(1)
            c = self.s(xs)
            return self.h(torch.cat([a, b, c], 1)).squeeze(1)
    return Net()


def _split(Xe, Xp):
    xe = np.clip(Xe[:, :64], -20, 20).astype(np.float32)
    xp = np.clip(Xp[:, 5:37], -20, 20).astype(np.float32)
    xs = np.clip(np.column_stack([Xe[:, 64:66], Xp[:, :5]]), -10, 10).astype(np.float32)
    return xe, xp, xs


class CNNModel:
    def __init__(self, seed=42, epochs=5, batch=512, lr=1e-3):
        self.seed, self.epochs, self.batch, self.lr = seed, epochs, batch, lr
        self._cache = {}

    def fit(self, Xe, Xp, y):
        import torch
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.net = _net()
        xe, xp, xs = (torch.from_numpy(a) for a in _split(Xe, Xp))
        yt = torch.from_numpy(y.astype(np.float32))
        pos = float(y.sum()); neg = float(len(y) - y.sum())
        lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / max(pos, 1.0)))
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        g = torch.Generator().manual_seed(self.seed)
        for ep in range(self.epochs):
            self.net.train()
            perm = torch.randperm(len(yt), generator=g)
            tot = 0.0
            for k in range(0, len(perm), self.batch):
                b = perm[k:k + self.batch]
                opt.zero_grad()
                loss = lossf(self.net(xe[b], xp[b], xs[b]), yt[b])
                loss.backward()
                opt.step()
                tot += float(loss) * len(b)
            print(f"  CNN epoch {ep + 1}/{self.epochs} loss {tot / len(yt):.4f}", flush=True)
        self.net.eval()
        return self

    def logit(self, r):
        import torch
        key = str(r["id"])
        if key not in self._cache:
            xe, xp, xs = _split(r["Xe"], r["Xp"])
            out = []
            with torch.no_grad():
                for k in range(0, len(xe), 8192):
                    out.append(self.net(torch.from_numpy(xe[k:k + 8192]), torch.from_numpy(xp[k:k + 8192]),
                                        torch.from_numpy(xs[k:k + 8192])).numpy())
            self._cache[key] = np.concatenate(out).astype(np.float32) if out else np.zeros(0, np.float32)
        return self._cache[key].astype(float)
