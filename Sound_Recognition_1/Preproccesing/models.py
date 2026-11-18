"""
SoundSense — Deep Neural Network Architectures for ESC-50 Environmental Sound Recognition
Upgraded multi-modal architectures (SResNet with DMFF & SwiGLU, ResNet-18, VGG-16, CNN-GRU)
with full compatibility for single-channel callers (get_model / model(x)) and dataset_loader.py.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

tvm = models

SIZE = (128, 128)


def load_or_download_weights(model, model_filename):
    """Loads weights/<model_filename> if it exists; otherwise keeps the freshly initialised model."""
    os.makedirs("weights", exist_ok=True)
    local_path = os.path.join("weights", model_filename)
    if not os.path.exists(local_path):
        print(f"No checkpoint at {local_path}; keeping freshly initialised weights.")
        return model.eval()
    print(f"Loading weights into RAM from {local_path}...")
    ckpt = torch.load(local_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state"] if isinstance(ckpt, dict) and "state" in ckpt else ckpt
    model.load_state_dict(state_dict)
    return model.eval()


# ---------------------------------------------------------------------
# Input handling for named multi-feature pipelines
# ---------------------------------------------------------------------
def _resize(x):
    """Any feature -> [B,1,128,128] (resize only, no scaling)."""
    if x.dim() == 2:                         # batch of vectors [B,L]
        x = x.unsqueeze(1).unsqueeze(1)
    elif x.dim() == 3:                       # batch of maps [B,H,W]
        x = x.unsqueeze(1)
    if x.shape[-2:] != SIZE:
        x = F.interpolate(x, size=SIZE, mode="bilinear", align_corners=False)
    return x


def _prep(x, name):
    """_resize, then z-score per sample for everything except the RP."""
    x = _resize(x)
    if name != "rp":
        x = (x - x.mean(dim=(1, 2, 3), keepdim=True)) / (x.std(dim=(1, 2, 3), keepdim=True) + 1e-6)
    return x


class _Base(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        inputs = tuple(inputs)
        if not inputs:
            raise ValueError("inputs must name at least one feature")
        self.inputs = inputs
        self.criterion = nn.CrossEntropyLoss()

    def _maps(self, mel, rp, feats):
        src = dict(feats or {})
        if mel is not None:
            src.setdefault("log_mel", mel)
        if rp is not None:
            src.setdefault("rp", rp)
        if "rp" not in src and "recurrence_plot" in src:
            src["rp"] = src["recurrence_plot"]
        missing = [n for n in self.inputs if src.get(n) is None]
        if missing:
            raise ValueError(f"This model uses {self.inputs} but {missing} were not provided.")
        return {n: _prep(src[n], n) for n in self.inputs}

    def _stack(self, mel, rp, feats):
        maps = self._maps(mel, rp, feats)
        return torch.cat([maps[n] for n in self.inputs], dim=1)    # [B, len(inputs), 128, 128]

    def _finish(self, logits, label):
        if label is not None:
            return logits, self.criterion(logits, label)
        return logits


def _adapt_conv(old, in_ch):
    """Pretrained first conv -> in_ch channels."""
    new = nn.Conv2d(in_ch, old.out_channels, old.kernel_size, old.stride, old.padding, bias=old.bias is not None)
    with torch.no_grad():
        new.weight.copy_(old.weight.mean(dim=1, keepdim=True).repeat(1, in_ch, 1, 1) * (3.0 / in_ch))
        if old.bias is not None:
            new.bias.copy_(old.bias)
    return new


def _format_inputs(mel, rp, target_size=(128, 128)):
    if mel.dim() == 3:
        mel = mel.unsqueeze(1)
    if rp is None:
        rp = mel
    elif rp.dim() == 3:
        rp = rp.unsqueeze(1)

    if mel.shape[-2:] != target_size:
        mel = F.interpolate(mel, size=target_size, mode="bilinear", align_corners=False)
    if rp.shape[-2:] != target_size:
        rp = F.interpolate(rp, size=target_size, mode="bilinear", align_corners=False)

    return mel, rp


# =====================================================================
# 1. Baseline: AudioResNet (ResNet-18)
# =====================================================================
class AudioResNet(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.criterion = nn.CrossEntropyLoss()

        original_conv1 = self.model.conv1
        self.model.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)

        with torch.no_grad():
            self.model.conv1.weight[:, 0:1, :, :] = original_conv1.weight[:, [0], :, :]
            self.model.conv1.weight[:, 1:2, :, :] = original_conv1.weight[:, [1], :, :]

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, mel, rp=None, label=None):
        mel, rp = _format_inputs(mel, rp)
        x = torch.cat([mel, rp], dim=1)
        logits = self.model(x)

        if label is not None:
            loss = self.criterion(logits, label)
            return logits, loss
        return logits


# =====================================================================
# 2. Baseline: AudioVGG16
# =====================================================================
class AudioVGG16(_Base):
    def __init__(self, num_classes=50, inputs=("log_mel", "rp"), pretrained=True):
        super().__init__(inputs)
        vgg = tvm.vgg16_bn(weights=tvm.VGG16_BN_Weights.DEFAULT if pretrained else None)
        vgg.features[0] = _adapt_conv(vgg.features[0], len(self.inputs))
        self.features = vgg.features
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.5), nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(256, num_classes),
        )

    def forward(self, mel=None, rp=None, label=None, feats=None):
        return self._finish(self.head(self.features(self._stack(mel, rp, feats))), label)


# =====================================================================
# 3. Baseline: AudioCNNGRU
# =====================================================================
class AudioCNNGRU(_Base):
    def __init__(self, num_classes=50, inputs=("log_mel", "rp"), hidden=128):
        super().__init__(inputs)

        def conv_block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.cnn = nn.Sequential(
            conv_block(len(self.inputs), 32),
            conv_block(32, 64),
            conv_block(64, 128),
        )
        self.gru = nn.GRU(128 * 16, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, mel=None, rp=None, label=None, feats=None):
        x = self.cnn(self._stack(mel, rp, feats))                   # [B, 128, 16, 16]
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)    # [B, 16, 2048]
        out, _ = self.gru(x)                                        # [B, 16, 256]
        logits = self.fc(self.drop(out.mean(dim=1)))                # [B, 50]
        return self._finish(logits, label)


# =====================================================================
# 4. Proposed Model: AudioSResNet with DMFF & SwiGLU (Log-Mel + RP, no GFCC)
# =====================================================================
class SwiGLUResBlock(nn.Module):
    """Residual block whose second conv is SwiGLU-gated."""
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.value = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.gate = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Identity() if (stride == 1 and cin == cout) else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.value(h) * F.silu(self.gate(h))
        return F.relu(self.bn2(h) + self.shortcut(x))


class DMFF(nn.Module):
    """Dual Model Feature Fusion via Cross-Attention Token Matching."""
    def __init__(self, ch, grid=8, dim=64, heads=4):
        super().__init__()
        self.grid = grid
        self.pool = nn.AdaptiveAvgPool2d(grid)
        self.in_a, self.in_b = nn.Linear(ch, dim), nn.Linear(ch, dim)
        self.pos_a = nn.Parameter(torch.zeros(1, grid * grid, dim))
        self.pos_b = nn.Parameter(torch.zeros(1, grid * grid, dim))
        self.norm_a, self.norm_b = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn_a = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.attn_b = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.out_a, self.out_b = nn.Linear(dim, ch), nn.Linear(dim, ch)

    def _tokens(self, x, proj, pos, norm):
        return norm(proj(self.pool(x).flatten(2).transpose(1, 2)) + pos)

    def _back(self, e, proj, size):
        m = proj(e).transpose(1, 2).reshape(e.shape[0], -1, self.grid, self.grid)
        return F.interpolate(m, size=size, mode="bilinear", align_corners=False)

    def forward(self, a, b):
        ta = self._tokens(a, self.in_a, self.pos_a, self.norm_a)
        tb = self._tokens(b, self.in_b, self.pos_b, self.norm_b)
        ea, _ = self.attn_a(ta, tb, tb)
        eb, _ = self.attn_b(tb, ta, ta)
        size = a.shape[-2:]
        return a + self._back(ea, self.out_a, size), b + self._back(eb, self.out_b, size)


class AudioSResNet(_Base):
    """Proposed Dual-Stream SResNet utilizing DMFF, Dilated RP Stem, and SwiGLU (without GFCC)."""
    def __init__(self, num_classes=50, inputs=("log_mel", "rp"), stem=32):
        super().__init__(inputs)
        self.lin_names = [n for n in self.inputs if n != "rp"]
        self.has_lin = len(self.lin_names) > 0
        self.has_rp = "rp" in self.inputs

        def stem_conv(cin, **kw):
            return nn.Sequential(nn.Conv2d(cin, stem, 3, bias=False, **kw), nn.BatchNorm2d(stem), nn.ReLU(inplace=True))

        if self.has_lin:
            self.stem_lin = stem_conv(len(self.lin_names), stride=2, padding=1)
        if self.has_rp:
            self.stem_rp = stem_conv(1, stride=2, padding=2, dilation=2)
        self.dmff = DMFF(stem) if (self.has_lin and self.has_rp) else None

        c_in = stem * (2 if self.dmff is not None else 1)
        self.trunk = nn.Sequential(
            SwiGLUResBlock(c_in, 64, 1), SwiGLUResBlock(64, 128, 2),
            SwiGLUResBlock(128, 256, 2), SwiGLUResBlock(256, 512, 2),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(512, num_classes))

    def forward(self, mel=None, rp=None, label=None, feats=None):
        maps = self._maps(mel, rp, feats)
        f_lin = self.stem_lin(torch.cat([maps[n] for n in self.lin_names], dim=1)) if self.has_lin else None
        f_rp = self.stem_rp(maps["rp"]) if self.has_rp else None
        if self.dmff is not None:
            f_lin, f_rp = self.dmff(f_lin, f_rp)
            x = torch.cat([f_lin, f_rp], dim=1)
        else:
            x = f_lin if self.has_lin else f_rp
        return self._finish(self.head(self.trunk(x)), label)


# =====================================================================
# Registry & Native Builder
# =====================================================================
MODELS = {
    "resnet": AudioResNet,
    "vgg16": AudioVGG16,
    "cnn_gru": AudioCNNGRU,
    "sresnet": AudioSResNet,
}

LEGACY = {"resnet"}


def build_model(name, inputs=("log_mel", "rp"), num_classes=50, pretrained=True):
    if name in LEGACY:
        model = MODELS[name](num_classes=num_classes)
        model.inputs = ("log_mel", "rp")
        return model
    kw = {"pretrained": pretrained} if name == "vgg16" else {}
    return MODELS[name](num_classes=num_classes, inputs=tuple(inputs), **kw)


# =====================================================================
# Compatibility Layer for Single-Tensor Callers (get_model / model(x))
# =====================================================================
class _SingleTensorCompat(nn.Module):
    """Bridges single-spectrogram callers (model(x)) to dual-input models."""
    def __init__(self, inner_model):
        super().__init__()
        self.inner = inner_model

    def forward(self, x, label=None):
        if isinstance(self.inner, AudioResNet):
            return self.inner(x, x, label=label)
        feats = {name: x for name in getattr(self.inner, "inputs", ("log_mel", "rp"))}
        return self.inner(feats=feats, label=label)


class ResNet18_ESC50(_SingleTensorCompat):
    def __init__(self, in_channels=1, num_classes=50):
        super().__init__(build_model("resnet", inputs=("log_mel", "rp"), num_classes=num_classes))


class VGG16_ESC50(_SingleTensorCompat):
    def __init__(self, in_channels=1, num_classes=50, dropout=0.5):
        super().__init__(build_model("vgg16", inputs=("log_mel", "rp"), num_classes=num_classes, pretrained=True))


class SResNet(_SingleTensorCompat):
    def __init__(self, in_channels=1, num_classes=50):
        super().__init__(build_model("sresnet", inputs=("log_mel", "rp"), num_classes=num_classes))


class CNNGRU(_SingleTensorCompat):
    def __init__(self, in_channels=1, num_classes=50, hidden_size=128):
        super().__init__(build_model("cnn_gru", inputs=("log_mel", "rp"), num_classes=num_classes))


def get_model(model_name: str, num_classes: int = 50) -> nn.Module:
    """Drop-in replacement for the friend's get_model() factory."""
    name = model_name.lower().replace("-", "").replace("_", "")
    if "resnet18" in name or name == "resnet":
        return ResNet18_ESC50(num_classes=num_classes)
    elif "vgg16" in name:
        return VGG16_ESC50(num_classes=num_classes)
    elif "sresnet" in name:
        return SResNet(num_classes=num_classes)
    elif "cnngru" in name:
        return CNNGRU(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name '{model_name}'. Choose from ['resnet18', 'vgg16', 'sresnet', 'cnngru'].")


# =====================================================================
# Standalone Training Execution (Uses dataset_loader.py)
# =====================================================================
def mixup_batch(feats_b, label_b, num_classes=50, alpha=0.4):
    """Blends pairs of multi-modal features and returns soft one-hot target labels."""
    if alpha <= 0:
        return feats_b, F.one_hot(label_b, num_classes=num_classes).float()

    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    batch_size = label_b.size(0)
    perm = torch.randperm(batch_size, device=label_b.device)

    mixed_feats = {
        k: lam * v + (1.0 - lam) * v[perm]
        for k, v in feats_b.items()
    }

    y_a = F.one_hot(label_b, num_classes=num_classes).float()
    y_b = F.one_hot(label_b[perm], num_classes=num_classes).float()
    mixed_targets = lam * y_a + (1.0 - lam) * y_b
    return mixed_feats, mixed_targets


def soft_cross_entropy(logits, soft_targets):
    """Cross-entropy loss supporting soft Mixup target distributions."""
    log_probs = F.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()


if __name__ == "__main__":
    import sys
    from pathlib import Path
    import torch.optim as optim

    # Add parent directory so 'Preproccesing' is recognized as a package
    parent_dir = str(Path(__file__).resolve().parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    from Preproccesing.dataset_loader import get_dataloaders
    loaders = get_dataloaders(fold=1, batch_size=32)
    train_loader, val_loader = loaders["train"], loaders["val"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Executing training on device: {device} | "
        f"Train clips: {len(train_loader.dataset)} | Val clips: {len(val_loader.dataset)}"
    )

    models_to_train = {
        #"sresnet": build_model("sresnet", inputs=("log_mel", "rp"), num_classes=50),
        #"resnet":  build_model("resnet",  inputs=("log_mel", "rp"), num_classes=50),
        #"cnn_gru": build_model("cnn_gru", inputs=("log_mel", "rp"), num_classes=50),
        "vgg16":   build_model("vgg16",   inputs=("log_mel", "rp"), num_classes=50),
    }

    os.makedirs("weights", exist_ok=True)
    max_epochs = 1
    early_stop_patience = 10

    for model_name, model in models_to_train.items():
        print("\n" + "=" * 65)
        print(f"TRAINING MODEL: {model_name.upper()} | Inputs: {model.inputs}")
        print("=" * 65)

        model = model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

        best_val_acc = 0.0
        best_val_loss = float("inf")
        epochs_no_improve = 0
        checkpoint_path = os.path.join("weights", f"{model_name}_best_model.pth")

        for epoch in range(1, max_epochs + 1):
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0

            for batch in train_loader:
                label_b = batch["target"].to(device)
                feats_b = {
                    "log_mel": batch["log_mel"].to(device),
                    "rp": batch["recurrence_plot"].to(device),
                }

                mixed_feats, soft_targets = mixup_batch(feats_b, label_b, num_classes=50, alpha=0.2)

                optimizer.zero_grad()
                if model_name == "resnet":
                    logits = model(mixed_feats["log_mel"], mixed_feats["rp"])
                else:
                    logits = model(feats=mixed_feats)

                loss = soft_cross_entropy(logits, soft_targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item() * label_b.size(0)
                train_correct += (torch.argmax(logits, dim=1) == label_b).sum().item()
                train_total += label_b.size(0)

            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    label_b = batch["target"].to(device)
                    feats_b = {
                        "log_mel": batch["log_mel"].to(device),
                        "rp": batch["recurrence_plot"].to(device),
                    }

                    if model_name == "resnet":
                        logits, loss = model(feats_b["log_mel"], feats_b["rp"], label=label_b)
                    else:
                        logits, loss = model(feats=feats_b, label=label_b)

                    val_loss += loss.item() * label_b.size(0)
                    val_correct += (torch.argmax(logits, dim=1) == label_b).sum().item()
                    val_total += label_b.size(0)

            avg_train_loss = train_loss / train_total
            avg_val_loss = val_loss / val_total
            tr_acc = (train_correct / train_total) * 100.0
            va_acc = (val_correct / val_total) * 100.0
            current_lr = optimizer.param_groups[0]["lr"]

            scheduler.step(avg_val_loss)

            print(
                f"[{model_name.upper()}] Epoch [{epoch:02d}/{max_epochs:02d}] (LR: {current_lr:.1e}) | "
                f"Train: {avg_train_loss:.4f} ({tr_acc:.2f}%) | "
                f"Val: {avg_val_loss:.4f} ({va_acc:.2f}%)"
            )

            improved = (va_acc > best_val_acc) or (va_acc == best_val_acc and avg_val_loss < best_val_loss)
            if improved:
                best_val_acc = va_acc
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stop_patience:
                    print(f"Early stopping triggered after {epoch} epochs (no val improvement for {early_stop_patience} epochs).")
                    break

        print(
            f"Best Val Accuracy for {model_name.upper()}: "
            f"{best_val_acc:.2f}% (Loss: {best_val_loss:.4f}) -> Saved to {checkpoint_path}"
        )

    print("\nAll model checkpoints successfully trained and saved in /weights.")