from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast


@dataclass
class TrainingSummary:
    best_epoch: int
    best_metrics: Dict[str, float]
    history: List[Dict[str, float]] = field(default_factory=list)
    best_checkpoint: Optional[Path] = None
    last_checkpoint: Optional[Path] = None


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler=None,
        grad_clip: Optional[float] = None,
        use_amp: bool = False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.use_amp = bool(use_amp)
        self.scaler = GradScaler(enabled=self.use_amp)
        self.model.to(self.device)

    def _run_epoch(self, loader: DataLoader, train: bool = True) -> Dict[str, float]:
        if train:
            self.model.train()
        else:
            self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        ctx = torch.enable_grad if train else torch.no_grad
        with ctx():
            for images, labels in loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with autocast(enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                        if self.grad_clip:
                            self.scaler.unscale_(self.optimizer)
                            nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.grad_clip
                            )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        if self.grad_clip:
                            nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.grad_clip
                            )
                        self.optimizer.step()

                running_loss += loss.item() * labels.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        loss = running_loss / max(1, total)
        acc = correct / max(1, total)
        return {"loss": loss, "acc": acc}

    def fit(
        self,
        loaders: Dict[str, DataLoader],
        epochs: int,
        patience: int,
        output_dir: Path,
        checkpoint_prefix: str = "emotion_model",
        checkpoint_freq: int = 5,
    ) -> TrainingSummary:
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        best_path = output_dir / f"{checkpoint_prefix}_best.pth"
        last_path = output_dir / f"{checkpoint_prefix}.pth"
        history_path = output_dir / "training_history.json"

        history: List[Dict[str, float]] = []
        best_metric = 0.0
        best_epoch = 0
        patience_ctr = 0

        train_loader = loaders["train"]
        val_loader = loaders["val"]

        for epoch in range(1, epochs + 1):
            train_metrics = self._run_epoch(train_loader, train=True)
            val_metrics = self._run_epoch(val_loader, train=False)

            if self.scheduler:
                try:
                    self.scheduler.step(val_metrics["acc"])
                except TypeError:
                    self.scheduler.step()

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["acc"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            history.append(epoch_record)
            _write_json(history_path, history)

            torch.save(self.model.state_dict(), last_path)

            if val_metrics["acc"] > best_metric:
                best_metric = val_metrics["acc"]
                best_epoch = epoch
                patience_ctr = 0
                torch.save(self.model.state_dict(), best_path)
            else:
                patience_ctr += 1

            if checkpoint_freq and epoch % checkpoint_freq == 0:
                torch.save(
                    self.model.state_dict(),
                    output_dir / f"{checkpoint_prefix}_epoch{epoch}.pth",
                )

            if patience_ctr >= patience:
                break

        summary = TrainingSummary(
            best_epoch=best_epoch,
            best_metrics={"val_acc": best_metric},
            history=history,
            best_checkpoint=best_path if best_path.exists() else None,
            last_checkpoint=last_path if last_path.exists() else None,
        )
        return summary


def _write_json(path: Path, payload):
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
