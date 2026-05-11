import os

import torch

from engine import evaluate_classifier
from utils import knn_eval, move_to_device

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


class Trainer:
    def __init__(
        self,
        args,
        method,
        optimizer,
        scheduler,
        trainloader,
        testloader,
        writer,
        device,
        knn_trainloader=None,
        checkpoint_dir=None,
    ):
        self.args = args
        self.method = method
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.trainloader = trainloader
        self.testloader = testloader
        self.writer = writer
        self.device = device
        self.knn_trainloader = knn_trainloader
        self.disable_progress = getattr(args, "disable_progress", False)
        self.best_acc = 0.0
        self.best_knn_acc = 0.0
        self.checkpoint_dir = checkpoint_dir or "./checkpoint"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_one_epoch(self, epoch):
        self.method.model.train()

        train_loss = 0.0
        progress = self.trainloader
        if tqdm is not None and not self.disable_progress:
            progress = tqdm(self.trainloader, desc=f"Epoch {epoch} [train]", leave=False)

        for step, batch in enumerate(progress, start=1):
            batch = move_to_device(batch, self.device)

            self.optimizer.zero_grad()
            loss = self.method(batch)

            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

            if tqdm is not None and not self.disable_progress:
                progress.set_postfix(loss=f"{train_loss / step:.4f}")

        avg_loss = train_loss / len(self.trainloader)
        print(f"[Epoch {epoch}] - Train Loss: {avg_loss:.4f}")
        self.writer.add_scalar("train_loss", avg_loss, epoch)

    def evaluate_supervised(self, epoch):
        # Keep supervised evaluation in one shared helper so training-time eval,
        # checkpoint eval, and pruned-artifact eval report metrics consistently.
        metrics = evaluate_classifier(self.method.model, self.testloader, self.device)
        avg_loss = metrics["loss"]
        acc = metrics["acc"]

        if acc > self.best_acc:
            self.best_acc = acc
            self.save_checkpoint(self.method.model, acc, epoch, "_cls")

        print(
            f"[Epoch {epoch}] - Test Loss: {avg_loss:.4f}, "
            f"Test Acc: {acc:.2f}%, Best Acc: {self.best_acc:.2f}"
        )
        self.writer.add_scalar("test_loss", avg_loss, epoch)
        self.writer.add_scalar("test_acc", acc, epoch)

    def evaluate_ssl(self, epoch):
        if epoch % 5 != 0:
            return

        knn_acc = knn_eval(
            self.method.model,
            self.knn_trainloader,
            self.testloader,
            self.device,
        )

        if knn_acc > self.best_knn_acc:
            self.best_knn_acc = knn_acc
            self.save_checkpoint(self.method.model, knn_acc, epoch, "_knn")

        print(
            f"[Epoch {epoch}] 1NN top-1: {knn_acc:.2f}% "
            f"Best 1nn top-1: {self.best_knn_acc:.2f}%"
        )
        self.writer.add_scalar("knn_acc", knn_acc, epoch)

    def save_checkpoint(self, model, acc, epoch, suffix):
        state = {
            "model": model.state_dict(),
            "acc": acc,
            "epoch": epoch,
            "args": vars(self.args),
        }
        if hasattr(model, "export_config"):
            state["model_config"] = model.export_config()
        if hasattr(model, "export_merged_state"):
            state["merged_model"] = model.export_merged_state()

        filename = os.path.join(self.checkpoint_dir, f"best{suffix}_ckpt.pth")
        torch.save(state, filename)

    def fit(self):
        for epoch in range(self.args.num_epochs):
            self.train_one_epoch(epoch)

            if self.args.mode in {"two_crop", "multi_crop"}:
                self.evaluate_ssl(epoch)
            else:
                self.evaluate_supervised(epoch)

            self.scheduler.step()
