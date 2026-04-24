import torch
import torch.nn.functional as F

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
        self.best_acc = 0.0
        self.best_knn_acc = 0.0

    def train_one_epoch(self, epoch):
        self.method.model.train()

        train_loss = 0.0
        progress = self.trainloader
        if tqdm is not None:
            progress = tqdm(self.trainloader, desc=f"Epoch {epoch} [train]", leave=False)

        for step, batch in enumerate(progress, start=1):
            batch = move_to_device(batch, self.device)

            self.optimizer.zero_grad()
            loss = self.method(batch)

            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

            if tqdm is not None:
                progress.set_postfix(loss=f"{train_loss / step:.4f}")

        avg_loss = train_loss / len(self.trainloader)
        print(f"[Epoch {epoch}] - Train Loss: {avg_loss:.4f}")
        self.writer.add_scalar("train_loss", avg_loss, epoch)

    def evaluate_supervised(self, epoch):
        self.method.model.eval()

        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            progress = self.testloader
            if tqdm is not None:
                progress = tqdm(self.testloader, desc=f"Epoch {epoch} [test]", leave=False)

            for step, batch in enumerate(progress, start=1):
                images, labels = move_to_device(batch, self.device)

                outputs = self.method.model(images)
                loss = F.cross_entropy(outputs, labels)

                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                if tqdm is not None:
                    progress.set_postfix(
                        loss=f"{test_loss / step:.4f}",
                        acc=f"{100.0 * correct / total:.2f}",
                    )

        avg_loss = test_loss / len(self.testloader)
        acc = 100.0 * correct / total

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
        }

        quant_suffix = "_quantized" if getattr(self.args, "use_quantization", False) else ""
        filename = f"./checkpoint/{self.args.method}_{self.args.model}{quant_suffix}{suffix}_ckpt.pth"
        torch.save(state, filename)

    def fit(self):
        for epoch in range(self.args.num_epochs):
            self.train_one_epoch(epoch)

            if self.args.mode in {"two_crop", "multi_crop"}:
                self.evaluate_ssl(epoch)
            else:
                self.evaluate_supervised(epoch)

            self.scheduler.step()

