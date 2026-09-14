import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import build_model, MEAN, STD

MODEL = "cnn" 
EPOCHS = 15
BATCH_SIZE = 64
LR = 1e-3
AUGMENT = False
USE_SCHEDULER = False

def get_loaders():
    base = [T.ToTensor(), T.Normalize(MEAN, STD)]
    augment = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()] if AUGMENT else []
    train_tf = T.Compose(augment + base)
    val_tf = T.Compose(base)
    train_set = torchvision.datasets.CIFAR10("data", train=True, download=True, transform=train_tf)
    val_set = torchvision.datasets.CIFAR10("data", train=False, download=True, transform=val_tf)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return train_loader, val_loader

def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total

def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history["train_loss"], label="train")
    ax1.plot(history["val_loss"], label="val")
    ax1.set_title("Loss")
    ax1.legend()
    ax2.plot(history["train_acc"], label="train")
    ax2.plot(history["val_acc"], label="val")
    ax2.set_title("Accuracy")
    ax2.legend()
    fig.suptitle(MODEL)
    plt.savefig(f"results/curves_{MODEL}.png")
    plt.close()

def main():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Training {MODEL} on {device}")
        train_loader, val_loader = get_loaders()
        model = build_model(MODEL).to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler =torch.optim.lr_scheduler.StepLR(optimizer, step_size=10,gamma=0.1)if USE_SCHEDULER else None

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_acc = 0.0
        for epoch in range(EPOCHS):
            tl, ta=run_epoch(model, train_loader, loss_fn, device, optimizer)
            vl, va=run_epoch(model, val_loader, loss_fn, device)
            if scheduler:
                scheduler.step()
            for key, val in zip(history, [tl, vl, ta, va]):
                history[key].append(val)
            print(f"Epoch {epoch+1}/{EPOCHS}| Train Loss: {tl:.3f} acc {ta:.3f} |val loss {vl:.3f} acc {va:.3f}")
            if va > best_acc:
                best_acc=va
                torch.save(model.state_dict(), f"best_{MODEL}.pth")
        print(f"Best validation accuracy: {best_acc:.3f}")
        plot_history(history)
if __name__ == "__main__":
        main()