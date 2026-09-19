import sys
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from model import build_model, CLASSES, MEAN, STD
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "cnn_improved"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    test_set = torchvision.datasets.CIFAR10("data", train=False, download=True, transform=transform)
    loader = DataLoader(test_set, batch_size=256, shuffle=False)

    model = build_model(model_name, pretrained=False).to(device)
    model.load_state_dict(torch.load(f"best_{model_name}.pth", map_location=device))
    model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for images, targets in loader:
            outputs = model(images.to(device))
            preds += outputs.argmax(1).cpu().tolist()
            labels += targets.tolist()
            
    print(f"Per-class accuracy for {model_name}:")
    for i, name in enumerate(CLASSES):
        idx = [j for j, label in enumerate(labels) if label == i]
        acc = sum(preds[j] == i for j in idx) / len(idx)
        print(f" {name:12s} {acc:.3f}")
    overall = sum(p == l for p, l in zip(preds, labels)) / len(labels)
    print(f"Overall: {overall:.3f}")

    fig, ax = plt.subplots(figsize=(9, 9))
    ConfusionMatrixDisplay.from_predictions(labels, preds, display_labels=CLASSES,

    ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(f"Confusion matrix: {model_name}")
    plt.tight_layout()
    plt.savefig(f"results/confusion_{model_name}.png")
    print(f"Saved results/confusion_{model_name}.png")

if __name__ == "__main__":
    main()