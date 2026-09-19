import sys
import torch
import torchvision.transforms as T
from PIL import Image
from model import build_model, CLASSES, MEAN, STD
def main():
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py IMAGE_PATH [model_name]")
        return
    image_path = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "cnn_improved"

    model = build_model(model_name, pretrained=False)
    model.load_state_dict(torch.load(f"best_{model_name}.pth", map_location="cpu"))
    model.eval()

    transform = T.Compose([T.Resize((32, 32)), T.ToTensor(), T.Normalize(MEAN, STD)])
    image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(model(image), dim=1)[0]

    top = probs.topk(3)
    for p, i in zip(top.values, top.indices):
        print(f"{CLASSES[i]:12s} {p.item():.1%}")
if __name__ == "__main__":
    main()