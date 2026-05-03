import torch

# 🔥 IMPORTANT: allow YOLO model class
from models.yolo import DetectionModel
import torch.serialization

torch.serialization.add_safe_globals({
    "models.yolo.DetectionModel": DetectionModel
})

# ✅ Load with correct flag
ckpt = torch.load("best.pt", map_location="cpu", weights_only=False)

# 🔥 Clean unwanted keys
if isinstance(ckpt, dict):
    ckpt.pop("optimizer", None)
    ckpt.pop("training_results", None)
    ckpt.pop("wandb_id", None)

# ✅ Save clean model
torch.save(ckpt, "best_clean.pt")

print("✅ Clean model saved as best_clean.pt")