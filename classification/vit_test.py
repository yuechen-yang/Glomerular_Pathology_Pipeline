from transformers import ViTForImageClassification, ViTFeatureExtractor, Trainer

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image

# 1. 自定义 ImageFolder 仅保留 _mask.jpg 文件
class MaskOnlyImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        # 过滤 samples，只保留文件名以 _mask.jpg 结尾的图片
        self.samples = [s for s in self.samples if s[0].endswith("_img.png")]
        # 同步更新 targets 列表
        self.targets = [s[1] for s in self.samples]

save_path = "./vit-finetuned"
model = ViTForImageClassification.from_pretrained(save_path)
feature_extractor = ViTFeatureExtractor.from_pretrained(save_path)
class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, feature_extractor, transform=None):
        # 假设数据集根目录的结构为：dataset/label1 和 dataset/label2
        self.dataset = MaskOnlyImageFolder(root=root_dir, transform=transform)
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 返回 PIL 图片和对应标签
        image, label = self.dataset[idx]
        # 使用 feature_extractor 进行预处理（例如：resize、归一化等）
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        # 去除 batch 维度，便于后续 batch 拼接
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        inputs["labels"] = torch.tensor(label)
        # inputs = feature_extractor(images=image, return_tensors="pt")
        return inputs

# 3. 定义数据预处理（这里示例设置为 ViT 通常要求的 224x224）
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    # transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


test_dataset = CustomImageDataset(
    root_dir='/home/yangy50/project/dl_final_project/dataset',
    feature_extractor=feature_extractor,
    transform=transform
)
from transformers import TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}



training_args = TrainingArguments(
    output_dir="./vit-finetune",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    # 旧版用 do_eval=True 开启验证
    do_train=True,
    do_eval=True,
    eval_steps=50,
    save_steps=50,
    logging_steps=10,
    learning_rate=5e-5,
    # 旧版没有 load_best_model_at_end；如果需要可手动在回调里实现
)

# 复用之前定义的 compute_metrics
test_trainer = Trainer(
    model=model,
    args=training_args,       # 只要 evaluation 相关的 args（batch_size、device 等）
    compute_metrics=compute_metrics
)


metrics = test_trainer.evaluate(eval_dataset=test_dataset)
print("Test metrics:", metrics)
# 其中 metrics['eval_accuracy'] 就是你的 test accuracy
