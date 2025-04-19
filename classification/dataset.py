from torch.utils.data import Dataset
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, feature_extractor):
        self.dataset = datasets.ImageFolder(root=root_dir)
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # 使用 feature_extractor 来预处理图片
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        # inputs 中的像素信息在 "pixel_values" 字段，需要 squeeze 才适合 batch 操作
        inputs = {key: val.squeeze() for key, val in inputs.items()}
        inputs["labels"] = torch.tensor(label)
        return inputs

# 构造训练和验证数据集
train_data = CustomImageDataset(root_dir='dataset/train', feature_extractor=feature_extractor)
val_data = CustomImageDataset(root_dir='dataset/val', feature_extractor=feature_extractor)

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./vit-finetune",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="steps",
    eval_steps=50,
    save_steps=50,
    learning_rate=5e-5,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",  # 需提前定义一个 accuracy 计算函数
)

import numpy as np
from datasets import load_metric
metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    compute_metrics=compute_metrics
)

trainer.train()
