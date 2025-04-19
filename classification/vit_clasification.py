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

# 2. 自定义 Dataset，将过滤后的数据与 feature_extractor 结合
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

# 4. 设置模型预处理器和模型
model_name = "google/vit-base-patch16-224"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name, use_auth_token=False)
model = ViTForImageClassification.from_pretrained(model_name, num_labels=2,ignore_mismatched_sizes=True, use_auth_token=False)

# 5. 加载整个数据集（假设数据集目录为 'dataset'，且子目录中存放不同标签的数据）
dataset = CustomImageDataset(root_dir='/home/yangy50/project/dl_final_project/KPMP_Patches', feature_extractor=feature_extractor, transform=transform)

# 6. 手动分割训练集和验证集（此处以 80%/20% 进行分割）
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])

# 7. 构造 DataLoader
train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False, num_workers=4)

# 8. 后续训练部分（Trainer 例子）
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score

# 定义 accuracy 评价指标
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}

# training_args = TrainingArguments(
#     output_dir="./vit-finetune",
#     num_train_epochs=5,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     evaluation_strategy="steps",
#     eval_steps=50,
#     save_steps=50,
#     learning_rate=5e-5,
#     logging_steps=10,
#     load_best_model_at_end=True,
#     metric_for_best_model="accuracy",
# )


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



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    compute_metrics=compute_metrics
)

trainer.train()

# 将 model、config、feature_extractor（可选）一起保存到指定目录
save_path = "./vit-finetuned"
trainer.save_model(save_path)
feature_extractor.save_pretrained(save_path)
