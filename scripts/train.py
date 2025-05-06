import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from models import ECAResTCN, MiniECAResTCN, ECAResNetMfccClassifier, SeparableTr  # 确保models.py在同一目录下
from datasets import ESC50Dataset
from torch.utils.data import DataLoader


def train_model(t_loader, v_loader, model, criterion, optimizer, scheduler):
    # 训练记录
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lrs": []
    }

    # 早停初始化
    best_val_loss = np.inf
    epochs_no_improve = 0
    early_stop = False

    # 时间记录
    t = time.localtime()
    current_time = time.strftime("%m%d-%H%M", t)

    for epoch in range(config["epochs"]):
        if early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in t_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计指标
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # 更新学习率
        scheduler.step()

        # 计算训练指标
        train_loss = running_loss / total
        train_acc = correct / total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["lrs"].append(scheduler.get_last_lr()[0])

        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in v_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # 打印进度
        print(f"Epoch {epoch + 1}/{config['epochs']}")
        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc * 100:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} Acc: {val_acc * 100:.2f}%")
        print(f"Learning Rate: {history['lrs'][-1]:.2e}\n")

        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # 保存最佳模型
            # torch.save(model.state_dict(), config["model_save_path"] + 'best_model' + current_time + '.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config["patience"]:
                early_stop = True

    # 训练结束可视化
    plt.figure(figsize=(12, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Validation")
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train")
    plt.plot(history["val_acc"], label="Validation")
    plt.title("Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"../plots/{current_time}.png")
    plt.show()

    return model, history


def main():
    # model = ECAResNetMfccClassifier(in_channels=128, num_classes=50, layers=[2, 2, 6, 2]).to(device)
    model = SeparableTr().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(),
                            lr=config["lr"],
                            weight_decay=config["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["t_max"])

    train_loader = DataLoader(ESC50Dataset(root_dir='../data/ESC-50', mode='train', folds=[1, 2, 3, 4], augment=True,
                                           out_dim=3),
                              batch_size=config["batch_size"], shuffle=True, num_workers=8)

    valid_loader = DataLoader(ESC50Dataset(root_dir='../data/ESC-50', mode='valid', folds=[5], augment=False,
                                           out_dim=3),
                              batch_size=config["batch_size"], shuffle=False, num_workers=8)

    # 开始训练
    train_model(train_loader, valid_loader, model, criterion, optimizer, scheduler)


if __name__ == "__main__":
    device = torch.device("cuda")

    config = {
        "lr": 1e-3,
        "weight_decay": 0,
        "batch_size": 4,
        "epochs": 200,
        "t_max": 50,  # CosineAnnealing参数
        "patience": 30,  # 早停耐心值
        "model_save_path": "../saved_weights/"
    }

    main()
