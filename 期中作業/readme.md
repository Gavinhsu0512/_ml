#  專案簡介

-  使用 PyTorch 框架
-  使用 CNN 模型結構（兩層卷積 + ReLU + MaxPool）
-  可在低資源環境下快速訓練完成
-  資料集：MNIST 手寫數字資料集（28x28 圖片）

---

##  使用套件

pip install torch torchvision

---

## 程式內容說明

1. **資料處理**：
    - 使用 `torchvision.datasets.MNIST`
    - 僅取前 5000 筆做為訓練集（加快訓練）
2. **模型設計**：
    - 兩層卷積層（`Conv2d + ReLU + MaxPool2d`）
    - 一層全連接輸出層 `Linear(16*7*7, 10)`
3. **訓練迴圈**：
    - 使用 Adam 優化器與 CrossEntropyLoss
    - 訓練 3 輪（epoch），印出 Loss
4. **預期輸出**：
    ```bash
    Epoch 1, Loss: 0.53
    Epoch 2, Loss: 0.31
    Epoch 3, Loss: 0.22
    訓練完成：模型可辨識 0~9 手寫數字
    ```

---

##  模型架構圖

```
輸入圖像 (1 x 28 x 28)
→ Conv2d(1, 8, kernel=3) → ReLU → MaxPool2d
→ Conv2d(8, 16, kernel=3) → ReLU → MaxPool2d
→ Flatten → Linear(784, 10)
→ Softmax (隱含於 CrossEntropyLoss)
```

