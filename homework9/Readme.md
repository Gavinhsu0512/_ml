# 線性回歸模型（Linear Regression with PyTorch）

本專案使用 PyTorch 搭配人為產生的數據，實作簡單的一維線性回歸模型。  
訓練目標是逼近真實關係： `y = 2x + 3`（加入一點隨機雜訊），並使用梯度下降法找出參數 `w` 與 `b`。

---

##  專案說明

-  模型形式： `y = wx + b`
-  優化方式：使用 **均方誤差（MSE）** 作為損失函數
-  更新方式：手動計算 `.backward()` 並在 `torch.no_grad()` 下手動更新參數
-  附加：用 `matplotlib` 視覺化訓練結果

---

##  測試資料來源

使用 ChatGpt來生成資料點

X = torch.linspace(-5, 5, 100).unsqueeze(1)
Y = 2 * X + 3 + 0.5 * torch.randn(X.size())

