from math import isclose

# 引入 micrograd Value 類別（請先定義在前面或另存 micrograd.py 引入）
from micrograd import Value  # 如果你已經有完整 Value 類別

# 建立變數
x = Value(0.0)
y = Value(0.0)
z = Value(0.0)

# 訓練參數
lr = 0.1
max_iter = 100

for step in range(max_iter):
    # forward: 計算損失函數
    f = x**2 + y**2 + z**2 - 2*x - 4*y - 6*z + 8

    # 清空梯度
    x.grad = y.grad = z.grad = 0.0

    # backward: 自動反向傳播
    f.backward()

    # 梯度下降更新
    x.data -= lr * x.grad
    y.data -= lr * y.grad
    z.data -= lr * z.grad

    # 印出進度
    if step % 10 == 0 or step == max_iter - 1:
        print(f"step {step:03d} | f = {f.data:.6f} | x = {x.data:.4f}, y = {y.data:.4f}, z = {z.data:.4f}")

# 最終結果
print("\n✅ 最小值結果：")
print(f"x = {x.data:.4f}, y = {y.data:.4f}, z = {z.data:.4f}, f(x,y,z) = {f.data:.6f}")
