\[
f(x, y, z) = x^2 + y^2 + z^2 - 2x - 4y - 6z + 8
\]

---

##  使用技術

- 使用 `Value` 類別追蹤計算圖與反向傳遞
- 手動寫出梯度下降流程（Gradient Descent）
- 每次反向傳遞後手動清空梯度 (`x.grad = 0.0` 等)

---

##  更新規則

每一輪：
```python
f = x**2 + y**2 + z**2 - 2*x - 4*y - 6*z + 8
f.backward()
x -= lr * x.grad
...


