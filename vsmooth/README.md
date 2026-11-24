# vsmooth

## 题面（可直接粘贴）

子目录：vsmooth。Laplacian 网格平滑（体积保持）。约束：平滑后体积变化 ≤1%（Taubin μ|λ 交替）；顶点数不变；单步位移不超过给定上界；非法迭代数可区分报错；1 万顶点不超过 500 ms；并发安全。验收：cd vsmooth && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd vsmooth && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
