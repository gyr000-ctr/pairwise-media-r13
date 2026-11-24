# brdf

## 题面（可直接粘贴）

子目录：brdf。GGX 微表面 BRDF。约束：权重能量守恒（方向积分 ≤1）；粗糙度 0 时接近镜面（半角余弦 >0.999）；Fresnel 在掠射角趋近 1；非法粗糙度可区分报错；10 万次求值不超过 300 ms；并发安全。验收：cd brdf && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd brdf && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
