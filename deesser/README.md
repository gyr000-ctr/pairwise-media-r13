# deesser

## 题面（可直接粘贴）

子目录：deesser。齿音抑制（分频段动态压缩）。约束：齿音频段（4–9 kHz）增益下降 ≥6 dB 而低频段增益变化 ≤0.5 dB；检测用频段能量且带迟滞；跨缓冲连续无爆音；100 万样本不超过 600 ms；非法参数可区分报错；并发安全。验收：cd deesser && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd deesser && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：音视频处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
