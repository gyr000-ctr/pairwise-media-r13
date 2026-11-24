# inputbuffer

## 题面（可直接粘贴）

子目录：inputbuffer。输入缓冲与回滚预测。约束：固定窗口内输入按帧序消费；窗口外丢弃并计数；回滚后重放结果与原始一致；非法帧号可区分报错；10 万次不超过 100 ms；并发安全。验收：cd inputbuffer && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd inputbuffer && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：游戏开发
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
