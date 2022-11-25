# DRL-for-Autonomous-driving
探索深度强化学习在自动驾驶决策规划部分的使用

- 训练

`param.py`中设置`mode: str = "train"`

```shell
python demo/train.py
```

- 评估

`param.py`中设置`mode: str = "eval"`

```
# 启动envision
scl envision start -s ./scenarios -p 8081 &
# 评估
python demo/evaluate.py
```

