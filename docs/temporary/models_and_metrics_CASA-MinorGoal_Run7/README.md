# CASA-MinorGoal / Run7

双曲空间内的原型压缩：把类条件池化从「先在欧式空间求均值、再提升到 Lorentz」改为「先把成员像素提升到 Lorentz，再在流形上求加权重心（Klein 重心）」，K 与 V 都走这条路；同时删去 TAR 中 `β·D_s` 的可学习缩放。

实验顺序：SYSU → LEVIR，仅用 GPU1；另做 2×2 消融（无/有 TAR-CG × 无/有 HRA）。
