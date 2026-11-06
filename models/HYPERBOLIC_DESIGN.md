# Run11：一阶内容压缩与双曲跨尺度检索

## 数据流

~~~
A/B → 共享MobileNetV2 → 四级TAR T_s
                              │
                     X_s = LN(T_s)
                     ├─ guide → p_s
                     ├─ 全位置Q_s
                     └─ 区域p/(1−p)一阶池化P_s
                              ├─ K_s
                              └─ 欧式V_s
四尺度K/V联合 → 双曲或匹配欧式检索 → AV
→ AV−空间均值AV → 内容门控 → W_out → gamma残差
→ 加入TAR与细层局部高通 → DCR → 共享头
~~~

网格8/4/4/4，每格变化/背景两个原型，KV共224。完整Q共5440；1头32维。类权重w_i^+=p_i/Σp，w_i^-=(1−p_i)/Σ(1−p)。空类权重为0；支持度Σw_i m_i下限1e−6。P=ΣwX；K=L(Wqk P/√32)，V=Wv P；Q=L(Wqk X/√32)，L(z)=(√(1+||z||²),z)。K与V来自同一内容统计。

双曲距离使用曲率−1测地平方距离；匹配欧式使用同一空间坐标的欧氏平方距离。logits=−d²/(2τ²)+log(support)+scale_bias−log(M_source)。温度τ属于检索，可学习。半径asinh(||z||)保留变化。

读取R=softmax(logits)V，C=R−mean_hw(R)，U=Wout[gate·C]。gate输入本地V、C、原guide和检索到的占有率；输出门控不要求最终残差空间均值为0。T'_s=T_s+γ_s U_s+local_s，local只在细两层。

## 内部正样本加权

所有组辅助为0.1 raw-guide面积BCE。各头主损失BCEWithLogits+Dice求和。普通组BCE直接均值；加权组仅第一级使用：

w=1+0.25·y·I[GT两像素边界带之外]

BCE_weighted=Σw·BCE(logit,y)/Σw。

GT边界带为replicate填充的5×5形态学梯度。目标、Dice、其他头和所有推理操作相同。该损失用于独立诊断内部覆盖，作为训练对照评价，不作为几何机制的归因证据。

## 证据与评价

Run4/Run10冻结全量val干预支持保留径向信息和逐位置全局检索；Run10后验关闭的四库F1变化极小。下一轮以H/E/CTRL×普通/内部加权的同预算单seed矩阵分别评估几何与损失。历史对照存在辅助目标差异，冻结依赖不等同于重训收益。

训练预算、预处理、优化和test选模口径保持共同协议。验证与实现检查覆盖直接池化对冻结posterior_off的等价性、全0/1类支持、有限梯度、常量路由、空间读出、损失归一化、部署与状态恢复。
