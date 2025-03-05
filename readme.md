# 20250306
- toR_hat.py直接放弃，收进./deprecated文件夹里，替代为raw2Rhat.py
- raw2Rhat.py输入为3维时域data，输出为3维（numChirpsx8x8）complex64 R_hat，分解后对应float32。
  raw2Rhat不再需要另外调用函数，所有函数写在文件内部
  raw2Rhat最后带一个ifMain的示例函数，可以可视化结果
- 还没有做toR_hat与其他函数的对接，getData等文件里相关部分需要修改


# 20250302

- 基于 0228 采集数据修改了此前 LSTM 网络架构，以适配新数据。新数据未经 toR_hat.py 中的各种处理，直接喂给网络进行训练
- 加入输入和目标数据归一化操作，并保存归一化参数，供后续使用模型预测时使用
- 基于 cnn 分支经验，加入数据集拆分、早停功能并优化，防止过拟合
- 将获取训练数据、验证数据，DataLoader，早停机制等放入独立文件，通过引用方式调用
- 新增 predict 预测数据，但仅通过 AI 生成，未进行调试

# 20250123

2dFFTtest.py是实验用的，改参数可以看具体过程和图片输出
compute_background_and_subtraction.py是ai照论文写的背景消除函数
readDCA1000.py是之前转好的读bin文件
toR_hat.py最终把读到的bin文件转成论文里R^的格式，还没拆成两个实数阵
results_target1/2.xlsx是从ecg读到的心率呼吸率数据

上述文件路径，包括xlsx里的文件路径都要修改，现在只考虑用2_symmetricalPosition的数据先，整体做通了再加上非对称的，包括results_target也只有对称数据# mmwave_project

# 20250127

- 改变.bin文件夹格式，所有.bin文件放同一文件夹内，position文件名统一
- 修改toR_hat内无关代码
- 修改CBAS防止第一行为全0