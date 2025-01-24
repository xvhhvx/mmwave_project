# pack1说明

20250123

2dFFTtest.py是实验用的，改参数可以看具体过程和图片输出
compute_background_and_subtraction.py是ai照论文写的背景消除函数
readDCA1000.py是之前转好的读bin文件
toR_hat.py最终把读到的bin文件转成论文里R^的格式，还没拆成两个实数阵
results_target1/2.xlsx是从ecg读到的心率呼吸率数据

上述文件路径，包括xlsx里的文件路径都要修改，现在只考虑用2_symmetricalPosition的数据先，整体做通了再加上非对称的，包括results_target也只有对称数据# mmwave_project
