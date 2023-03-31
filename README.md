# lhmind

lhmind是lhhc的算法工具，其中集成了数据预处理，模型训练，模型预测的相关方法。

项目目录树形图如下：
```
lhmind/
├── lhmind/
│   ├── models/
│   │   ├── base.py
│   │   ├── logistic_regression.py
│   │   ├── random_forest.py
│   │   └── svm.py
│   ├── constants.py
│   ├── data_processing.py
│   ├── predict.py
│   ├── train.py
│   └── utils.py
├── test/
│   ├── reference.csv
│   ├── test.csv
│   └── test.py
│── .gitignore
│── README.md
└── setup.py
```

## lhmind/constants.py

本文件包含一些常量，用于表示血液检测指标及其对应的中文名称。

### 变量

#### blood_test_all_indicators：血液检测所有指标字典，包含血常规和血生化指标。

#### basic_information：基本信息字典，如年龄。

#### blood_routine_indicators：血常规指标字典。

#### blood_biochemical_indicators：血生化指标字典。

### 示例

```python
from lhmind.constants import blood_routine_indicators

print(blood_routine_indicators['RBC'])  # 输出：红细胞计数
```
此示例展示了如何从blood_routine_indicators字典中获取某个指标的中文名称。

## lhmind/data_processing.py

本文件包含数据预处理功能。

### 函数列表

1. `validate_and_filter_data(data, train_type)`: 验证和过滤数据
2. `normalize_data(data, reference_values)`: 标准化数据
3. `filter_and_map_data(data, target_col)`: 过滤和映射数据
4. `preprocess_data(data, train_type, reference_values)`: 数据预处理
5. `preprocess_data_for_prediction(data, reference_values, median_values_dict, feature_names)`: 预测数据预处理

### 函数详细说明

#### validate_and_filter_data(data, train_type)

- 输入：数据（DataFrame），训练类型（字符串）
- 输出：过滤后的数据（DataFrame）
- 功能：根据训练类型验证和过滤输入数据

#### normalize_data(data, reference_values)

- 输入：数据（DataFrame），参考值（DataFrame）
- 输出：标准化后的数据（DataFrame），标签为0的分组的中位数值字典（字典）
- 功能：根据参考值对数据进行标准化处理

在normalize_data函数中，补充缺失值的逻辑是根据性别、疾病和参考值来进行的。下面详细介绍这个过程：
1. 首先，根据性别（男或女）和疾病类型对数据进行分组。
2. 对于每个特征，根据当前特征和性别从参考值表中获取最小值（min_val）和最大值（max_val）。
3. 对于每个分组，检查该分组中是否至少有一个非空值。如果有至少一个非空值，计算该分组的特征的中位数（median_val）；否则，使用最小值和最大值的平均值作为中位数（median_val）。
4. 使用计算出的中位数（median_val）填充每个分组中特征的缺失值。
5. 最后，使用最小值（min_val）和最大值（max_val）对每个分组的特征数据进行归一化处理。
这个逻辑可以确保根据性别和疾病类型对缺失值进行合理的补充，并且对数据进行归一化处理以消除特征之间的量级差异。

#### filter_and_map_data(data, target_col)

- 输入：数据（DataFrame），目标列名称（字符串）
- 输出：过滤和映射后的数据（DataFrame）
- 功能：根据目标列名称过滤和映射输入数据

#### preprocess_data(data, train_type, reference_values)

- 输入：数据（DataFrame），训练类型（字符串），参考值（DataFrame）
- 输出：预处理后的数据（元组），标签为0的分组的中位数值字典（字典）
- 功能：对输入数据进行预处理（包括验证、过滤、标准化等）

#### preprocess_data_for_prediction(data, reference_values, median_values_dict, feature_names)

- 输入：数据（DataFrame），参考值（DataFrame），中位数值字典（字典），特征名称列表（列表）
- 输出：预处理后的预测数据（DataFrame）
- 功能：对输入数据进行预处理，用于模型预测




## lhmind/train.py

本文件包含训练模型的功能。

### 函数列表

1. `train_model(params)`: 训练模型

### 函数详细说明

#### train_model(params)

- 输入：参数字典（字典）
- 输出：模型评估结果（字典）
- 功能：根据输入的参数训练模型，并将训练好的模型保存到指定路径

参数字典包括以下键值对：

- `hospital_phone`（字符串）：医院电话
- `train_type`（字符串）：训练类型
- `disease_type`（字符串）：疾病类型
- `blood_test_data`（DataFrame）：血液检测数据
- `model_file_storage_path`（字符串）：模型文件存储路径
- `model_name`（字符串）：模型名称，可以是 "logistic_regression"、"random_forest" 或 "svm"
- `reference_values`（DataFrame）：参考值

训练过程中，会先对血液检测数据进行预处理，然后根据指定的模型名称创建对应的模型实例并进行训练。训练完成后，评估模型性能，创建相应的目录结构并将训练好的模型保存到指定路径。

## lhmind/predict.py

本文件包含预测功能，通过加载已经训练好的模型对输入的血液检测数据进行预测。

### predict 函数

`predict` 函数接收以下参数：

- hospital_phone：医院的电话号码
- train_type：训练类型，如 'normal_cancer'、'normal_inflam' 等
- disease_type：疾病类型
- blood_test_data：血液检测数据
- model_file_storage_path：模型文件存储路径
- model_name：模型名称，如 'logistic_regression'、'random_forest'、'svm' 等
- reference_values：参考值

根据提供的参数执行以下操作：
1. 实例化相应的模型。
2. 根据给定的医院电话号码、训练类型和疾病类型创建模型文件的目录结构。
3. 加载训练好的模型。
4. 加载训练信息，包括特征名称和中位数值字典。
5. 对血液检测数据进行预处理，包括填充缺失值和数据归一化。
6. 使用训练好的模型对预处理后的血液检测数据进行预测。
7. 返回预测结果。

## lhmind/base_model.py

本文件包含基础模型类，用于实现各种模型的共同功能。

### BaseModel 类

`BaseModel` 类实现了各种模型的共同功能，可以继承此类来创建不同类型的模型。它包括以下方法：

- fit(self, X, y)：用训练数据拟合模型。

- predict(self, X)：对给定的数据进行预测。

- score(self, X)：返回预测类别为 1 的概率值。

- evaluate(self, X, y)：评估模型在给定数据上的性能，返回分类报告。

- save_training_information(self, model_file_path, train_type, training_results)：保存训练信息到 JSON 文件。

- save_model(self, model_file_path, evaluation_result, feature_names, median_values_dict)：保存训练好的模型到文件，并更新训练信息。

- load_model(self, model_file_path)：从文件加载训练好的模型。

- load_training_info(self, model_file_path)：从文件加载训练信息。

### 创建新算法模型示例

```python
from sklearn.linear_model import LogisticRegression
from lhmind.base_model import BaseModel

class LogisticRegressionModel(BaseModel):
    def __init__(self):
        super().__init__(LogisticRegression())
```
这个示例展示了如何使用 BaseModel 类创建一个逻辑回归模型。只需将逻辑回归模型传递给 BaseModel 构造函数，然后就可以使用 BaseModel 中的方法来训练、评估和保存模型。