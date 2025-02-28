import re
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, precision_recall_curve, roc_auc_score
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import numpy as np
from torch.utils.data import Dataset
import random
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import pandas as pd
from sklearn.metrics import roc_curve, auc

matplotlib.use('TkAgg')

# 固定随机种子
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


# 定义线性层，用于嵌入向量转换
class LinearFeatureExtractor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearFeatureExtractor, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, embeddings):
        return self.linear(embeddings)

# 卷积层模块：提取隐藏特征
class ConvFeatureExtractor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvFeatureExtractor, self).__init__()
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x

# 定义PretainFeatureExtractor类
class PretainFeatureExtractor(nn.Module):
    def __init__(self, used_embedding_types, l_output_dim=128, conv_input_dim=128, conv_output_dim=10, dropout_prob=0.5):
        super(PretainFeatureExtractor, self).__init__()

        self.used_embedding_types = used_embedding_types
        self.l_output_dim = l_output_dim
        self.conv_input_dim = conv_input_dim
        self.conv_output_dim = conv_output_dim

        # 定义卷积特征提取器
        self.conv_extractor = ConvFeatureExtractor(input_dim=conv_input_dim, output_dim=conv_output_dim)

        # Dropout 层
        self.dropout = nn.Dropout(dropout_prob)

        # 定义每种嵌入类型的维度（您可以根据需要修改）
        self.embedding_dim_dict = {
            "RDKFingerprint": 2048,
            "MACCSkeys": 167,
            "EStateFingerprint": 79,
            "MolT5": 768,
            "BioT5": 768,
            "AttrMask": 300,
            "GPT-GNN": 300,
            "GraphCL": 300,
            "MolCLR": 512,
            "GraphMVP": 300,
            "GROVER": 300
        }

        # 定义嵌入处理部分，将所有嵌入映射到128维
        self.embedding_to_128_layers = nn.ModuleList()

        for embed_type in self.used_embedding_types:
            input_dim = self.embedding_dim_dict.get(embed_type, 128)  # 默认128维
            self.embedding_to_128_layers.append(LinearFeatureExtractor(input_dim, l_output_dim))

    def forward(self, data_dict):
        # 处理嵌入部分
        all_embeddings = []
        for embed_type in self.used_embedding_types:
            if embed_type in data_dict:
                fp_tensor = data_dict[embed_type]
                embedding_layer = self.embedding_to_128_layers[self.used_embedding_types.index(embed_type)]
                with torch.no_grad():
                    transformed_fp_tensor = embedding_layer(fp_tensor)
                    transformed_fp_tensor = transformed_fp_tensor.unsqueeze(1)  # 添加通道维度
                    all_embeddings.append(transformed_fp_tensor)
            else:
                print(f"Warning: Embedding type '{embed_type}' is missing in data_dict.")

        if len(all_embeddings) == 0:
            raise ValueError("No valid embeddings found in the data_dict.")

        # 合并所有嵌入
        final_embeddings = torch.cat(all_embeddings, dim=1)

        embeddings = final_embeddings.permute(0, 2, 1)  # 交换维度为 (batch_size, 128, 9)

        with torch.no_grad():
            conv_output = self.conv_extractor(embeddings)

        # 使用 Dropout 层
        conv_output = self.dropout(conv_output)  # 应用 Dropout

        # 展平卷积输出，形状应为 (batch_size, channels * length) = (48, 90)
        conv_output_flattened = conv_output.view(conv_output.size(0), -1)  # 展平为 (48, 90)

        return conv_output_flattened

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, output_dim=1, dropout_prob=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)  # 添加 Dropout 层
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)  # 使用 Dropout
        x = self.fc2(x)
        return x

class AttentionLayer(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(AttentionLayer, self).__init__()
        # 一个简单的线性层来产生注意力权重
        self.attention_fc = nn.Linear(input_dim, out_dim)

    def forward(self, attributes_tensor):
        # 计算每个公式的注意力权重
        attention_weights = self.attention_fc(attributes_tensor)
        # 使用softmax来归一化注意力权重，使得它们的总和为1
        attention_weights = torch.softmax(attention_weights, dim=1)  # 注意这里是dim=1
        return attention_weights


class FormulaLayer(nn.Module):
    def __init__(self, attributes_tuple, formula_list, formula_signs):
        super(FormulaLayer, self).__init__()
        self.attributes_tuple = attributes_tuple
        self.formula_list = formula_list
        self.formula_signs = formula_signs  # 新增：每个公式的符号信息
        self.input_dim = len(attributes_tuple)
        self.output_dim = len(formula_list)
        self.attention_weights = AttentionLayer(self.input_dim, self.output_dim)

        # 替换运算符的映射
        self.operation_map = {
            r'np\.log': 'torch.log',  # np.log -> torch.log
            r'np\.exp': 'torch.exp',  # np.exp -> torch.exp
            r'log': 'torch.log',  # log -> torch.log
            r'abs': 'torch.abs',  # abs -> torch.abs
            r'exp': 'torch.exp',  # exp -> torch.exp
            r'sqrt': 'torch.sqrt',  # sqrt -> torch.sqrt
            r'sin': 'torch.sin',  # sin -> torch.sin
            r'cos': 'torch.cos',  # cos -> torch.cos
            # 处理^运算符，注意使用小括号包裹操作数
            r'\^(-?\d+)': r'torch.pow(\g<0>, \g<1>)'  # ^n -> torch.pow(x, n)
        }

    def get_formula_indices(self, formula):
        # 匹配公式中的属性符号（字母组成的单词）
        pattern = r'\b[A-Za-z_][A-Za-z0-9_]*\b'  # 匹配标识符
        matches = re.findall(pattern, formula)

        # 对每个匹配项，查找它在attributes中的索引
        indices = [self.attributes_tuple.index(attr) for attr in matches if attr in self.attributes_tuple]
        return indices

    def convert_to_torch_expression(self, formula, attributes_tensor):
        # 替换公式中的属性为对应的tensor列索引
        for i, attr in enumerate(self.attributes_tuple):
            formula = formula.replace(attr, f"attributes_tensor[:, {i}]")

        # 替换公式中的运算符为PyTorch对应的操作
        for pattern, replacement in self.operation_map.items():
            formula = re.sub(pattern, replacement, formula)

        return formula

    def forward(self, attributes_tensor):
        # 计算每个公式的注意力权重
        attention_weights = self.attention_weights(attributes_tensor)

        # 初始化公式的输出列表
        outputs = []

        # 遍历所有公式
        for i, formula in enumerate(self.formula_list):
            # 获取公式的属性索引
            # indices = self.get_formula_indices(formula)
            # print(f"Formula: {formula}")
            # print(f"Indices: {indices}")

            # 将公式转换为torch表达式
            torch_formula = self.convert_to_torch_expression(formula, attributes_tensor)
            # print(f"Converted to Torch Expression: {torch_formula}")

            # 计算公式的输出
            try:
                result = eval(torch_formula)

                # 根据公式的符号调整结果的正负
                if self.formula_signs[i] == '-':
                    result = -result  # 如果符号是负，取负值

                outputs.append(result)
                # print(f"Formula Result: {result}\n")
            except Exception as e:
                print(f"Error in evaluating formula: {e}\n")
                # 出现错误时添加一个默认的输出（例如，返回零张量）
                outputs.append(torch.zeros_like(attributes_tensor[:, 0]))

        # 确保 outputs 中每个张量的形状与 attention_weights 匹配
        outputs_tensor = torch.stack(outputs, dim=1)  # 将输出按公式堆叠起来，形状应该是 [batch_size, output_dim]

        # attention_weights 是 [batch_size, output_dim]，与 outputs_tensor 形状相匹配
        weighted_output = outputs_tensor * attention_weights  # 每个公式的输出按注意力权重加权
        final_output = weighted_output.sum(dim=1)  # 对所有公式的加权输出求和，最终形状为 [batch_size]

        return final_output

class FGKpuuPredictor(nn.Module):
    def __init__(self, used_embedding_types, attributes_tuple, formula_list, formula_signs,
                 l_output_dim=128, conv_input_dim=128, conv_output_dim=10,
                 feature_dim=18, mlp_hidden_dim=32, mlp_output_dim=1, dropout_prob=0.5):
        super(FGKpuuPredictor, self).__init__()
        self.pretrain_emb_layer = PretainFeatureExtractor(used_embedding_types, l_output_dim=l_output_dim,
                                conv_input_dim=conv_input_dim, conv_output_dim=conv_output_dim, dropout_prob=dropout_prob)
        mlp_input_dim = len(used_embedding_types) * conv_output_dim + feature_dim
        self.mlp_layer = MLP(mlp_input_dim, mlp_hidden_dim, mlp_output_dim, dropout_prob)
        self.formula_layer = FormulaLayer(attributes_tuple, formula_list, formula_signs)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 学习加权系数

    def forward(self, ori_embeddings_dict, attributes_tensor):
        prtrain_embedding = self.pretrain_emb_layer(ori_embeddings_dict)
        attributes_refined_embedding = torch.cat((prtrain_embedding, attributes_tensor), dim=1)
        pred_kpuu = self.mlp_layer(attributes_refined_embedding)
        weight_formula = self.formula_layer(attributes_tensor)
        pred_kpuu = self.alpha * pred_kpuu.squeeze() + (1 - self.alpha) * weight_formula
        return pred_kpuu, weight_formula


class FormulaGuideRegressionLoss(nn.Module):
    def __init__(self, alpha=0.1, lambda_init=0.1, min_norm_value=1e-6, max_lambda=0.5):
        super(FormulaGuideRegressionLoss, self).__init__()
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init))  # 初始化lambda为可学习的参数
        self.min_norm_value = min_norm_value  # 设置L2范数最小值，防止过小
        self.max_lambda = max_lambda  # 设置lambda的最大值，防止过大

    def forward(self, predictions, targets, formula_outputs):
        # 计算主损失函数（MSELoss），通常不会是负数
        mse_loss = nn.MSELoss()(predictions, targets)

        # 计算guide_loss作为正则化项，使用L2范数
        guide_loss = torch.norm(formula_outputs, p=2)

        # 确保L2范数不会过小，加入min_norm_value来调整
        guide_loss = torch.max(guide_loss, torch.tensor(self.min_norm_value, device=formula_outputs.device))

        # 使用sigmoid使lambda_param的值保持在[0, 1]区间
        lambda_scaled = torch.sigmoid(self.lambda_param)  # 通过sigmoid将lambda参数限制在[0, 1]区间

        # 总损失：包括主损失和正则化项，lambda_scaled控制正则化的强度
        total_loss = mse_loss + lambda_scaled * guide_loss

        return total_loss



# 提取特定的属性
def extract_selected_attributes(data_dict, attributes):
    attributes_list = []
    for idx, sample in data_dict.items():
        selected_values = [sample[attr] for attr in attributes if attr in sample]
        attributes_list.append(selected_values)
    return torch.tensor(attributes_list, dtype=torch.float32)

# 提取目标值
def extract_target_values(data_dict, target_name):
    targets = []
    for idx, sample in data_dict.items():
        target_value = sample.get(target_name, None)
        if target_value is not None:
            targets.append(target_value)
    return torch.tensor(targets, dtype=torch.float32)

# 提取特定的嵌入类型
def extract_selected_embeding(data_dict, used_embedding_types):
    # 创建一个字典，用于保存每个嵌入类型对应的所有数据
    embeddings_dict = {embed_type: [] for embed_type in used_embedding_types}

    # 遍历数据
    for idx, sample in data_dict.items():
        for embed_type in used_embedding_types:
            if embed_type in sample:  # 检查 sample 中是否有该嵌入类型
                fp_tensor = sample[embed_type]  # 获取对应的嵌入类型数据
                embeddings_dict[embed_type].append(fp_tensor)  # 将该嵌入数据添加到字典中

    # 将每种嵌入类型的 list 转换为 tensor，去除大小为 1 的维度
    for embed_type in embeddings_dict:
        embeddings_dict[embed_type] = torch.stack(embeddings_dict[embed_type])  # 转换为 tensor
        embeddings_dict[embed_type] = embeddings_dict[embed_type].squeeze(1)  # 去掉大小为 1 的维度

    return embeddings_dict  # 返回包含嵌入类型的字典


# 定义自定义数据集类
class pretrainDataset(Dataset):
    def __init__(self, embeddings_dict, attributes, targets, used_embedding_types):
        self.embeddings_dict = embeddings_dict
        self.attributes = attributes
        self.targets = targets
        self.used_embedding_types = used_embedding_types
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, item):
        embeddings = {embed_type: self.embeddings_dict[embed_type][item] for embed_type in self.used_embedding_types}
        attribute = self.attributes[item]
        target = self.targets[item]

        return embeddings, attribute, target
    @staticmethod
    def collate_fn(batch):
        """
        批量合并函数，将所有样本合并为一个批次
        """
        # 初始化用于存储嵌入、属性和目标的容器
        embeddings_batch = {key: [] for key in batch[0][0].keys()}  # 初始化字典用于存储每种嵌入类型的样本
        attributes_batch = []
        targets_batch = []

        # 遍历所有样本，合并数据
        for sample in batch:
            embeddings, attribute, target = sample
            for key in embeddings:
                embeddings_batch[key].append(embeddings[key])  # 将嵌入数据添加到对应的嵌入类型列表
            attributes_batch.append(attribute)
            targets_batch.append(target)

        # 将每种嵌入类型的样本列表转换为张量
        for key in embeddings_batch:
            embeddings_batch[key] = torch.stack(embeddings_batch[key], dim=0)  # 转换为张量

        # 将属性和目标值转换为张量
        attributes_batch = torch.stack(attributes_batch, dim=0)
        targets_batch = torch.stack(targets_batch, dim=0)

        return embeddings_batch, attributes_batch, targets_batch


class EarlyStopping:
    def __init__(self, patience=20, verbose=False, delta=0, metric_type='RMSE'):
        """
        :param patience: 容忍的没有改善的 epochs 数量
        :param verbose: 是否打印早停信息
        :param delta: 只有当损失改善超过 delta 时，才认为是有效改善
        :param metric_type: 用于判断的评价指标，'RMSE' 或 'R2'
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.metric_type = metric_type
        self.counter = 0
        self.best_metric = float('inf') if metric_type == 'RMSE' else -float('inf')  # 根据评价指标初始化
        self.best_weights = None

    def step(self, val_metric, model):
        """
        :param val_metric: 当前验证集的指标（RMSE 或 R²）
        :param model: 当前模型
        :return: 是否触发早停
        """
        if self.metric_type == 'RMSE':
            # 如果当前 RMSE 小于历史最佳 RMSE，则更新最佳 RMSE
            if val_metric < self.best_metric - self.delta:
                self.best_metric = val_metric
                self.best_weights = model.state_dict()
                self.counter = 0
            else:
                self.counter += 1
        elif self.metric_type == 'R2':
            # 如果当前 R² 大于历史最佳 R²，则更新最佳 R²
            if val_metric > self.best_metric + self.delta:
                self.best_metric = val_metric
                self.best_weights = model.state_dict()
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            if self.verbose:
                print(f"Early stopping triggered after {self.counter} epochs without improvement.")
            return True
        return False

    def restore_best_weights(self, model):
        """ 恢复最佳的模型权重 """
        if self.best_weights:
            model.load_state_dict(self.best_weights)

def train_model(model, criterion, optimizer, scheduler, train_dataloader,
                val_dataloader=None, epochs=10, patience=20, use_early_stopping=True):
    best_r2 = -float('inf')  # 用来保存当前最佳的 R² 值
    early_stopping = None
    if use_early_stopping:
        # 初始化早停法
        early_stopping = EarlyStopping(patience=patience, verbose=True, metric_type='RMSE')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            embeddings_batch, attributes_batch, targets_batch = batch
            embeddings_batch = {key: val.to(device) for key, val in embeddings_batch.items()}
            attributes_batch = attributes_batch.to(device)
            targets_batch = targets_batch.to(device)

            optimizer.zero_grad()  # 清空之前的梯度
            predictions, formula_outputs = model(embeddings_batch, attributes_batch)
            loss = criterion(predictions, targets_batch, formula_outputs)
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新权重

            running_loss += loss.item()

        if scheduler:
            # 更新学习率
            scheduler.step()
            # 每个epoch结束后输出当前学习率
            current_lr = scheduler.get_last_lr()[0]
            print(f"Current Learning Rate: {current_lr:.6f}")

        avg_train_loss = running_loss / len(train_dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}")

        # 验证过程
        if val_dataloader:
            val_loss, val_rmse, val_r2 = evaluate_model(model, val_dataloader, criterion, device)
            print(f"Validation Loss: {val_loss:.4f}, RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")

            # 如果当前 R² 更高，则保存模型
            if val_r2 > best_r2:
                best_r2 = val_r2
                print("Saving model with highest R²...")
                torch.save(model.state_dict(), 'best_model.pth')

            # 早停法：检查当前 R² 是否比历史最高的 R² 更高
            if use_early_stopping and early_stopping.step(val_rmse, model):
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break

        # # 保存当前模型
        # model_filename = f"./models/model_epoch_{epoch + 1}.pth"
        # torch.save(model.state_dict(), model_filename)
        # print(f"Model for epoch {epoch + 1} saved as {model_filename}")

    # 恢复最佳模型权重
    if early_stopping:
        early_stopping.restore_best_weights(model)


def evaluate_model(model, val_dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    true_values = []
    predicted_values = []

    with torch.no_grad():
        for batch in val_dataloader:
            embeddings_batch, attributes_batch, targets_batch = batch
            embeddings_batch = {key: val.to(device) for key, val in embeddings_batch.items()}
            attributes_batch = attributes_batch.to(device)
            targets_batch = targets_batch.to(device)

            predictions, formula_outputs = model(embeddings_batch, attributes_batch)
            loss = criterion(predictions , targets_batch, formula_outputs)
            running_loss += loss.item()

            # 收集预测值和真实值
            true_values.append(targets_batch.cpu().numpy())
            predicted_values.append(predictions.squeeze().cpu().numpy())

    # 合并所有批次的真实值和预测值
    true_values = np.concatenate(true_values, axis=0)
    predicted_values = np.concatenate(predicted_values, axis=0)

    # 计算 RMSE 和 R²
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    pearson_corr, _ = pearsonr(true_values, predicted_values)
    R2 = pearson_corr ** 2

    # 打印指标
    print(f"Validation Loss: {running_loss / len(val_dataloader):.4f}")
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation R²: {R2:.4f}")

    return running_loss / len(val_dataloader), rmse, R2

# 定义测试过程
def test_model(model, test_dataloader):
    model.eval()
    true_values = []
    predicted_values = []

    with torch.no_grad():
        for batch in test_dataloader:
            embeddings_batch, attributes_batch, targets_batch = batch
            embeddings_batch = {key: val.to(device) for key, val in embeddings_batch.items()}
            attributes_batch = attributes_batch.to(device)
            targets_batch = targets_batch.to(device)

            predictions, formula_outputs = model(embeddings_batch, attributes_batch)
            true_values.append(targets_batch.cpu().numpy())
            predicted_values.append(predictions.cpu().numpy())

    true_values = np.concatenate(true_values, axis=0)
    predicted_values = np.concatenate(predicted_values, axis=0)

    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    pearson_corr, _ = pearsonr(true_values, predicted_values)
    R2 = pearson_corr ** 2

    print("*" * 50)
    print(f"Test RMSE: {rmse:.3f}")
    print(f"Test R²: {R2:.3f}")
    try:
        _, AUC = calculate_auc(true_values, predicted_values)
        print(f"Test AUC: {AUC:.3f}")
    except:
        pass
    return true_values, predicted_values

# 计算精确度、召回率和F1分数的函数
def calculate_auc(y_true, y_pred):
    """
    计算精确度、召回率和F1分数
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :return: 最优阈值、AUC值
    """
    y_true_binary_train = (y_true > 0.3).astype(int)
    precision, recall, thresholds = precision_recall_curve(y_true_binary_train, y_pred)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_threshold = thresholds[np.argmax(f1_scores)]
    auc = roc_auc_score(y_true_binary_train, y_pred)
    return best_threshold, auc

def create_dataloaders(data_dict, used_embedding_types, attributes, traget='Kpuu', batch_size=16, shuffle=True):
    """
    该函数用于根据输入的训练集和测试集数据，以及使用的嵌入类型和特征，生成相应的 DataLoader。

    参数：
    - train_data: 训练集数据
    - test_data: 测试集数据
    - used_embedding_types: 使用的嵌入类型
    - attributes: 特征列表
    - batch_size: 批大小，默认为16

    返回：
    - train_dataloader: 训练集 DataLoader
    - test_dataloader: 测试集 DataLoader
    """

    # 提取训练集和测试集的特征
    attributes = extract_selected_attributes(data_dict, attributes)
    embeddings_dict = extract_selected_embeding(data_dict, used_embedding_types)
    targets = extract_target_values(data_dict, traget)

    # 定义训练和测试数据集
    dataset = pretrainDataset(
        embeddings_dict=embeddings_dict,
        attributes=attributes,
        targets=targets,
        used_embedding_types=used_embedding_types
    )

    # 定义DataLoader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pretrainDataset.collate_fn
    )

    return dataloader


def plot_roc_curve(true_values, predicted_values):

    # 判断 true_values 是否为二进制（0 或 1），如果不是则转换
    if not all((true_values == 0) | (true_values == 1)):  # 如果 true_values 中有非 0 或 1 的值
        true_values = (true_values > 0.3).astype(int)  # 转换为二进制，0.3 为阈值
    # 设置seaborn的绘图风格为darkgrid
    sns.set(style="darkgrid")
    # 设置字体加粗
    matplotlib.rcParams['font.weight'] = 'bold'
    matplotlib.rcParams['axes.labelweight'] = 'bold'
    matplotlib.rcParams['axes.titlesize'] = 18  # 设置标题大小
    matplotlib.rcParams['axes.titleweight'] = 'bold'  # 设置标题加粗
    # 设置字体为Arial
    matplotlib.rcParams['font.family'] = 'Arial'

    # 计算 ROC 曲线的假阳性率和真阳性率
    fpr, tpr, thresholds = roc_curve(true_values, predicted_values)

    # 计算 AUC (Area Under Curve)
    roc_auc = auc(fpr, tpr)

    # 创建绘图
    plt.figure(figsize=(6, 5))

    # 绘制 ROC 曲线
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.3f})')

    # 绘制对角线，表示随机猜测
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

    # 添加图形标题和标签
    # plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')

    # 显示图例
    plt.legend(loc='lower right')

    # 显示 ROC 曲线
    plt.show()

    # 输出 AUC 值
    print(f"AUC: {roc_auc:.3f}")

# 主程序
if __name__ == "__main__":
    set_random_seed(2024)
    # 读取训练集和测试集数据
    with open('train_data_cleaned.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('test_data_cleaned.pkl', 'rb') as f:
        test_data = pickle.load(f)
    with open('test_data_friden.pkl', 'rb') as f:
        friden_data = pickle.load(f)
    with open('test_data_Schrodinger.pkl', 'rb') as f:
        Schrodinger_data = pickle.load(f)

    # 定义要使用的嵌入类型
    used_embedding_types = (
        'RDKFingerprint', 'MACCSkeys', 'EStateFingerprint', 'MolT5', 'BioT5', 'AttrMask', 'GPT-GNN', 'GraphCL', 'MolCLR',
        'GraphMVP', 'GROVER',
    )

    # 属性名称列表
    attributes = (
        'PSA', 'ALIE_Ave', 'ALIEmax', 'Pi', 'MPI', 'Nonpolar_Area', 'Polar_Area', 'LEAmin', 'LEA_Ave', 'ESPmin',
        'ESPmax', 'Quadrupole_Moment', 'Density', 'Overall_Variance', 'Weight', 'HOMO_number', 'LEAmax', 'HOMO_LUMO_Gap'
    )

    formula_list = [
        '(abs((PSA+LEA_Ave)-(Pi/Overall_Variance))/(abs(PSA-Nonpolar_Area)+Nonpolar_Area))',
        '(abs((Pi/Overall_Variance)-PSA)/((Density-Nonpolar_Area)-abs(PSA-Nonpolar_Area)))',
        '((Polar_Area/LEAmin)/abs((PSA*Overall_Variance)-(Pi*Density)))'
    ]

    # 每个公式的相关性符号 (+ 或 -)
    formula_signs = ['-', '+', '-']  # 注意：此处需要为每个公式添加符号


    train_dataloader = create_dataloaders(train_data, used_embedding_types, attributes,batch_size=16, shuffle=True)
    test_dataloader = create_dataloaders(test_data, used_embedding_types, attributes, batch_size=len(test_data), shuffle=False)
    friden_dataloader = create_dataloaders(friden_data, used_embedding_types, attributes, batch_size=len(friden_data), shuffle=False)
    Schrodinger_dataloader = create_dataloaders(Schrodinger_data, used_embedding_types, attributes, batch_size=len(Schrodinger_data), shuffle=False)

    train = False
    # 初始化模型
    model = FGKpuuPredictor(used_embedding_types, attributes, formula_list, formula_signs,
                            l_output_dim=32, conv_input_dim=32, conv_output_dim=8,
                            feature_dim=18, mlp_hidden_dim=64, mlp_output_dim=1)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if train:
        # 定义损失函数和优化器
        criterion = FormulaGuideRegressionLoss()
        optimizer = optim.Adam(list(model.parameters()) + [criterion.lambda_param], lr=0.0001)
        # 定义余弦退火学习率调度器
        scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)
        # scheduler = None

        train_model(model, criterion, optimizer, scheduler, train_dataloader,
                    test_dataloader, epochs=1000, use_early_stopping=False)

    # 开始测试
    model.load_state_dict(torch.load('./best_model_2.pth'))
    test_model(model, test_dataloader)
    print("<chembl2023>")
    test_model(model, friden_dataloader)
    print("<friden>")
    true_values, predicted_values = test_model(model, Schrodinger_dataloader)
    print("<Schrodinger>")
    pearson_corr, _ = pearsonr(true_values, predicted_values)
    print(pearson_corr)

    # # 创建一个包含 'True Values' 和 'Predicted Values' 的 DataFrame
    # df = pd.DataFrame({
    #     'True Values': true_values,
    #     'Predicted Values': predicted_values
    # })
    #
    # # 保存到 CSV 文件中
    # df.to_csv('test_values.csv', index=False)
    #
    # print("True and Predicted values have been saved to 'true_predicted_values.csv'")

    # # 调用函数绘制 ROC 曲线
    # plot_roc_curve(true_values, predicted_values)

