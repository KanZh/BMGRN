import sys
import os

# =======================================================
# 1. 解决路径和 "No module named 'duplex'" 报错的神奇代码
# =======================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
# 把 code/ 和 code/train_edge/ 都加入系统路径
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'train_edge'))

try:
    import mylogging
    import types

    # 动态伪造一个 'duplex' 包，骗过 Python 的 import 检查
    duplex_module = types.ModuleType('duplex')
    duplex_module.mylogging = mylogging
    sys.modules['duplex'] = duplex_module
    sys.modules['duplex.mylogging'] = mylogging
except Exception as e:
    print(f"路径配置警告: {e}")

# =======================================================
# 2. 正常导入其他库
# =======================================================
import pandas as pd
import numpy as np
import torch
import dgl
import train_edge.data_preprocessing as dp  # 路径加好后，这里可以直接引用
from train_edge.config import const_args
import argparse

# 包装参数
args = argparse.Namespace(**const_args)


def main():
    # --- 修正后的绝对路径 ---
    csv_path = r"D:\project\XATGRN-main\Prediction\Ecoli_cold_FGRN\traindata\Ecoli_GRN_3types.csv"
    tsv_path = r"D:\project\XATGRN-main\Prediction\Ecoli_cold_FGRN\traindata\Ecoli_GRN_3types_ids.tsv"

    # 读取原始数据
    df_csv = pd.read_csv(csv_path, header=None, engine='python')
    # TSV 文件通常是用制表符或空格分隔，这里使用正则自适应分隔符
    df_tsv = pd.read_csv(tsv_path, sep=r'\s+|,|\t', header=None, engine='python')

    print(f"原始 CSV 边数: {len(df_csv)}")
    print(f"原始 TSV 边数: {len(df_tsv)}")
    assert len(df_csv) == len(df_tsv), "CSV 和 TSV 行数不一致，请检查数据！"

    # 获取原始图的总节点数 (防止因为丢弃边导致图的维度坍缩)
    original_graph = dgl.load_graphs('./edge_data/ecoli/whole.graph')[0][0]
    num_nodes = original_graph.num_nodes()
    print(f"原始网络总节点数: {num_nodes}")

    # 需要丢弃边的比例 (Ablation study)
    drop_rates = [10, 20, 30, 40, 50]

    for rate in drop_rates:
        print(f"\n==================================================")
        print(f"🚀 正在生成缺失 {rate}% 边的实验数据...")

        # 计算保留比例并随机抽样 (random_state=42 保证可复现)
        frac_to_keep = 1.0 - (rate / 100.0)
        kept_indices = df_csv.sample(frac=frac_to_keep, random_state=42).index

        # 分别对 CSV 和 TSV 获取保留下来的同一批行
        df_csv_drop = df_csv.loc[kept_indices]
        df_tsv_drop = df_tsv.loc[kept_indices]

        # --- A. 保存给主模型 (第一段代码) 用的 CSV ---
        new_csv_path = csv_path.replace('.csv', f'_drop{rate}.csv')
        df_csv_drop.to_csv(new_csv_path, index=False, header=False)
        print(f"已保存缺失 {rate}% 的分类器先验边至: {new_csv_path}")

        # 3. 保存路径名称改成 ecoli_drop
        dataset_name = f'ecoli_drop{rate}'
        save_path = f'./edge_data/{dataset_name}/'
        os.makedirs(save_path, exist_ok=True)

        # 提取源节点和目标节点构建新的 DGL 图
        src = torch.tensor(df_tsv_drop.iloc[:, 0].values, dtype=torch.int64)
        dst = torch.tensor(df_tsv_drop.iloc[:, 1].values, dtype=torch.int64)

        # 构建新图 (必须传入相同的 num_nodes)
        new_graph = dgl.graph((src, dst), num_nodes=num_nodes)

        # 集合去重逻辑
        src_new, dst_new = new_graph.edges()
        unique_edges = set(zip(src_new.tolist(), dst_new.tolist()))
        src_final, dst_final = zip(*unique_edges)
        deduplicated_graph = dgl.graph((src_final, dst_final), num_nodes=num_nodes)

        # 保存新的 whole.graph
        dgl.data.utils.save_graphs(save_path + 'whole.graph', [deduplicated_graph])

        print(f"开始为 {dataset_name} 生成 0-9 折交叉验证目录 (这可能需要几秒钟)...")
        # 按照原来的逻辑生成 task 1,2,3,4 和 seed 0-9
        for task in [1, 2, 3, 4]:
            for seed in range(10):
                # 调用 data_preprocessing.py 里的切分函数
                dp.split_data(args, deduplicated_graph, save_path, seed, task)

        print(f"✅ {dataset_name} 生成完毕！")


if __name__ == "__main__":
    main()