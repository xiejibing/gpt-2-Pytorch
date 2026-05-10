import numpy as np

import matplotlib.pyplot as plt 
from matplotlib.axes import Axes

def cretae_rotate_table(max_seq_len,  head_dim):
    # 1. 计算频率 theta
    # 公式: theta_i = 10000 ^ (-2i / d)
    # np.arange(0, head_dim, 2): 只取偶数维 (0, 2, 4...)，因为 RoPE 是两两配对的
    theta = 10000 ** (-np.arange(0, head_dim, 2) / head_dim)

    # 2. 扩展theta
    # 因为 theta_0 控制第0维和第1维，所以需要把每个频率复制两份
    # [theta_0, theta_1] -> [theta_0, theta_0, theta_1, theta_1]
    theta = theta.reshape(-1, 1).repeat(2, axis=1).flatten()
    
    # 3.生成位置矩阵
    pos = np.arange(0, max_seq_len)

    # 4. m * theta
    table = pos.reshape(-1, 1) @ theta.reshape(1, -1)

    # 5. cos(m * theta)
    cos_table = np.cos(table)

    # 6.  # 【核心 Trick】: 这里把偶数维的 sin 取负号
    # 原理: 旋转公式是 [x*cos - y*sin, x*sin + y*cos]
    # 我们希望统一写成加法形式: a + b
    # 所以构造 sin_cache 为 [-sin, sin, -sin, sin ...]
    # 后面配合 rotate_half 得到的 [y, x]，就能凑出 (-y*sin) 和 (x*sin)
    sin_theta = np.sin(table)
    sin_theta[:, ::2] = -sin_theta[:, ::2]

    return sin_theta, cos_table 


def rotate(x, m, sin_table, cos_table):
    # 对应公式: 
    # x' = x * cos - y * sin
    # y' = y * cos + x * sin
    #
    # 代码实现:
    # 第一项: vec * cos -> [x*cos, y*cos]
    # 第二项: rotate_half(vec) * sin -> [y, x] * [-sin, sin] = [-y*sin, x*sin]
    #
    # 相加:
    # 偶数位: x*cos + (-y*sin) = x*cos - y*sin
    # 奇数位: y*cos + (x*sin)  = y*cos + x*sin

    return x * cos_table[m] + rotate_half(x) * sin_table[m]


def rotate_half(x):
    
    """
    交换向量中两两配对的元素: [x, y, a, b] -> [y, x, b, a]
    这是为了配合旋转公式中的交叉项
    """
    # 1. reshape(-1, 2): 把向量变成 [[x, y], [a, b], ...]
    # 2. [:, ::-1]: 把每一行的元素翻转 -> [[y, x], [b, a], ...]
    # 3. flatten(): 展平回一维 -> [y, x, b, a]

    return x.reshape(-1, 2)[:, ::-1].flatten()



def plot(plt_obj: Axes, pic_index, query_index=0, head_dim=256, max_num_tokens=8192, step=1):
    # 1. 初始化 Q 和 K 为全 1 向量
    # 目的: 剥离向量内容的语义影响，纯粹观察位置编码带来的几何性质
    q_vec = np.ones(head_dim)
    k_vec = np.ones(head_dim)
    
    # 2. 生成 sin/cos 缓存表
    sin_table, cos_table = cretae_rotate_table(max_num_tokens, head_dim)

    # 3. 旋转 Query (固定在 query_index 位置)
    rotated_q_vec = rotate(q_vec, query_index, sin_table, cos_table)
    
    # 4. 旋转 Key (遍历整个序列长度)
    k_indices = np.arange(0, max_num_tokens, step)
    # 注意: 这里 rotary 支持广播，一次性算出所有位置的旋转后向量
    rotated_k_vecs = rotate(k_vec, k_indices, sin_table, cos_table)
    
    # 5. 计算 Attention Score (点积)
    # (Keys @ Query) / sqrt(d)
    attn_scores = (rotated_k_vecs @ rotated_q_vec) / np.sqrt(head_dim)

    # 6. 绘图
    plt_obj.plot(k_indices, attn_scores)
    plt_obj.set_title(f"Figure {pic_index}: query_index={query_index}, head_dim={head_dim}")
    plt_obj.set_xlabel("key index")
    plt_obj.set_ylabel("attention score")

# --- 画图配置 ---
plt.rcParams.update({
    "font.sans-serif": ["Times New Roman", ], # 设置字体
    "font.size": 10
})

# 创建 2x2 的画布
_, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

# 图1: 短距离衰减 (Q在开头)
# 展示: 随着 Key 远离 Query，分数震荡下降 (远程衰减)
plot(axes[0, 0], 1, query_index=0, max_num_tokens=512)

# 图2: 相对位置对称性 (Q在中间)
# 展示: 左右两侧是对称的，说明 RoPE 关注的是相对距离 |m-n|
plot(axes[0, 1], 2, query_index=256, max_num_tokens=512)

# 图3: 超长上下文 (65k)
# 展示: 在极长距离下，衰减依然有效，且保留了微弱的震荡感知
plot(axes[1, 0], 3, query_index=0, max_num_tokens=65535)

# 图4: 维度过小的反面教材 (Head Dim=8)
# 展示: 维度太小会导致“时钟”转太快，发生周期重叠，长距离外推能力变差
plot(axes[1, 1], 4, query_index=0, head_dim=8, max_num_tokens=65535)

plt.tight_layout()
plt.show()
plt.savefig("rope_attention.png", dpi=300, bbox_inches="tight")