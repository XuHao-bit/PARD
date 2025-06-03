import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42

def draw_heatmap(path):
  raw_data = torch.load(path)
  # embedding_euser = raw_data['server']['embedding_euser.weight']
  data = torch.zeros(0, 128)
  for uidx in list(raw_data['client'].keys()):
    user = raw_data['client'][uidx]
    # row = torch.cat((user['embedding_puser.weight'], embedding_euser), dim=1)
    # row = torch.cat((user['embedding_puser.weight'], user['test_euser_emb']), dim=1)
    row = torch.cat((user['embedding_puser.weight'], user['embedding_euser.weight']), dim=1)
    data = torch.vstack((data, row))

  corr_matrix = torch.corrcoef(data.T)  # 输出形状[128, 128]
  # 转换为绝对值矩阵（关键修改点）
  abs_corr = torch.abs(corr_matrix)

  # 转换为numpy数组（GPU数据需先转CPU）
  abs_np = abs_corr.cpu().numpy()

  # 创建画布与子图
  fig, ax = plt.subplots(figsize=(8, 7))

  # 绘制热力图（关键参数调整）
  heatmap = ax.imshow(abs_np,
                    cmap='Blues',  # 改用黄-橙-红渐变色系表示强度[7](@ref)
                    vmin=0,         # 最小值设为0[6](@ref)
                    vmax=1)         # 最大值保持1[7](@ref)

  # 添加颜色条
  cbar = fig.colorbar(heatmap, ax=ax, shrink=0.7)
  # cbar.set_label('Absolute Correlation', rotation=270, labelpad=20)  # 修改颜色条标签[6](@ref)

  # 坐标轴设置
  # ax.set_title("Absolute Pearson Correlation Heatmap", fontsize=14, pad=20)
  ax.set_xlabel("Feature Index", fontsize=12)
  ax.set_ylabel("Feature Index", fontsize=12)

  # 优化刻度显示（每10个特征显示一个主刻度）
  ax.set_xticks(ticks=range(0, 128, 10))  
  ax.set_yticks(ticks=range(0, 128, 10))
  ax.tick_params(axis='both', which='major', labelsize=8)

  # 添加网格线（提升可读性）
  ax.grid(which='major', color='w', linestyle='--', linewidth=0.5, alpha=0.3)
  plt.tight_layout()
  plt.savefig(path + '.svg', bbox_inches='tight')

path = 'saved_model/ml-100k/pretrain_fedncf_ml100k_new+lr0.5+eta80'
path = 'saved_model/ml-100k/pretrain_fedncf_ml100k_peuser_client'
# draw_heatmap(path)
path = 'saved_model/ml-100k/trained_fedncf_ml100k_new+lr0.1+eta1.0'
path = 'saved_model/ml-100k/trained_fedncf_ml100k_peuser_client'
draw_heatmap(path)


# path = 'saved_model/ali-ads/pretrain_fedncf_aliads_new+lr0.5+eta80'
# draw_heatmap(path)
# # # path = 'saved_model/ali-ads/trained_fedncf_aliads_new+lr0.1+eta80.0'
# path = 'saved_model/ali-ads/trained_fedncf_aliads_epoch38+lr0.1+eta80.0'
# draw_heatmap(path)