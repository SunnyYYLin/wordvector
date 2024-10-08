import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import os

def plot_curves(log_dir, tags, plot_title="Log Curves"):
    """
    从多个 TensorBoard 日志文件中提取损失曲线，并为每个标签绘制独立的子图。
    
    参数:
        log_dir (str): TensorBoard 日志文件的文件夹路径，包含多个日志文件。
        tags (list[str]): 要提取和绘制的损失名称列表，例如 ['loss']。
        plot_title (str): 图表的总标题。
    """
    # 获取log_dir目录下的所有文件
    log_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if os.path.isfile(os.path.join(log_dir, f))]
    
    num_tags = len(tags)
    fig, axes = plt.subplots(1, num_tags, figsize=(6*num_tags, 5))

    # 如果只有一个标签，axes不会是列表，强制转换为列表
    if num_tags == 1:
        axes = [axes]

    # 遍历每个标签，为每个标签创建单独的子图
    for i, tag in enumerate(tags):
        ax = axes[i]
        
        # 遍历每个日志文件
        for log_file in log_files:
            # 获取日志文件名
            log_name = os.path.basename(log_file).removeprefix('events.out.tfevents.')
            
            # 创建事件累加器来加载日志数据
            ea = event_accumulator.EventAccumulator(log_file, size_guidance={'scalars': 0})
            ea.Reload()

            # 检查是否存在当前的标签
            if tag in ea.Tags()['scalars']:
                # 获取每个标签的事件
                events = ea.Scalars(tag)
                # 提取步数和损失值
                steps = [e.step for e in events]
                values = [e.value for e in events]

                # 绘制损失曲线，以log文件名作为图例
                ax.plot(steps, values, label=log_name)

        # 设置每个子图的标签和标题
        ax.set_xlabel('Steps')
        ax.set_ylabel('Value')
        ax.set_title(f"{tag} Curve", loc='center', pad=10)
        ax.legend(loc='best')
        ax.grid(True)

    # 设置整个图的标题
    plt.suptitle(plot_title, fontsize=16, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整整体布局以防止标题重叠
    plt.show()