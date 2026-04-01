import os
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # 强制使用纯后台渲染，防止 GUI 卡死、崩溃或弹窗
import matplotlib.pyplot as plt
from io import BytesIO

from config import FileAddress, MapConfig
from main import build_env
from visualization import visualize_current_environment

def generate_temporal_current_frames():
    # ===================== 核心配置 =====================
    REAL_TIME_HOURS = 48         # 要展示的真实海流跨度（48小时）
    GIF_DURATION_SECONDS = 10    # GIF 总播放时长（10秒）
    VIS_INTERVAL_MINUTES = 20    # 每20分钟生成一帧（48小时共145帧，兼顾流畅度与内存）
    
    # 计算GIF总时长（毫秒），用于后续分配每帧停留时间
    total_gif_duration_ms = GIF_DURATION_SECONDS * 1000

    # 1. 强制覆盖配置，确保环境正确加载时变真实海流
    MapConfig.current_mode = "real"
    MapConfig.use_temporal_current = True
    MapConfig.time_compress_hours = REAL_TIME_HOURS  # 传递真实时长到配置
    
    print("Building environment...")
    env = build_env()
    env.reset()
    
    if not env.current_on or not getattr(env.current, "enable_temporal", False):
        raise ValueError("时变海流未开启！请检查 MapConfig 设置。")

    # 2. 计算时间步长与帧数
    total_real_seconds = REAL_TIME_HOURS * 3600       # 48小时总秒数
    vis_interval_seconds = VIS_INTERVAL_MINUTES * 60  # 帧间隔秒数
    total_frames = int(total_real_seconds // vis_interval_seconds) + 1
    
    # 创建输出文件夹
    output_dir = FileAddress.results / "real-time_current_fig"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_time = env.current.time0
    
    map_frames = []    # 三面板图的内存图像列表
    d3_frames = []     # 纯3D图的内存图像列表

    print("\n==================================================")
    print(f"开始生成海流快照：")
    print(f"▶ 真实物理时长：{REAL_TIME_HOURS} 小时")
    print(f"▶ GIF 播放时长：{GIF_DURATION_SECONDS} 秒")
    print(f"▶ 采样时间间隔：{VIS_INTERVAL_MINUTES} 分钟/帧")
    print(f"▶ 预计生成帧数：{total_frames} 帧")
    print("==================================================\n")

    # 3. 循环遍历时间，生成帧
    for i in range(total_frames):
        # 计算当前帧对应的真实时间（从base_time开始推进）
        elapsed_real_sec = i * vis_interval_seconds
        current_time = base_time + pd.Timedelta(seconds=elapsed_real_sec)
        env.current.set_simulation_time(current_time)
        
        # 显示真实时间偏移（便于验证）
        elapsed_real_hours = elapsed_real_sec / 3600
        title = f"Temporal Current Field | +{elapsed_real_hours:.1f} hrs | {current_time.strftime('%H:%M:%S')}"
        
        # 核心逻辑：不传 save_path，让可视化模块直接返回 fig 和 fig_3d 内存对象
        fig, fig_3d = visualize_current_environment(
            env=env,
            save_path=None, 
            title=title
        )
        
        # -------- 处理三面板图 (fig) --------
        map_img_buffer = BytesIO()
        fig.savefig(map_img_buffer, format='png', dpi=150, bbox_inches="tight") 
        map_img_buffer.seek(0)
        map_frame = Image.open(map_img_buffer)
        map_frame.load()  # 强制加载进内存，防止 buffer 意外关闭导致图片黑屏
        map_frames.append(map_frame)
        
        # -------- 处理单3D图 (fig_3d) --------
        d3_img_buffer = BytesIO()
        fig_3d.savefig(d3_img_buffer, format='png', dpi=150, bbox_inches="tight")
        d3_img_buffer.seek(0)
        d3_frame = Image.open(d3_img_buffer)
        d3_frame.load()
        d3_frames.append(d3_frame)
        
        # 必须 close，释放 matplotlib 占用的系统内存
        plt.close(fig)
        plt.close(fig_3d)
        
        print(f"[{i+1:03d}/{total_frames}] 已渲染 -> 进度：{elapsed_real_hours:.1f} 小时 / {REAL_TIME_HOURS} 小时")

    # 4. 生成 GIF
    print("\n所有帧渲染完毕！正在打包生成 GIF...")
    # 计算每帧的停留时长（毫秒）：总GIF时长 / 总帧数
    frame_duration_ms = total_gif_duration_ms / total_frames
    
    if map_frames:
        create_gif_from_frames(map_frames, output_dir / "temporal_currents_map.gif", duration_ms=frame_duration_ms)
    if d3_frames:
        create_gif_from_frames(d3_frames, output_dir / "temporal_currents_3D.gif", duration_ms=frame_duration_ms)

def create_gif_from_frames(frames, output_path, duration_ms=200):
    print(f"正在保存: {output_path.name}")
    print(f"  - 每帧停留时长：{duration_ms:.1f} 毫秒 | 帧率：约 {1000/duration_ms:.1f} FPS")
    
    frames[0].save(
        output_path,
        format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=duration_ms, 
        loop=0 
    )
    print(f"  - √ 保存成功！(路径: {output_path})")

if __name__ == "__main__":
    generate_temporal_current_frames()