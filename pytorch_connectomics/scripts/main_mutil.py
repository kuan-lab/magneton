import yaml
import copy
import subprocess

# config_file = "configs/SNEMI/SNEMI-Affinity-UNet.yaml"
# config_base = "configs/SNEMI/SNEMI-Base.yaml"
# checkpoint = "outputs/SNEMI_UNet_FTv5p2/checkpoint_46000.pth.tar"

config_file = "configs/SNEMI/SNEMI-Affinity-UNet.yaml"
config_base = "configs/SNEMI/SNEMI-Base.yaml"
checkpoint = "outputs/FT4/checkpoint_24000.pth.tar"

# 想动态替换的参数集合（示例）
configs_to_run = [
    {
        "INFERENCE": {
            "IMAGE_NAME": f"worm_{i:03d}.tif",   # 格式化成三位数
            "OUTPUT_NAME": f"worm_{i:03d}.h5"
        }
    }
    for i in range(224)  # 0 ~ 55
]

for i, new_params in enumerate(configs_to_run):
    # 加载原始 YAML
    with open(config_base) as f:
        config_data = yaml.safe_load(f)

    # 替换参数（支持递归替换）
    def update_dict(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = update_dict(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    update_dict(config_data, new_params)

    # 保存到一个临时新配置文件
    new_config_path = f"configs/SNEMI/chunks_full/temp_config_{i}.yaml"
    with open(new_config_path, "w") as f:
        yaml.safe_dump(config_data, f)

    # 运行主程序
    # print(f"Running with config {new_config_path}")
    # subprocess.run([
    #     "python", "-u", "scripts/main.py",
    #     "--config-base", new_config_path,
    #     "--config-file", config_file,
    #     "--inference",
    #     "--checkpoint", checkpoint
    # ])
