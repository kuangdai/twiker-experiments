import os
from pathlib import Path


def replace_in_file(source, replace_dict, dest=None):
    if dest is None:
        dest = source
    with open(source, 'r') as fs:
        text = fs.read()
    for key, val in replace_dict.items():
        text = text.replace(key, str(val))
    with open(dest, 'w') as fs:
        fs.write(text)


if __name__ == "__main__":
    config_path = Path("configs/gpt2")
    data_path = Path('data/datasets')
    ds_paths = [f for f in data_path.iterdir() if not f.is_file()]
    top_path = data_path.parent.parent.absolute()

    for ds_path in ds_paths:
        name = ds_path.name
        os.system(f"cp -r ./templates {str(config_path)}/{name}")
        replace_in_file(f"{str(config_path)}/{name}/train_base.json", {
            "__TOP_DIR__": str(top_path),
            "__DATA_NAME__": name
        })
        replace_in_file(f"{str(config_path)}/{name}/train_twiker.json", {
            "__TOP_DIR__": str(top_path),
            "__DATA_NAME__": name
        })
