import argparse
import json

import numpy as np
import torch


def load_weights(name, kernel_size, heads, temperature, using_keys):
    """Process the weight tensor and compute distances."""
    weights_path = f"./weights/{name}.pt"
    w = torch.load(weights_path)
    assert w.shape[1] == 2 * kernel_size * heads, "Mismatch in weight dimensions."

    w = w.reshape(-1, 2, heads, kernel_size)
    w = w[:, 0 if using_keys else 1, :, :]
    w = torch.mean(w, dim=1)  # Mean over heads
    w = torch.softmax(w / temperature, dim=-1)
    return w


def main():
    parser = argparse.ArgumentParser(description="Process GPT-2 tokens.")
    parser.add_argument("-n", "--name", type=str, required=True, help="Name of dataset.")
    parser.add_argument("-c", "--count", type=int, default=1, help="Minimum count.")
    parser.add_argument("-t", "--temperature", type=float, default=0.4, help="Temperature in softmax.")
    parser.add_argument("-m", "--heads", type=int, default=1, help="Number of heads.")
    parser.add_argument("-s", "--kernel_size", type=int, default=3, help="Kernel size.")
    parser.add_argument("--using-keys", action="store_true", help="Trained with keys not values.")
    args = parser.parse_args()

    # Load poses
    with open(f"results/pos/{args.name}.json", "r") as file:
        token_pos_dict = json.load(file)
    print("Total tokens in dataset:", len(token_pos_dict))

    # Load weights
    origin = torch.zeros(args.kernel_size)
    origin[args.kernel_size // 2] = 1.
    w_model = load_weights(args.name, args.kernel_size, args.heads, args.temperature, args.using_keys)
    ids_all = np.array(list(token_pos_dict.keys())).astype(int)
    count_all = np.array([len(v["poses"]) for v in token_pos_dict.values()])
    ids_all = ids_all[count_all >= args.count]
    dist_all = (w_model[ids_all] - origin[None, :]).norm(dim=1).mean()
    print("Mean distance to origin vector:", dist_all)

    # Classes
    unique_poses = sorted(set(pos for v in token_pos_dict.values() for pos in v["poses"]))
    unique_poses.append("prop")
    unique_poses.remove("adp")
    unique_poses.remove("part")
    pos_weight_dict = {}
    for p in unique_poses:
        pos_weight_dict[p] = {}
        pos_weight_dict[p]["w_sum"] = torch.zeros(args.kernel_size)
        pos_weight_dict[p]["w_factor"] = 0.

    for token_id, token_dict in token_pos_dict.items():
        token_id = int(token_id)
        token_poses = token_dict["poses"]
        token_poses = ["prop" if x in {"part", "adp"} else x for x in token_poses]
        unique, count = np.unique(token_poses, return_counts=True)
        for p, c in zip(unique, count):
            if c >= args.count:
                factor = c / len(token_poses)
                pos_weight_dict[p]["w_sum"] += w_model[token_id] * factor
                pos_weight_dict[p]["w_factor"] += factor

    pos_dist_dict = {"all": dist_all.item()}
    for p in unique_poses:
        w = pos_weight_dict[p]["w_sum"] / pos_weight_dict[p]["w_factor"]
        pos_dist_dict[p] = (w - origin[None, :]).norm(dim=1).mean().item()
    with open(f"results/distance/{args.name}.json", "w") as file:
        json.dump(pos_dist_dict, file)


if __name__ == "__main__":
    main()
