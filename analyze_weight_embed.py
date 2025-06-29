import argparse
import json

import numpy as np
import torch


def load_weights(name):
    """Read the weight tensor."""
    weights_path = f"./weights/embed/{name}.pt"
    w = torch.load(weights_path)
    return w


def main():
    parser = argparse.ArgumentParser(description="Process GPT-2 tokens.")
    parser.add_argument("-n", "--name", type=str, required=True, help="Name of dataset.")
    parser.add_argument("-c", "--count", type=int, default=1, help="Minimum count.")
    args = parser.parse_args()

    # Load poses
    print(args.name)
    with open(f"results/pos/{args.name}.json", "r") as file:
        token_pos_dict = json.load(file)
    print("Total tokens in dataset:", len(token_pos_dict))

    # Load weights
    w_model = load_weights(args.name)
    ids_all = np.array(list(token_pos_dict.keys())).astype(int)
    count_all = np.array([len(v["poses"]) for v in token_pos_dict.values()])
    ids_all = ids_all[count_all >= args.count]
    dist_all = w_model[ids_all].norm(dim=1).mean()
    print("Mean embedding:", dist_all)

    # Classes
    unique_poses = sorted(set(pos for v in token_pos_dict.values() for pos in v["poses"]))
    unique_poses.append("prop")
    unique_poses.remove("adp")
    unique_poses.remove("part")
    pos_weight_dict = {}
    for p in unique_poses:
        pos_weight_dict[p] = []

    for token_id, token_dict in token_pos_dict.items():
        token_id = int(token_id)
        token_poses = token_dict["poses"]
        token_poses = ["prop" if x in {"part", "adp"} else x for x in token_poses]
        unique, count = np.unique(token_poses, return_counts=True)
        p = unique[np.argmax(count)]
        pos_weight_dict[p].append(w_model[token_id])

    pos_dist_dict = {"all": dist_all.item()}
    for p in unique_poses:
        if len(pos_weight_dict[p]) == 0:
            pos_dist_dict[p] = 0.
        else:
            w = torch.stack(pos_weight_dict[p], dim=0)
            pos_dist_dict[p] = w.norm(dim=1).mean().item()
    with open(f"results/embed/{args.name}.json", "w") as file:
        json.dump(pos_dist_dict, file)


if __name__ == "__main__":
    main()
