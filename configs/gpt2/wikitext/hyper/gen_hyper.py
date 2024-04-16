if __name__ == "__main__":
    K = [7, 5, 3]
    S = ["true", "false"]
    H = ["true", "false"]
    L = ["true", "false"]
    C = ["none", "only_left_half", "truncate_near_boundary", "shrink_near_boundary"]
    with open('template.json') as f:
        template = f.read()
    script = ""
    for k in K:
        for s in S:
            for h in H:
                for l in L:
                    for c in C:
                        out = template.replace('@K', str(k))
                        out = out.replace('@S', s)
                        out = out.replace('@H', h)
                        out = out.replace('@L', l)
                        out = out.replace('@C', c)
                        with open(f'train_twiker_{k}_{s}_{h}_{l}_{c}.json', 'w') as f:
                            f.write(out)
                        script += f"python main.py hyper/train_twiker_{k}_{s}_{h}_{l}_{c}.json\n"
    with open('../run_all_hyper.sh', 'w') as f:
        f.write(script)
