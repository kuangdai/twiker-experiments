if __name__ == "__main__":
    K = [7, 5, 3]
    S = ["true", "false"]
    H = ["true", "false"]
    L = ["true", "false"]
    with open('template.json') as f:
        template = f.read()
    script = ""
    for k in K:
        for s in S:
            for h in H:
                for l in L:
                    out = template.replace('@K', str(k))
                    out = out.replace('@S', s)
                    out = out.replace('@H', h)
                    out = out.replace('@L', l)
                    with open(f'train_twiker_{k}_{s}_{h}_{l}.json', 'w') as f:
                        f.write(out)
                    script += f"python run_clm.py hyper/train_twiker_{k}_{s}_{h}_{l}.json\n"
    with open('run_all.sh', 'w') as f:
        f.write(script)
