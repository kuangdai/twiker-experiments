if __name__ == "__main__":
    K = [3, 5, 7]
    P = ["true", "false"]
    H = ["true", "false"]
    L = ["true", "false"]
    with open('twike_train_template.json') as f:
        template = f.read()
    script = ""
    for k in K:
        for p in P:
            for h in H:
                for l in L:
                    out = template.replace('@K', str(k))
                    out = out.replace('@P', p)
                    out = out.replace('@H', h)
                    out = out.replace('@L', l)
                    with open(f'hyper/twike_train_{k}_{p}_{h}_{l}.json', 'w') as f:
                        f.write(out)
                    script += f"python run_clm.py config/gpt2/wikitext/hyper/twike_train_{k}_{p}_{h}_{l}.json\n"
    with open('run_all.sh', 'w') as f:
        f.write(script)
