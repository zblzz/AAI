import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import time


def run_one(python_exe: str, train_script: str, data_dir: str, model_dir: str,
            spca_k: int, n_components: int, seed: int, voters: str, clf_method: str, dim_method: str):
    cmd = [
        python_exe, train_script,
        "--data_dir", data_dir,
        "--model_dir", model_dir,
        "--dim_method", dim_method,
        "--clf_method", clf_method,
        "--voters", voters,
        "--spca_k", str(spca_k),
        "--n_components", str(n_components),
        "--seed", str(seed),
    ]
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dur = time.time() - start

    stdout = proc.stdout
    stderr = proc.stderr

    # 解析 no-leak CV 的 acc/auc
    # 你的 train.py 输出格式：
    # CV Accuracy (no-leak): 0.7006 (+/- 0.1140)
    # CV AUC (no-leak)     : 0.7595 (+/- 0.1266)
    def extract_metric(text: str, key: str):
        for line in text.splitlines():
            if key in line:
                # 取冒号后第一个浮点数
                try:
                    right = line.split(":")[1].strip()
                    val = float(right.split()[0])
                    return val, line.strip()
                except Exception:
                    return None, line.strip()
        return None, None

    acc, acc_line = extract_metric(stdout, "CV Accuracy (no-leak)")
    auc, auc_line = extract_metric(stdout, "CV AUC (no-leak)")

    return {
        "spca_k": spca_k,
        "n_components": n_components,
        "seed": seed,
        "acc_no_leak": acc,
        "auc_no_leak": auc,
        "returncode": proc.returncode,
        "duration_sec": round(dur, 2),
        "acc_line": acc_line or "",
        "auc_line": auc_line or "",
        "stderr_tail": "\n".join(stderr.splitlines()[-30:]) if stderr else "",
    }


def main():
    base_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Grid search over --spca_k and --n_components by running scripts/train.py")
    parser.add_argument("--python", default=sys.executable, help="Python executable to run train.py (default: current)")
    parser.add_argument("--train_script", default=str(base_dir / "scripts" / "train.py"))
    parser.add_argument("--data_dir", default=str(base_dir / "data"))
    parser.add_argument("--model_dir", default=str(base_dir / "models"))  # train.py 仍会写入模型，这里只是必传
    parser.add_argument("--out_csv", default=str(base_dir / "output" / "grid_spca_results.csv"))
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--dim_method", default="ensemble_spca")
    parser.add_argument("--clf_method", default="stacking")
    parser.add_argument("--voters", default="lr,svm_calib,xgb")

    parser.add_argument("--spca_k_list", default="1500,2000,2500")
    parser.add_argument("--n_components_list", default="60,80,100,120")

    parser.add_argument("--stop_on_error", action="store_true", help="Stop grid search on first failure.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    spca_ks = [int(x.strip()) for x in args.spca_k_list.split(",") if x.strip()]
    n_comps = [int(x.strip()) for x in args.n_components_list.split(",") if x.strip()]

    rows = []
    total = len(spca_ks) * len(n_comps)
    idx = 0

    print(f"Running grid search: {total} runs")
    print(f"spca_k_list={spca_ks}")
    print(f"n_components_list={n_comps}")
    print(f"Output CSV: {args.out_csv}\n")

    for k in spca_ks:
        for nc in n_comps:
            idx += 1
            print(f"[{idx}/{total}] spca_k={k}, n_components={nc}")
            res = run_one(
                python_exe=args.python,
                train_script=args.train_script,
                data_dir=args.data_dir,
                model_dir=args.model_dir,
                spca_k=k,
                n_components=nc,
                seed=args.seed,
                voters=args.voters,
                clf_method=args.clf_method,
                dim_method=args.dim_method,
            )
            rows.append(res)

            if res["returncode"] != 0:
                print("  -> FAILED")
                if res["stderr_tail"]:
                    print("  stderr (tail):")
                    print(res["stderr_tail"])
                if args.stop_on_error:
                    break
            else:
                print(f"  -> acc_no_leak={res['acc_no_leak']} auc_no_leak={res['auc_no_leak']} time={res['duration_sec']}s")

        if args.stop_on_error and rows and rows[-1]["returncode"] != 0:
            break

    # 写CSV
    fieldnames = [
        "spca_k", "n_components", "seed",
        "acc_no_leak", "auc_no_leak",
        "returncode", "duration_sec",
        "acc_line", "auc_line",
        "stderr_tail",
    ]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # 打印 Top 结果（按 AUC）
    ok = [r for r in rows if r["returncode"] == 0 and r["auc_no_leak"] is not None]
    ok_sorted = sorted(ok, key=lambda x: x["auc_no_leak"], reverse=True)
    print("\nDone. Top results by no-leak AUC:")
    for r in ok_sorted[:10]:
        print(f"  spca_k={r['spca_k']}, n_components={r['n_components']}, auc={r['auc_no_leak']:.4f}, acc={r['acc_no_leak']:.4f}, time={r['duration_sec']}s")


if __name__ == "__main__":
    main()