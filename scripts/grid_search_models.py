import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import time


def extract_metric(text: str, key: str):
    """
    从 train.py stdout 里提取类似：
      CV Accuracy (no-leak): 0.7006 (+/- 0.1140)
      CV AUC (no-leak)     : 0.7595 (+/- 0.1266)
    """
    for line in text.splitlines():
        if key in line:
            try:
                right = line.split(":")[1].strip()
                val = float(right.split()[0])
                return val, line.strip()
            except Exception:
                return None, line.strip()
    return None, None


def run_one(python_exe: str, train_script: str, data_dir: str, model_dir: str,
            dim_method: str, clf_method: str, voters: str,
            spca_k: int, n_components: int, seed: int):
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

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    acc, acc_line = extract_metric(stdout, "CV Accuracy (no-leak)")
    auc, auc_line = extract_metric(stdout, "CV AUC (no-leak)")

    return {
        "dim_method": dim_method,
        "clf_method": clf_method,
        "voters": voters,
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


def split_csv_list(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    base_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Grid search over dim reducers & classifiers at fixed spca_k/n_components by running scripts/train.py"
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--train_script", default=str(base_dir / "scripts" / "train.py"))
    parser.add_argument("--data_dir", default=str(base_dir / "data"))
    parser.add_argument("--model_dir", default=str(base_dir / "models"))
    parser.add_argument("--out_csv", default=str(base_dir / "output" / "grid_models_results.csv"))

    parser.add_argument("--seed", type=int, default=42)

    # 固定参数（按你的要求）
    parser.add_argument("--spca_k", type=int, default=3500)
    parser.add_argument("--n_components", type=int, default=120)

    # 搜索空间
    parser.add_argument("--dim_methods", default="ensemble_spca,spca,pca,baseline_select,fa,none",
                        help="降维器列表（逗号分隔），必须是 utils.get_dim_reducer 支持的名字")
    parser.add_argument("--clf_methods", default="stacking,voting,lr,svm_calib,rf,xgb,ridge,gnb,lr_elastic,knn",
                        help="分类器列表（逗号分隔），必须是 utils.get_classifier 支持的名字")

    # stacking/voting 用到 voters
    parser.add_argument("--voters", default="lr,svm_calib,xgb",
                        help="只在 clf_method=stacking/voting 时使用（逗号分隔）")

    parser.add_argument("--stop_on_error", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    dim_methods = split_csv_list(args.dim_methods)
    clf_methods = split_csv_list(args.clf_methods)

    rows = []
    total = len(dim_methods) * len(clf_methods)
    idx = 0

    print(f"Fixed params: spca_k={args.spca_k}, n_components={args.n_components}, seed={args.seed}")
    print(f"Dim methods: {dim_methods}")
    print(f"Clf methods: {clf_methods}")
    print(f"Voters     : {args.voters}")
    print(f"Total runs : {total}")
    print(f"Output CSV : {args.out_csv}\n")

    for dm in dim_methods:
        for cm in clf_methods:
            idx += 1
            print(f"[{idx}/{total}] dim_method={dm}, clf_method={cm}")
            res = run_one(
                python_exe=args.python,
                train_script=args.train_script,
                data_dir=args.data_dir,
                model_dir=args.model_dir,
                dim_method=dm,
                clf_method=cm,
                voters=args.voters,
                spca_k=args.spca_k,
                n_components=args.n_components,
                seed=args.seed
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

    # 保存 CSV
    fieldnames = [
        "dim_method", "clf_method", "voters",
        "spca_k", "n_components", "seed",
        "acc_no_leak", "auc_no_leak",
        "returncode", "duration_sec",
        "acc_line", "auc_line",
        "stderr_tail",
    ]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["returncode"] == 0 and r["auc_no_leak"] is not None]
    ok_sorted = sorted(ok, key=lambda x: x["auc_no_leak"], reverse=True)

    print("\nDone. Top results by no-leak AUC:")
    for r in ok_sorted[:10]:
        print(
            f"  dim={r['dim_method']:<14} clf={r['clf_method']:<10} "
            f"auc={r['auc_no_leak']:.4f} acc={r['acc_no_leak']:.4f} time={r['duration_sec']}s voters={r['voters']}"
        )


if __name__ == "__main__":
    main()