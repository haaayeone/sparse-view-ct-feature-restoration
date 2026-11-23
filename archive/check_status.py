"""
check_status.py — 실험이 어디까지 됐는지 한 번에 확인. 파일을 읽기만 하고 아무것도 바꾸지 않는다.

보는 것
  [1] 지금 돌고 있는 작업 (tmux 세션, GPU 사용 프로세스, 우리 스크립트 python 프로세스)
  [2] compare/results: 인코더 × view × 방법별로 끝난 seed  → MLP 주 결과 / 공정 비교 2단계 / A+C 진행률
  [3] 집계·bootstrap csv (results_restorers_*.csv): 언제 만들었는지, ref가 무엇인지, 그 뒤에 새 결과가 생겼는지
  [4] 프로브 비교 (compare_probe), [5] 처음 보는 view 수 (compare_unseen*), [6] 검사 단위 평가 (preds, study_auroc, sens_at_spec)
  [7] 최근 로그 5개와 마지막 줄
  [8] (--features) feature 추출 개수 (조금 느림)

실행 위치: 서버. 저장소 루트에서.
Usage (서버):
  conda activate hayeon
  cd ~/hayeon/ich_cf_clean
  python check_status.py
  python check_status.py --features
"""
import argparse
import csv
import glob
import json
import os
import re
import subprocess
import time
from collections import defaultdict

ENCS = ["medsiglip_after", "siglip_window", "dinov2_window", "resnet50_window", "densenet121_window"]
SHORT = {"medsiglip_after": "MedSigLIP", "siglip_window": "SigLIP", "dinov2_window": "DINOv2",
         "resnet50_window": "ResNet50", "densenet121_window": "DenseNet121"}
MAIN_VIEWS = [180, 90, 45, 23, 12, 6]
CMP_VIEWS = [6, 23, 90]
CMP_METHODS = ["identity", "meanshift", "coral", "ridge", "mlp", "direct", "diffusion"]
CLOSED = {"identity", "meanshift", "coral", "ridge"}
SEEDS = [42, 43, 44]
SHOW_METHODS = ["identity", "meanshift", "coral", "ridge", "mlp", "direct", "diffusion", "diffusion_vl",
                "diffusion_nc"]
LABEL = {"identity": "id", "meanshift": "mshift", "coral": "coral", "ridge": "ridge", "mlp": "mlp", "direct": "direct",
         "diffusion": "diff", "diffusion_vl": "diff_vl", "diffusion_nc": "A+C"}


def ts(path):
    return time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(path))) if os.path.exists(path) else "-"


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"(실행 실패: {e})"


def head(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def parse_tag(tag):
    m = re.match(r"^(.+?)_r360_nv(\d+)$", tag)
    return (m.group(1), int(m.group(2))) if m else (tag, None)


def load_results(d):
    """d/results/*.json → {(enc, view): {method: set(seeds)}} , 가장 최근 수정 시각"""
    out, newest = defaultdict(lambda: defaultdict(set)), 0
    for f in glob.glob(os.path.join(d, "results", "*.json")):
        name = os.path.basename(f)[:-5]
        parts = name.split("__")
        if len(parts) < 3:
            continue
        tag, method, s = parts[0], parts[1], parts[2]
        seed = int(s[1:]) if s.startswith("s") and s[1:].isdigit() else s
        enc, view = parse_tag(tag)
        out[(enc, view)][method].add(seed)
        newest = max(newest, os.path.getmtime(f))
    return out, newest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", action="store_true", help="feature 추출 개수까지 센다 (느림)")
    ap.add_argument("--feat_root", default="/data0/hayeon/features_v2")
    a = ap.parse_args()
    print(f"확인 시각 {time.strftime('%Y-%m-%d %H:%M')}   위치 {os.getcwd()}")

    # [1] 돌고 있는 작업
    head("[1] 지금 돌고 있는 작업")
    tm = sh("tmux ls 2>&1")
    if not tm or "no server running" in tm or "error connecting" in tm:
        tm = "(없음)"
    print("tmux 세션:\n  " + tm.replace("\n", "\n  "))
    gpu = sh("nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>&1")
    print("GPU (번호, 사용률, 메모리):\n  " + (gpu or "(정보 없음)").replace("\n", "\n  "))
    keys = ("compare_restorers", "collect_restorers", "train_nocond", "train_probe_variants", "eval_unseen_views",
            "extract_test_views", "run_extract_encoder", "study_level_eval", "dump_preds", "compare_all",
            "train_diffusion", "run_eval")
    lines = []
    for ln in sh("ps -eo pid,etime,args").splitlines()[1:]:
        if any(k in ln for k in keys) and "check_status" not in ln and "grep" not in ln \
                and " -c " not in ln and ("python" in ln or "bash scripts" in ln):
            lines.append(ln[:160])
    procs = "\n".join(lines)
    print("우리 스크립트 프로세스 (pid, 실행 시간, 명령):\n  " + (procs or "(없음: 지금 돌고 있는 실험 없음)").replace("\n", "\n  "))

    # [2] compare/results
    head("[2] 복원기 결과 (compare/results)")
    res, newest_cmp = load_results("compare")
    if not res:
        print("결과 없음 (compare/results 비어 있음)")
    else:
        w = 8
        print(f"{'encoder':<12}{'view':>5}  " + "".join(f"{LABEL[m]:>{w + 1}}" for m in SHOW_METHODS))
        for enc in ENCS + sorted({e for e, _ in res} - set(ENCS)):
            for v in sorted({v for e, v in res if e == enc and v is not None}, reverse=True):
                row = res[(enc, v)]
                cells = []
                for m in SHOW_METHODS:
                    s = sorted(x for x in row.get(m, set()) if isinstance(x, int))
                    cells.append(",".join(str(x) for x in s) if s else "-")
                print(f"{SHORT.get(enc, enc):<12}{v:>5}  " + "".join(f"{c:>{w + 1}}" for c in cells))
        extra = sorted({m for r in res.values() for m in r} - set(SHOW_METHODS))
        if extra:
            print(f"(표에 없는 방법: {', '.join(extra)})")
        print("칸의 숫자 = 끝난 seed (닫힌해 방법은 0)")

        def has(enc, v, m, s):
            return s in res.get((enc, v), {}).get(m, set())

        miss = [(e, v) for e in ENCS for v in MAIN_VIEWS if not has(e, v, "mlp", 42)]
        print(f"\n● MLP 주 결과 (인코더 5 × view 6, seed 42): {30 - len(miss)}/30 완료")
        if miss:
            for e in ENCS:
                vs = [str(v) for ee, v in miss if ee == e]
                if vs:
                    print(f"  남은 것  {SHORT[e]:<12} view {'·'.join(vs)}")
        need, done = [], 0
        for e in ENCS:
            for v in CMP_VIEWS:
                for m in CMP_METHODS:
                    for s in ([0] if m in CLOSED else SEEDS):
                        if has(e, v, m, s):
                            done += 1
                        else:
                            need.append((e, v, m, s))
        total = done + len(need)
        print(f"● 공정 비교 2단계 (인코더 5 × view 3 × 방법 7, 학습 방법은 seed 3개): {done}/{total} 완료")
        if need:
            per_cfg = len(CLOSED) + (len(CMP_METHODS) - len(CLOSED)) * len(SEEDS)
            by = defaultdict(list)
            for e, v, m, s in need:
                by[(e, v)].append(f"{m}" + ("" if m in CLOSED else f"(s{s})"))
            for e in ENCS:
                full = [v for v in CMP_VIEWS if len(by.get((e, v), [])) == per_cfg]
                part = [v for v in CMP_VIEWS if 0 < len(by.get((e, v), [])) < per_cfg]
                if full:
                    print(f"  남은 것  {SHORT[e]:<12} view {'·'.join(map(str, full))}: 전부")
                for v in part:
                    print(f"  남은 것  {SHORT[e]:<12} view {v}: {', '.join(by[(e, v)])}")
        print(f"● A+C (MedSigLIP 6-view, seed 42): {'완료' if has('medsiglip_after', 6, 'diffusion_nc', 42) else '없음'}")
        nw = len(glob.glob("compare/weights/*.pt"))
        print(f"● 저장된 복원기 가중치 (compare/weights): {nw}개   결과 파일 최근 수정: "
              f"{time.strftime('%m-%d %H:%M', time.localtime(newest_cmp)) if newest_cmp else '-'}")

    # [3] 집계 csv
    head("[3] 집계·bootstrap 파일 (collect_restorers.py 출력)")
    for f in ("results_restorers_runs.csv", "results_restorers_summary.csv", "results_restorers_bootstrap.csv"):
        print(f"  {f:<36} {ts(f)}")
    bf = "results_restorers_bootstrap.csv"
    if os.path.exists(bf):
        rows = list(csv.DictReader(open(bf, encoding="utf-8")))
        refs = sorted({r.get("ref", "") for r in rows})
        has_nc = any(r.get("method", "").startswith("diffusion_nc") for r in rows)
        print(f"  bootstrap 기준(ref): {', '.join(refs)}   A+C(diffusion_nc) 포함: {'예' if has_nc else '아니오'}")
    if newest_cmp and os.path.exists("results_restorers_runs.csv") and newest_cmp > os.path.getmtime("results_restorers_runs.csv"):
        print("  !! 집계 이후 새 결과가 생김 → collect_restorers.py를 다시 돌려야 최신 표가 됨")

    # [4] 프로브 비교
    head("[4] 프로브 비교 (exp_probe → compare_probe)")
    pr = glob.glob("compare_probe/results/*.json")
    if pr:
        views = defaultdict(int)
        for f in pr:
            enc, v = parse_tag(os.path.basename(f).split("__")[0])
            views[(enc, v)] += 1
        for (e, v), n in sorted(views.items(), key=lambda x: (x[0][0], -(x[0][1] or 0))):
            print(f"  {SHORT.get(e, e)} {v}-view: 조건 {n}개")
    else:
        print("  결과 없음")
    for f in ("results_probe_matrix.csv", "results_probe_bootstrap.csv"):
        print(f"  {f:<36} {ts(f)}")

    # [5] 처음 보는 view 수
    head("[5] 처음 보는 view 수 (exp_unseen_views → compare_unseen*)")
    dirs = sorted(glob.glob("compare_unseen*"))
    if not dirs:
        print("  결과 없음")
    for d in dirs:
        n = len(glob.glob(os.path.join(d, "results", "*.json")))
        print(f"  {d:<28} 결과 {n}개")
    for f in sorted(glob.glob("results_unseen*.csv")):
        print(f"  {f:<36} {ts(f)}")

    # [6] 검사 단위 평가
    head("[6] 검사 단위 평가 (dump_preds → study_level_eval)")
    print(f"  preds 테스트 예측: {len(glob.glob('preds/*test*.npz'))}개   검증 예측: {len(glob.glob('preds/*val*.npz'))}개")
    for f in ("study_auroc_perlabel.csv", "study_auroc_summary.csv", "sens_at_spec.csv"):
        print(f"  {f:<36} {ts(f)}")

    # [7] 최근 로그
    head("[7] 최근 로그 5개 (마지막 줄)")
    logs = [f for p in ("logs/*.log", "compare*/logs/*.log") for f in glob.glob(p)]
    for f in sorted(logs, key=os.path.getmtime, reverse=True)[:5]:
        last = ""
        try:
            with open(f, "rb") as fh:
                fh.seek(max(0, os.path.getsize(f) - 4000))
                lines = [l.strip() for l in fh.read().decode("utf-8", "ignore").splitlines() if l.strip()]
                last = lines[-1][:110] if lines else ""
        except OSError:
            pass
        print(f"  {ts(f)}  {f}\n      └ {last}")
    if not logs:
        print("  로그 없음")

    # [8] feature 개수
    if a.features:
        head("[8] feature 추출 개수 (test 폴더 .npy)")
        for d in sorted(glob.glob(os.path.join(a.feat_root, "*", "feature_*"))):
            t = os.path.join(d, "test")
            n = len(os.listdir(t)) if os.path.isdir(t) else 0
            print(f"  {d:<62} test {n:,}")
    print("\n이 출력을 그대로 붙여 주시면 다음에 할 일을 정리해 드립니다.")


if __name__ == "__main__":
    main()
