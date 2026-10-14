# 행정동 폭염 적응 군집 — 피처셋 × 알고리즘 × k × (원공간/PCA2) 실루엣 스윕. (팀 공유·재현용)
"""
질문: 어떤 피처를 더하면, 몇 개 군집으로, 어떻게 하면 실루엣이 오르나? 그 군집은 의미있나?
실행: python 군집_스윕.py   (같은 폴더의 dong_features_rich.parquet 만 있으면 됨)
출력: 콘솔 표 + 군집_스윕_결과.csv + 군집_탐구_그림.png
요약 결론은 README_정리.md 참고. (핵심: 원공간에 피처를 더하면 실루엣이 떨어지고, PCA2로 압축하면 회복된다.)
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.rcParams["font.family"] = ["AppleGothic", "Apple SD Gothic Neo", "Malgun Gothic", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
df = pd.read_parquet(HERE / "dong_features_rich.parquet").copy()
# 타깃성 피처는 이상치 winsorize (소표본 동 노이즈 완화)
for c in ["avoid", "shift_mid", "day_ratio"]:
    if c in df: lo, hi = df[c].quantile([0.01, 0.99]); df[c] = df[c].clip(lo, hi)

# 피처셋 정의 — 팀원이 자유롭게 추가/수정해서 탐구하면 됨
SETS = {
    "노출+회피(기준)":            ["expo_log", "avoid"],
    "+실내전환":                 ["expo_log", "avoid", "shift_mid"],
    "+주간인구":                 ["expo_log", "avoid", "주간인구지수"],
    "+야외관광":                 ["expo_log", "avoid", "야외관광커버"],
    "+쉼터공급+관광커버":          ["expo_log", "avoid", "공급_가중", "관광커버"],
    "+쉼터공급+관광지수":          ["expo_log", "avoid", "공급_가중", "관광지수"],
    "+쉼터공급+관광커버+관광지수":   ["expo_log", "avoid", "공급_가중", "관광커버", "관광지수"],
    "큐레이션6(보고서 채택)":       ["expo_log", "avoid", "공급_가중", "near_acc_km", "야외관광커버", "주간인구지수"],
    "전체14피처":                ["expo_log", "avoid", "shift_mid", "day_ratio", "접근가능수", "공급_가중",
                               "near_acc_km", "관광커버", "야외관광커버", "상업관광커버", "관광지수",
                               "LQ_단기외국인", "share_단기", "주간인구지수"],
}
USED = sorted(set(sum(SETS.values(), [])))
df[USED] = df[USED].fillna(df[USED].median())

def labels(X, k, algo):
    if algo == "KMeans": return KMeans(k, n_init=50, random_state=42).fit_predict(X)
    if algo == "GMM":    return GaussianMixture(k, n_init=5, random_state=42).fit_predict(X)
    return AgglomerativeClustering(k).fit_predict(X)

# ── 스윕 ──

rows = []
for sname, feats in SETS.items():
    Z = StandardScaler().fit_transform(df[feats].values)
    variants = {"원공간": Z}
    if len(feats) >= 3: variants["PCA2"] = PCA(2, random_state=42).fit_transform(Z)
    for vname, X in variants.items():
        for algo in ["KMeans", "GMM", "Ward"]:
            for k in [2, 3, 4, 5]:
                try:
                    rows.append((sname, len(feats), vname, algo, k, round(silhouette_score(X, labels(X, k, algo)), 3)))
                except Exception:
                    pass
res = pd.DataFrame(rows, columns=["피처셋", "피처수", "변환", "알고리즘", "k", "실루엣"])
res.to_csv(HERE / "군집_스윕_결과.csv", index=False, encoding="utf-8-sig")

print("=== 실루엣 상위 15 ===")
print(res.sort_values("실루엣", ascending=False).head(15).to_string(index=False))
print("\n=== 피처셋별 최고 실루엣 ===")
for sname in SETS:
    b = res[res["피처셋"] == sname].sort_values("실루엣", ascending=False).iloc[0]
    print(f"  {sname:24s} {b['실루엣']:.3f}  ({b['알고리즘']} {b['변환']} k={int(b['k'])})")
print("\n=== k별·변환별 최고 (원공간 vs PCA2 효과 확인용) ===")
for k in [2, 3]:
    for v in ["원공간", "PCA2"]:
        sub = res[(res["k"] == k) & (res["변환"] == v)]
        if len(sub):
            b = sub.sort_values("실루엣", ascending=False).iloc[0]
            print(f"  k={k} {v:5s}: 최고 {b['실루엣']:.3f} ({b['피처셋']})")

# ── 핵심 군집 프로파일 (의미 점검): 큐레이션6 k=3, 전체14 k=2 ──
def profile(feats, k, tag):
    X = PCA(2, random_state=42).fit_transform(StandardScaler().fit_transform(df[feats].values))
    df["_c"] = KMeans(k, n_init=50, random_state=42).fit_predict(X)
    s = silhouette_score(X, df["_c"])
    print(f"\n[{tag}] PCA2 k={k} 실루엣 {s:.3f}")
    p = df.groupby("_c").agg(n=("동명", "size"), 노출=("노출_폭염한낮", "mean"),
                             회피=("avoid", "mean"), 쉼터거리=("near_acc_km", "mean"))
    print(p.to_string(formatters={"노출": "{:,.0f}".format, "회피": "{:+.2f}".format, "쉼터거리": "{:.2f}".format}))
    for c in sorted(df["_c"].unique()):
        print(f"   군집{c} 예: " + ", ".join(df[df._c == c].nlargest(4, "노출_폭염한낮")["동명"]))
    return X, df["_c"].values, s
print("\n=== 군집 프로파일 (의미 점검) ===")
Xc, lc, sc = profile(SETS["큐레이션6(보고서 채택)"], 3, "큐레이션6")
X2, l2, s2 = profile(SETS["전체14피처"], 2, "전체14피처")

# ── 그림 3패널: (a) 실루엣 비교 (b) k=2 산점 (c) k=3 산점 ──
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
keys = ["노출+회피(기준)", "+쉼터공급+관광커버+관광지수", "큐레이션6(보고서 채택)", "전체14피처"]
def best_for(name, k=None):
    sub = res[res["피처셋"] == name]
    if k: sub = sub[sub["k"] == k]
    return sub["실루엣"].max()
bars = [best_for("노출+회피(기준)"), best_for("+쉼터공급+관광커버+관광지수"), best_for("큐레이션6(보고서 채택)"), best_for("전체14피처")]
ax[0].bar(["노출+회피\n(기준)", "쉼터+관광\n추가", "큐레이션6", "전체14"], bars, color=["#999", "#74a9cf", "#d7301f", "#08519c"])
for i, v in enumerate(bars): ax[0].text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
ax[0].axhline(0.5, color="#2ca02c", ls="--", lw=1); ax[0].axhline(0.4, color="#ff7f0e", ls="--", lw=1)
ax[0].set_ylabel("최고 실루엣"); ax[0].set_ylim(0, 0.65); ax[0].set_title("(a) 피처셋별 최고 실루엣 (0.4·0.5선)", fontsize=11)
for c in sorted(set(l2)):
    ax[1].scatter(X2[l2 == c, 0], X2[l2 == c, 1], s=28, alpha=0.7, label=f"군집{c} (n={int((l2==c).sum())})")
ax[1].set_title(f"(b) 전체14 PCA2 k=2 (실루엣 {s2:.2f})", fontsize=11); ax[1].legend(fontsize=8)
ax[1].set_xlabel("주성분1"); ax[1].set_ylabel("주성분2")
for c in sorted(set(lc)):
    ax[2].scatter(Xc[lc == c, 0], Xc[lc == c, 1], s=28, alpha=0.7, label=f"군집{c} (n={int((lc==c).sum())})")
ax[2].set_title(f"(c) 큐레이션6 PCA2 k=3 (실루엣 {sc:.2f})", fontsize=11); ax[2].legend(fontsize=8)
ax[2].set_xlabel("주성분1"); ax[2].set_ylabel("주성분2")
fig.suptitle("행정동 폭염 적응 군집 — 피처 추가·차원축소·k의 효과", fontsize=12.5, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig(HERE / "군집_탐구_그림.png", dpi=140); plt.close()
print("\n저장: 군집_스윕_결과.csv, 군집_탐구_그림.png")
print("=== 완료 ===")
