# 행정동 폭염 적응 군집 — 최종 채택본 (보고서 5.3)
"""
최종 구성: 피처 4개(노출·회피·야외관광커버·상업관광커버) · 원공간(PCA 미사용) · KMeans.
결과: k=2 실루엣 0.456, k=3 실루엣 0.452.

실행: python 군집_스윕.py   (같은 폴더의 dong_features_rich.parquet 만 있으면 됨)
출력: 콘솔 표 + 군집_결과.csv + 군집_그림.png

[채택 사유 — 탐색 과정 요약]
- 발표본의 2피처(노출·회피)는 k=3 실루엣 0.388에 그쳐 분리도가 약했다.
- 노출이 '어디서' 일어나는지를 담는 야외관광커버·상업관광커버를 더하면
  k=2 0.456 / k=3 0.452로 양호한 분리도를 얻는다(부트스트랩 ARI 0.95/0.91).
- PCA2나 다피처(예: 큐레이션6, 전체14)가 더 높은 실루엣을 내지만,
  (1) 피처가 적어 PCA는 차원을 버려 점수를 부풀릴 뿐 부적절하고,
  (2) 관광커버 = 야외관광커버 + 상업관광커버 같은 공선 피처가 같은 축에
      가중을 더해 점수만 올린다. → 원공간 4피처를 최종 채택.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib
matplotlib.rcParams["font.family"] = ["AppleGothic", "Apple SD Gothic Neo", "Malgun Gothic", "NanumGothic", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
df = pd.read_parquet(HERE / "dong_features_rich.parquet").copy()

# 회피율(avoid)은 소표본 동의 이상치를 상하위 1%에서 winsorize
lo, hi = df["avoid"].quantile([0.01, 0.99]); df["avoid"] = df["avoid"].clip(lo, hi)

# ── 최종 채택 피처 (원공간) ──
FEATS = ["expo_log", "avoid", "야외관광커버", "상업관광커버"]
df[FEATS] = df[FEATS].fillna(df[FEATS].median())
Z = StandardScaler().fit_transform(df[FEATS].values)   # 원공간(표준화만, PCA 없음)

def km(k):
    return KMeans(k, n_init=50, random_state=42).fit_predict(Z)

# ── 1) 군집 수 선택: k별 실루엣 ──
sil = {k: round(silhouette_score(Z, km(k)), 3) for k in [2, 3, 4, 5, 6]}
print("=== k별 실루엣 (원공간 · 4피처) ===")
for k, s in sil.items():
    print(f"  k={k}: {s:.3f}")
print(f"  → 채택: k=2 {sil[2]:.3f}, k=3 {sil[3]:.3f} (k=4부터 급락 → 3이 자연스러운 상한)")

lab2, lab3 = km(2), km(3)

# ── 2) 군집 안정성: 80% 부트스트랩 재표집 ARI ──
def stability(k, B=100, frac=0.8):
    base = km(k); rng = np.random.default_rng(0); aris = []
    for _ in range(B):
        idx = rng.choice(len(df), int(len(df) * frac), replace=False)
        Zs = StandardScaler().fit_transform(df[FEATS].iloc[idx].values)
        lab = KMeans(k, n_init=20, random_state=int(rng.integers(1e6))).fit_predict(Zs)
        aris.append(adjusted_rand_score(base[idx], lab))
    return np.mean(aris), np.std(aris)
print("\n=== 군집 안정성 (부트스트랩 ARI) ===")
for k in [2, 3]:
    m, sd = stability(k); print(f"  k={k}: ARI {m:.3f} ± {sd:.3f}")

# ── 3) k=2 ↔ k=3 포함관계 (교차표) ──
print("\n=== k=2 × k=3 교차표 (k=3은 k=2의 세분인가) ===")
print(pd.crosstab(lab2, lab3).to_string())

# ── 4) 군집 프로파일 (의미 점검) ──
def profile(lab, k):
    d = df.copy(); d["군집"] = lab
    p = d.groupby("군집").agg(동수=("동명", "size"), 노출=("노출_폭염한낮", "mean"),
                             회피=("avoid", "mean"), 야외관광=("야외관광커버", "mean"),
                             상업관광=("상업관광커버", "mean"), 쉼터거리=("near_acc_km", "mean"))
    print(f"\n[k={k}] 실루엣 {silhouette_score(Z, lab):.3f}")
    print(p.to_string(formatters={"노출": "{:,.0f}".format, "회피": "{:+.2f}".format,
                                  "야외관광": "{:.2f}".format, "상업관광": "{:.2f}".format,
                                  "쉼터거리": "{:.2f}".format}))
    for c in sorted(d["군집"].unique()):
        print(f"   군집{c} 대표: " + ", ".join(d[d.군집 == c].nlargest(5, "노출_폭염한낮")["동명"]))
    return p
print("\n=== 군집 프로파일 ===")
profile(lab2, 2)
p3 = profile(lab3, 3)

# 타깃(고노출·비탄력) = 최고 노출 군집
tgt = df.assign(c=lab3).groupby("c")["노출_폭염한낮"].mean().idxmax()
print(f"\n>>> 신규 쉼터 수요 핵심 = 고노출·비탄력형 (k=3 군집{tgt}) ; 자치구 분포:")
print(df.assign(c=lab3).query("c == @tgt")["자치구"].value_counts().head(8).to_string())

# ── 5) 결과 저장 ──
pd.DataFrame({"k": list(sil), "실루엣": list(sil.values())}).to_csv(
    HERE / "군집_결과.csv", index=False, encoding="utf-8-sig")

# ── 6) 그림 2패널: (a) k별 실루엣 (b) 노출-회피 산점(3유형) ──
TGT_COLOR = {0: "#9aa0a6", 1: "#9aa0a6", 2: "#9aa0a6"}; TGT_COLOR[tgt] = "#0f9d8a"
NAMES = {tgt: "고노출·비탄력형(쉼터 핵심)"}
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
# (a) 실루엣 곡선
ks = list(sil); vs = list(sil.values())
ax[0].plot(ks, vs, "-o", color="#0f9d8a", lw=2)
for k, v in sil.items(): ax[0].annotate(f"{v:.3f}", (k, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
ax[0].axhline(0.4, color="#ff7f0e", ls="--", lw=1); ax[0].axhline(0.5, color="#2ca02c", ls="--", lw=1)
ax[0].set_xlabel("군집 수 k"); ax[0].set_ylabel("실루엣"); ax[0].set_xticks(ks)
ax[0].set_title("(a) 군집 수에 따른 실루엣 (4피처·원공간)", fontsize=11)
# (b) 노출-회피 산점 (실제 단위)
for c in sorted(set(lab3)):
    m = lab3 == c
    ax[1].scatter(df.loc[m, "노출_폭염한낮"], df.loc[m, "avoid"], s=26, alpha=0.75,
                  color=TGT_COLOR[c], label=NAMES.get(c, f"군집{c} (n={int(m.sum())})"))
ax[1].axhline(0, color="#888", ls=":", lw=1)
ax[1].set_xlabel("노출 — 폭염 한낮 단기 외국인 도착(명)"); ax[1].set_ylabel("회피율(%) — 0 근방=비탄력")
ax[1].set_title(f"(b) 노출-회피로 본 3유형 (k=3 실루엣 {sil[3]:.3f})", fontsize=11)
ax[1].legend(fontsize=8, loc="lower right")
fig.suptitle("행정동 폭염 적응 군집 — 최종 4피처·원공간", fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(HERE / "군집_그림.png", dpi=140); plt.close()

print("\n저장: 군집_결과.csv, 군집_그림.png")
print("=== 완료 ===")
