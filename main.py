import io
import unicodedata
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# =========================
# App Config
# =========================
st.set_page_config(page_title="🌱 극지식물 최적 EC 농도 연구", layout="wide")

# Korean font (Streamlit UI)
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)

PLOTLY_FONT = "Malgun Gothic, Apple SD Gothic Neo, Noto Sans KR, sans-serif"

SCHOOLS = ["송도고", "하늘고", "아라고", "동산고"]

TARGET_EC = {
    "송도고": 1.0,
    "하늘고": 2.0,  # 최적(가정/기대)
    "아라고": 4.0,
    "동산고": 8.0,
}

SCHOOL_COLOR = {
    "송도고": "#1f77b4",
    "하늘고": "#ff7f0e",
    "아라고": "#2ca02c",
    "동산고": "#d62728",
}


# =========================
# Helpers (NFC/NFD-safe)
# =========================
def _norm_variants(text: str) -> set[str]:
    return {
        unicodedata.normalize("NFC", text),
        unicodedata.normalize("NFD", text),
    }


def _simplify(text: str) -> str:
    """
    파일명 매칭 실패 방지:
    - NFC/NFD 정규화
    - 공백/언더바/하이픈/점/괄호 등 구분자 제거
    """
    t = unicodedata.normalize("NFC", str(text))
    for ch in [" ", "_", "-", ".", "(", ")", "[", "]", "{", "}", "　"]:
        t = t.replace(ch, "")
    return t


def _contains_all_tokens(name: str, tokens: list[str]) -> bool:
    """
    NFC/NFD + 구분자 제거 기반으로 토큰 포함 여부 판정
    """
    name_variants = {_simplify(v) for v in _norm_variants(name)}
    token_variants_list = [{_simplify(v) for v in _norm_variants(t)} for t in tokens]

    for token_variants in token_variants_list:
        if not any(any(tok in nm for tok in token_variants) for nm in name_variants):
            return False
    return True


def _pick_file_by_tokens(data_dir: Path, required_tokens: list[str], allowed_suffixes: set[str]) -> Path | None:
    """
    Must use Path.iterdir().
    No f-string filename composition.
    No glob-only approach.
    NFC/NFD bidirectional check.
    """
    if not data_dir.exists():
        return None

    for p in data_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in allowed_suffixes:
            continue
        if _contains_all_tokens(p.name, required_tokens):
            return p
    return None


def _pick_csv_for_school(data_dir: Path, school: str) -> Path | None:
    # tokens: 학교명 + 환경데이터 + .csv
    return _pick_file_by_tokens(
        data_dir=data_dir,
        required_tokens=[school, "환경데이터"],
        allowed_suffixes={".csv"},
    )


def _pick_growth_xlsx(data_dir: Path) -> Path | None:
    # tokens: 생육결과데이터 + .xlsx
    return _pick_file_by_tokens(
        data_dir=data_dir,
        required_tokens=["생육결과데이터"],
        allowed_suffixes={".xlsx"},
    )


def _resolve_data_dir() -> Path:
    """
    Streamlit Cloud/로컬 모두 안전:
    - main.py 위치 기준
    - 현재 작업 디렉토리 기준
    - 상위 폴더를 거슬러 올라가며 data 탐색
    """
    candidates = []

    # 1) main.py 기준
    try:
        here = Path(__file__).resolve().parent
        candidates.append(here / "data")
        candidates.append(here.parent / "data")
    except Exception:
        pass

    # 2) cwd 기준
    candidates.append(Path.cwd() / "data")

    # 3) 상위로 올라가며 data 탐색 (최대 4단계)
    p = Path.cwd().resolve()
    for _ in range(4):
        candidates.append(p / "data")
        p = p.parent

    for c in candidates:
        if c.exists() and c.is_dir():
            return c

    # 없으면 기본값(에러 메시지용)
    return Path.cwd() / "data"


# =========================
# Data Loading
# =========================
def _standardize_env_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    colmap = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in {"time", "datetime", "date", "timestamp"}:
            colmap[c] = "time"
        elif "temp" in cl or "temperature" in cl or "온도" in cl:
            colmap[c] = "temperature"
        elif "humid" in cl or "humidity" in cl or "습도" in cl:
            colmap[c] = "humidity"
        elif cl == "ph" or "산도" in cl:
            colmap[c] = "ph"
        elif cl == "ec" or "전기전도" in cl:
            colmap[c] = "ec"

    df = df.rename(columns=colmap)

    required = {"time", "temperature", "humidity", "ph", "ec"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"환경 데이터 필수 컬럼 누락: {sorted(missing)}")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    for c in ["temperature", "humidity", "ph", "ec"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _standardize_growth_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    def find_col(keys: list[str]) -> str | None:
        for c in df.columns:
            for k in keys:
                if k in str(c).replace(" ", ""):
                    return c
        return None

    col_id = find_col(["개체번호", "개체", "번호"])
    col_leaf = find_col(["잎수", "잎수(장)", "잎"])
    col_shoot = find_col(["지상부길이", "지상부", "지상부길이(mm)"])
    col_root = find_col(["지하부길이", "지하부", "지하부길이(mm)"])
    col_w = find_col(["생중량", "생중량(g)", "중량", "무게"])

    mapping = {}
    if col_id:
        mapping[col_id] = "id"
    if col_leaf:
        mapping[col_leaf] = "leaf_count"
    if col_shoot:
        mapping[col_shoot] = "shoot_len_mm"
    if col_root:
        mapping[col_root] = "root_len_mm"
    if col_w:
        mapping[col_w] = "fresh_weight_g"

    df = df.rename(columns=mapping)

    required = {"id", "leaf_count", "shoot_len_mm", "fresh_weight_g"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"생육 결과 필수 컬럼 누락/인식 실패: {sorted(missing)}")

    for c in ["leaf_count", "shoot_len_mm", "root_len_mm", "fresh_weight_g"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


@st.cache_data(show_spinner=False)
def load_environment_data(data_dir_str: str) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir_str)
    env: dict[str, pd.DataFrame] = {}

    for school in SCHOOLS:
        p = _pick_csv_for_school(data_dir, school)
        if p is None:
            env[school] = pd.DataFrame()
            continue

        last_err = None
        for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
            try:
                df = pd.read_csv(p, encoding=enc)
                df = _standardize_env_df(df)
                df["school"] = school
                env[school] = df
                last_err = None
                break
            except Exception as e:
                last_err = e

        if last_err is not None:
            env[school] = pd.DataFrame()

    return env


@st.cache_data(show_spinner=False)
def load_growth_data(data_dir_str: str) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir_str)
    xlsx_path = _pick_growth_xlsx(data_dir)
    if xlsx_path is None:
        return {}

    xls = pd.ExcelFile(xlsx_path, engine="openpyxl")
    sheets = xls.sheet_names

    out: dict[str, pd.DataFrame] = {}
    for sheet in sheets:
        raw = pd.read_excel(xlsx_path, sheet_name=sheet, engine="openpyxl")
        if raw is None or raw.empty:
            continue

        matched_school = None
        for s in SCHOOLS:
            if _contains_all_tokens(sheet, [s]):
                matched_school = s
                break

        label = matched_school if matched_school else sheet

        df = _standardize_growth_df(raw)
        df["school"] = label
        out[label] = df

    return out


def _safe_concat(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    dfs2 = [d for d in dfs if d is not None and not d.empty]
    if not dfs2:
        return pd.DataFrame()
    return pd.concat(dfs2, ignore_index=True)


def _plotly_layout(fig: go.Figure, title: str | None = None) -> go.Figure:
    fig.update_layout(
        title=title,
        font=dict(family=PLOTLY_FONT),
        legend_title_text="",
        margin=dict(l=20, r=20, t=60 if title else 30, b=20),
    )
    return fig


# =========================
# Load Data
# =========================
DATA_DIR = _resolve_data_dir()

# 디버그: 실제 data 폴더가 어디로 잡혔는지 + 내부 파일 표시
st.caption(f"📁 data 폴더 경로: {DATA_DIR}")
if not DATA_DIR.exists():
    st.error("data/ 폴더를 찾지 못했습니다. 리포지토리 루트에 data 폴더가 있는지 확인하세요.")
else:
    files = [p.name for p in DATA_DIR.iterdir() if p.is_file()]
    st.caption(f"📄 탐지된 파일: {', '.join(files) if files else '(없음)'}")

with st.spinner("데이터를 불러오는 중..."):
    env_by_school = load_environment_data(str(DATA_DIR))
    growth_by_school = load_growth_data(str(DATA_DIR))

env_all = _safe_concat([env_by_school.get(s, pd.DataFrame()) for s in SCHOOLS])
growth_all = _safe_concat([growth_by_school.get(k, pd.DataFrame()) for k in growth_by_school.keys()])

if env_all.empty:
    st.error("환경 데이터(CSV)를 찾지 못했거나 읽을 수 없습니다. data/ 폴더와 파일명을 확인하세요.")
if not growth_by_school:
    st.error("생육 결과 데이터(XLSX)를 찾지 못했거나 읽을 수 없습니다. data/ 폴더와 파일명을 확인하세요.")


# =========================
# Sidebar
# =========================
st.title("🌱 극지식물 최적 EC 농도 연구")

sel_school = st.sidebar.selectbox("학교 선택", ["전체"] + SCHOOLS, index=0)
selected_schools = SCHOOLS if sel_school == "전체" else [sel_school]


def get_selected_env() -> pd.DataFrame:
    return _safe_concat([env_by_school.get(s, pd.DataFrame()) for s in selected_schools])


def get_selected_growth() -> pd.DataFrame:
    dfs = []
    for s in selected_schools:
        if s in growth_by_school:
            dfs.append(growth_by_school[s])
    return _safe_concat(dfs)


# =========================
# Tabs
# =========================
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

# -------------------------
# Tab 1: Overview
# -------------------------
with tab1:
    st.subheader("연구 배경 및 목적")
    st.write(
        """
본 대시보드는 4개 학교에서 서로 다른 EC(전기전도도) 조건으로 극지식물을 재배한 데이터를 통합하여,
(1) 학교별 환경(온도/습도/pH/EC) 특성을 비교하고, (2) EC 조건별 생육(생중량/잎 수/길이)을 정량 비교하여,
(3) 최적 EC 농도를 도출하는 것을 목표로 한다.
"""
    )

    rows = []
    for s in SCHOOLS:
        n = int(growth_by_school.get(s, pd.DataFrame()).shape[0]) if s in growth_by_school else 0
        rows.append({"학교명": s, "EC 목표": TARGET_EC.get(s, None), "개체수": n, "색상": SCHOOL_COLOR.get(s, "#999999")})
    cond_df = pd.DataFrame(rows)

    st.markdown("#### 학교별 EC 조건")
    st.dataframe(cond_df, use_container_width=True, hide_index=True)

    env_sel = get_selected_env()
    growth_sel = get_selected_growth()

    total_n = int(growth_sel.shape[0]) if not growth_sel.empty else 0
    avg_temp = float(env_sel["temperature"].mean()) if not env_sel.empty else float("nan")
    avg_hum = float(env_sel["humidity"].mean()) if not env_sel.empty else float("nan")

    best_ec = None
    if not growth_all.empty:
        tmp = growth_all.copy()
        tmp = tmp[tmp["school"].isin(SCHOOLS)]
        if not tmp.empty and "fresh_weight_g" in tmp.columns:
            mean_w = tmp.groupby("school", as_index=False)["fresh_weight_g"].mean()
            mean_w["target_ec"] = mean_w["school"].map(TARGET_EC)
            mean_w = mean_w.dropna(subset=["target_ec"])
            if not mean_w.empty:
                best_row = mean_w.sort_values("fresh_weight_g", ascending=False).iloc[0]
                best_ec = float(best_row["target_ec"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 개체수", f"{total_n:,}")
    c2.metric("평균 온도(°C)", "-" if env_sel.empty else f"{avg_temp:.2f}")
    c3.metric("평균 습도(%)", "-" if env_sel.empty else f"{avg_hum:.2f}")
    c4.metric("도출된 최적 EC", "-" if best_ec is None else f"{best_ec:.1f}")

# -------------------------
# Tab 2: Environment
# -------------------------
with tab2:
    st.subheader("학교별 환경 데이터 비교")

    env_sel = get_selected_env()

    if env_all.empty:
        st.error("환경 데이터가 없어 시각화를 진행할 수 없습니다.")
    else:
        env_avg = (
            env_all.groupby("school", as_index=False)[["temperature", "humidity", "ph", "ec"]]
            .mean()
            .sort_values("school")
        )
        env_avg["target_ec"] = env_avg["school"].map(TARGET_EC)

        fig = make_subplots(rows=2, cols=2, subplot_titles=("평균 온도", "평균 습도", "평균 pH", "목표 EC vs 실측 EC(평균)"))

        fig.add_trace(go.Bar(x=env_avg["school"], y=env_avg["temperature"], name="온도"), row=1, col=1)
        fig.add_trace(go.Bar(x=env_avg["school"], y=env_avg["humidity"], name="습도"), row=1, col=2)
        fig.add_trace(go.Bar(x=env_avg["school"], y=env_avg["ph"], name="pH"), row=2, col=1)
        fig.add_trace(go.Bar(x=env_avg["school"], y=env_avg["target_ec"], name="목표 EC"), row=2, col=2)
        fig.add_trace(go.Bar(x=env_avg["school"], y=env_avg["ec"], name="실측 EC(평균)"), row=2, col=2)

        fig.update_layout(barmode="group")
        fig = _plotly_layout(fig, "학교별 환경 평균 비교(2x2)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 선택한 학교 시계열")

        def _timeseries_fig(metric: str, title: str, add_target_ec: bool = False) -> go.Figure:
            base = env_all if sel_school == "전체" else env_sel
            if base.empty:
                return go.Figure()

            fig_ts = go.Figure()
            for s in (SCHOOLS if sel_school == "전체" else [sel_school]):
                d = env_by_school.get(s, pd.DataFrame())
                if d is None or d.empty:
                    continue
                fig_ts.add_trace(go.Scatter(x=d["time"], y=d[metric], mode="lines", name=s))

            if add_target_ec and sel_school != "전체":
                t = TARGET_EC.get(sel_school, None)
                if t is not None:
                    fig_ts.add_hline(y=float(t), line_dash="dash", annotation_text="목표 EC", annotation_position="top left")

            fig_ts = _plotly_layout(fig_ts, title)
            fig_ts.update_xaxes(title_text="time")
            fig_ts.update_yaxes(title_text=metric)
            return fig_ts

        colA, colB, colC = st.columns(3)
        with colA:
            fig_t = _timeseries_fig("temperature", "온도 변화")
            if fig_t.data:
                st.plotly_chart(fig_t, use_container_width=True)
            else:
                st.error("선택 범위에 해당하는 온도 시계열 데이터가 없습니다.")
        with colB:
            fig_h = _timeseries_fig("humidity", "습도 변화")
            if fig_h.data:
                st.plotly_chart(fig_h, use_container_width=True)
            else:
                st.error("선택 범위에 해당하는 습도 시계열 데이터가 없습니다.")
        with colC:
            fig_e = _timeseries_fig("ec", "EC 변화 (목표 EC 수평선 포함)", add_target_ec=True)
            if fig_e.data:
                st.plotly_chart(fig_e, use_container_width=True)
            else:
                st.error("선택 범위에 해당하는 EC 시계열 데이터가 없습니다.")

        with st.expander("환경 데이터 원본 테이블 + CSV 다운로드"):
            show_df = env_sel if sel_school != "전체" else env_all
            if show_df.empty:
                st.error("표시할 환경 데이터가 없습니다.")
            else:
                st.dataframe(show_df.sort_values(["school", "time"]), use_container_width=True, hide_index=True)
                csv_bytes = show_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button("CSV 다운로드", data=csv_bytes, file_name="환경데이터_선택범위.csv", mime="text/csv")

# -------------------------
# Tab 3: Growth
# -------------------------
with tab3:
    st.subheader("EC별 생육 결과 비교")

    if growth_all.empty:
        st.error("생육 결과 데이터가 없어 분석을 진행할 수 없습니다.")
    else:
        g = growth_all.copy()
        g = g[g["school"].isin(SCHOOLS)].copy()

        if g.empty:
            st.error("생육 결과에서 학교 매칭에 실패했습니다. XLSX 시트명에 학교명이 포함되어 있는지 확인하세요.")
        else:
            g["target_ec"] = g["school"].map(TARGET_EC)

            summary = (
                g.groupby(["school", "target_ec"], as_index=False)
                .agg(
                    mean_weight=("fresh_weight_g", "mean"),
                    mean_leaf=("leaf_count", "mean"),
                    mean_shoot=("shoot_len_mm", "mean"),
                    count=("id", "count"),
                )
                .sort_values("target_ec")
            )

            best = summary.sort_values("mean_weight", ascending=False).iloc[0]
            best_school = str(best["school"])
            best_ec_val = float(best["target_ec"])
            best_w = float(best["mean_weight"])

            note = "⭐ 최댓값" if best_school == "하늘고" else "최댓값"
            st.markdown("### 🥇 핵심 결과")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("최대 평균 생중량(EC)", f"{best_w:.3f} g", delta=f"{best_ec_val:.1f}")
            c2.metric("최대 평균 생중량 학교", best_school, delta=note)
            exp_opt = TARGET_EC.get("하늘고", None)
            c3.metric("가정/조건상 최적 EC(하늘고)", "-" if exp_opt is None else f"{exp_opt:.1f}")
            c4.metric("분석 포함 개체수(4개교 합)", f"{int(g.shape[0]):,}")

            fig2 = make_subplots(rows=2, cols=2, subplot_titles=("평균 생중량(⭐)", "평균 잎 수", "평균 지상부 길이(mm)", "개체수 비교"))

            fig2.add_trace(go.Bar(x=summary["target_ec"], y=summary["mean_weight"], name="평균 생중량"), row=1, col=1)
            fig2.add_trace(go.Bar(x=summary["target_ec"], y=summary["mean_leaf"], name="평균 잎 수"), row=1, col=2)
            fig2.add_trace(go.Bar(x=summary["target_ec"], y=summary["mean_shoot"], name="평균 지상부 길이"), row=2, col=1)
            fig2.add_trace(go.Bar(x=summary["target_ec"], y=summary["count"], name="개체수"), row=2, col=2)

            fig2.add_vline(x=best_ec_val, line_dash="dash", annotation_text="최적(평균 생중량 최대)", annotation_position="top left", row=1, col=1)

            fig2.update_layout(barmode="group")
            fig2 = _plotly_layout(fig2, "EC별 생육 지표 비교(2x2)")
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown("#### 학교별 생중량 분포")
            fig_box = px.box(g, x="school", y="fresh_weight_g", points="outliers", title="학교별 생중량 분포(박스플롯)")
            fig_box = _plotly_layout(fig_box)
            st.plotly_chart(fig_box, use_container_width=True)

            st.markdown("#### 상관관계 분석")
            cc1, cc2 = st.columns(2)

            with cc1:
                fig_sc1 = px.scatter(g, x="leaf_count", y="fresh_weight_g", color="school", title="잎 수 vs 생중량")
                fig_sc1 = _plotly_layout(fig_sc1)
                st.plotly_chart(fig_sc1, use_container_width=True)

            with cc2:
                fig_sc2 = px.scatter(g, x="shoot_len_mm", y="fresh_weight_g", color="school", title="지상부 길이 vs 생중량")
                fig_sc2 = _plotly_layout(fig_sc2)
                st.plotly_chart(fig_sc2, use_container_width=True)

            with st.expander("학교별 생육 데이터 원본 + XLSX 다운로드"):
                g_sel = get_selected_growth()
                if sel_school == "전체":
                    show_g = g.sort_values(["school", "id"])
                else:
                    show_g = g_sel.sort_values(["school", "id"]) if not g_sel.empty else pd.DataFrame()

                if show_g.empty:
                    st.error("표시할 생육 데이터가 없습니다. (선택 학교의 시트 매칭/데이터를 확인하세요)")
                else:
                    st.dataframe(show_g, use_container_width=True, hide_index=True)

                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                    if sel_school == "전체":
                        for s in SCHOOLS:
                            df_s = g[g["school"] == s].copy()
                            if not df_s.empty:
                                df_s.to_excel(writer, index=False, sheet_name=s)
                    else:
                        show_g.to_excel(writer, index=False, sheet_name=sel_school)

                buffer.seek(0)
                st.download_button(
                    "XLSX 다운로드",
                    data=buffer,
                    file_name="생육결과_선택범위.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
