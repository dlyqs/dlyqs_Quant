import os
import sys
import re
import time
import datetime as dt
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# 需要：akshare, pandas, sqlalchemy, psycopg2-binary, python-dotenv, openpyxl
import akshare as ak

load_dotenv()
DB_URL = os.getenv("DB_URL")
if not DB_URL:
    print("请在 .env 里设置 DB_URL")
    sys.exit(1)

engine = create_engine(DB_URL, future=True)

# ============ 规范化与校验 ============
DIGITS6 = re.compile(r"^\d{1,6}$")

def fix_code_to_6digits(raw: str) -> str | None:
    s = str(raw).strip().upper().replace('.SZSE','').replace('.SSE','')
    if '.' in s and len(s.split('.')[0]) == 6:
        code6, suf = s.split('.', 1)
        if not code6.isdigit(): return None
        if suf not in ('SH','SZ','BJ'): return None
        return f"{code6}.{suf}"
    digits = re.sub(r"\D", "", s)
    if not digits or not DIGITS6.match(digits):
        return None
    code6 = digits.zfill(6)
    if code6[0] == '6': return f"{code6}.SH"
    if code6[0] in ('0','3'): return f"{code6}.SZ"
    if code6[0] in ('8','4'): return f"{code6}.BJ"
    return None

def ak_symbol(sec_code: str) -> str:
    return sec_code.split('.')[0]

def parse_date(s: str) -> dt.date:
    s = str(s).strip().replace('/','-').replace('.','-')
    return dt.datetime.strptime(s, "%Y-%m-%d").date()

# ============ 建表 ============
DDL = """
CREATE TABLE IF NOT EXISTS ref_list (
  sec_code      VARCHAR(12) PRIMARY KEY,
  sec_name      VARCHAR(64),
  pick_dates    TEXT NOT NULL,   -- 记录每段的“首个入选日”，多个段用逗号分隔
  streaks       TEXT,            -- 与 pick_dates 对应的段内连续天数
  updated_at    TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS stock_info (
  sec_code      VARCHAR(12) PRIMARY KEY,
  sec_name      VARCHAR(64) NOT NULL,
  float_mktcap_100m NUMERIC(20,2),
  updated_at    TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS stock_window (
  sec_code      VARCHAR(12) NOT NULL,
  pick_date     DATE NOT NULL,
  o_m3 NUMERIC(18,4), c_m3 NUMERIC(18,4), pctc_m3 NUMERIC(10,4), pcto_m3 NUMERIC(10,4), amp_m3 NUMERIC(10,4), turn_m3 NUMERIC(10,4), amt_m3 NUMERIC(20,2),
  o_m2 NUMERIC(18,4), c_m2 NUMERIC(18,4), pctc_m2 NUMERIC(10,4), pcto_m2 NUMERIC(10,4), amp_m2 NUMERIC(10,4), turn_m2 NUMERIC(10,4), amt_m2 NUMERIC(20,2),
  o_m1 NUMERIC(18,4), c_m1 NUMERIC(18,4), pctc_m1 NUMERIC(10,4), pcto_m1 NUMERIC(10,4), amp_m1 NUMERIC(10,4), turn_m1 NUMERIC(10,4), amt_m1 NUMERIC(20,2),
  o_d0 NUMERIC(18,4), c_d0 NUMERIC(18,4), pctc_d0 NUMERIC(10,4), pcto_d0 NUMERIC(10,4), amp_d0 NUMERIC(10,4), amt_d0 NUMERIC(20,2), turn_d0 NUMERIC(10,4),
  o_p1 NUMERIC(18,4), c_p1 NUMERIC(18,4), pctc_p1 NUMERIC(10,4), pcto_p1 NUMERIC(10,4), amp_p1 NUMERIC(10,4), turn_p1 NUMERIC(10,4), amt_p1 NUMERIC(20,2),
  o_p2 NUMERIC(18,4), c_p2 NUMERIC(18,4), pctc_p2 NUMERIC(10,4), pcto_p2 NUMERIC(10,4), amp_p2 NUMERIC(10,4), turn_p2 NUMERIC(10,4), amt_p2 NUMERIC(20,2),
  o_p3 NUMERIC(18,4), c_p3 NUMERIC(18,4), pctc_p3 NUMERIC(10,4), pcto_p3 NUMERIC(10,4), amp_p3 NUMERIC(10,4), turn_p3 NUMERIC(10,4), amt_p3 NUMERIC(20,2),
  updated_at    TIMESTAMP DEFAULT NOW(),
  PRIMARY KEY (sec_code, pick_date)
);

CREATE INDEX IF NOT EXISTS idx_window_date ON stock_window(pick_date);
"""

def ensure_tables():
    with engine.begin() as conn:
        conn.execute(text(DDL))

# ============ 读取 Excel ============
def read_picks(excel_path: str) -> pd.DataFrame:
    if excel_path.lower().endswith('.csv'):
        df = pd.read_csv(excel_path)
    else:
        df = pd.read_excel(excel_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if not {'pick_date','code'}.issubset(df.columns):
        raise ValueError("Excel/CSV 需包含列：pick_date, code")
    df['sec_code'] = df['code'].apply(fix_code_to_6digits)
    df['pick_date'] = df['pick_date'].apply(parse_date)
    bad = df[df['sec_code'].isna()]
    if not bad.empty:
        print("⚠️ 发现无效代码（已跳过）条数：", len(bad))
        print("示例：", bad.head(10).to_dict(orient='records'))
    df = df[df['sec_code'].notna()]
    df = df[['sec_code','pick_date']].drop_duplicates().sort_values(['sec_code','pick_date'])
    return df

# ============ 分段 ============
def episodes_from_dates(dates: list[dt.date]) -> tuple[list[dt.date], list[int]]:
    if not dates: return [], []
    starts, streaks = [], []
    cur_start, cur_len = dates[0], 1
    prev = dates[0]
    for d in dates[1:]:
        if (d - prev).days <= 7:
            cur_len += 1
        else:
            starts.append(cur_start); streaks.append(cur_len)
            cur_start, cur_len = d, 1
        prev = d
    starts.append(cur_start); streaks.append(cur_len)
    return starts, streaks

def build_ref_list(df_picks: pd.DataFrame) -> pd.DataFrame:
    out = []
    for sec, g in df_picks.groupby('sec_code'):
        ds = sorted(g['pick_date'].tolist())
        starts, streaks = episodes_from_dates(ds)
        out.append({
            'sec_code': sec,
            'sec_name': None,
            'pick_dates': ','.join(d.strftime('%Y-%m-%d') for d in starts),
            'streaks': ','.join(str(x) for x in streaks)
        })
    return pd.DataFrame(out)

# ============ 名称/市值（带重试） ============
def fetch_name_mktcap(sym: str, max_retry: int = 3, sleep_sec: float = 0.6):
    last_err = None
    for _ in range(max_retry):
        try:
            info = ak.stock_individual_info_em(sym)
            if info is not None and isinstance(info, pd.DataFrame) and not info.empty:
                if info.shape[1] >= 2:
                    info = info.iloc[:, :2].copy()
                    info.columns = ['item','value']
                    kv = dict(zip(info['item'], info['value']))
                    name = kv.get('证券简称') or kv.get('股票简称') or kv.get('简称')
                    v = kv.get('流通市值(元)') or kv.get('流通市值')
                    if isinstance(v, str):
                        v = float(v.replace(',', '').replace('元','').strip() or 0)
                    cap100m = (v/1e8) if v else None
                    return name, cap100m
            return None, None
        except Exception as e:
            last_err = e
            time.sleep(sleep_sec)
    if last_err:
        print(f"[{sym}] 获取名称/市值失败：{last_err}")
    return None, None

def fill_names_and_mktcap(ref_df: pd.DataFrame) -> pd.DataFrame:
    names, caps = [], []
    for sec in ref_df['sec_code']:
        name, cap = fetch_name_mktcap(ak_symbol(sec))
        names.append(name); caps.append(cap)
    ref_df = ref_df.copy()
    ref_df['sec_name'] = names
    with engine.begin() as conn:
        upsert_b = text("""
            INSERT INTO stock_info (sec_code, sec_name, float_mktcap_100m)
            VALUES (:sec, :name, :cap)
            ON CONFLICT (sec_code) DO UPDATE
            SET sec_name = COALESCE(EXCLUDED.sec_name, stock_info.sec_name),
                float_mktcap_100m = COALESCE(EXCLUDED.float_mktcap_100m, stock_info.float_mktcap_100m),
                updated_at = NOW();
        """)
        for sec, name, cap in zip(ref_df['sec_code'], ref_df['sec_name'], caps):
            conn.execute(upsert_b, {'sec': sec, 'name': name or '', 'cap': cap})
    return ref_df

def save_ref_list(ref_df: pd.DataFrame):
    with engine.begin() as conn:
        upsert = text("""
            INSERT INTO ref_list (sec_code, sec_name, pick_dates, streaks)
            VALUES (:sec, :name, :dates, :streaks)
            ON CONFLICT (sec_code) DO UPDATE
            SET sec_name = COALESCE(EXCLUDED.sec_name, ref_list.sec_name),
                pick_dates = EXCLUDED.pick_dates,
                streaks = EXCLUDED.streaks,
                updated_at = NOW();
        """)
        for _, r in ref_df.iterrows():
            conn.execute(upsert, {
                'sec': r['sec_code'],
                'name': r['sec_name'] if pd.notna(r['sec_name']) else None,
                'dates': r['pick_dates'],
                'streaks': r['streaks']
            })
    print(f"✅ 已写入 A表 ref_list：{len(ref_df)} 条")

# ============ 交易日窗口：按“交易日索引”取 T-3~T+3 ============
def fetch_hist_df(sec_code: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    sym = ak_symbol(sec_code)
    try:
        df = ak.stock_zh_a_hist(
            symbol=sym, period="daily",
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            adjust="qfq"
        )
    except Exception:
        return pd.DataFrame()
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    mapping = {
        "日期":"date","开盘":"open","最高":"high","最低":"low","收盘":"close",
        "涨跌幅":"pct_chg","振幅":"amplitude","成交量":"volume","成交额":"amount",
        "换手率":"turnover_rate","量比":"volume_ratio","前收盘":"pre_close"
    }
    for cn,en in mapping.items():
        if cn in df.columns: df.rename(columns={cn:en}, inplace=True)
    if "date" not in df.columns and "日期" in df.columns:
        df.rename(columns={"日期":"date"}, inplace=True)

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df[pd.notna(df["date"])].sort_values("date")
    if df.empty: return pd.DataFrame()

    if "pre_close" not in df.columns or df["pre_close"].isna().all():
        df["pre_close"] = df["close"].shift(1)
    if "pct_chg" not in df.columns or df["pct_chg"].isna().all():
        df["pct_chg"] = (df["close"] / df["pre_close"] - 1.0) * 100
    if "amplitude" not in df.columns or df["amplitude"].isna().all():
        df["amplitude"] = (df["high"] - df["low"]) / df["pre_close"] * 100
    df["pcto"] = (df["open"] / df["pre_close"] - 1.0) * 100
    return df

def pick_by_trade_offset(hist: pd.DataFrame, t: dt.date, delta: int):
    """
    在 hist 的交易日序列里：找到日期 t 的索引 i，返回 i+delta 那天的指标。
    若 t 不在 hist（极少见，或 Excel 给了休市日），则返回空。
    """
    dates = hist["date"].tolist()
    try:
        i = dates.index(t)
    except ValueError:
        return (None,)*7
    j = i + delta
    if j < 0 or j >= len(dates):
        return (None,)*7
    r = hist.iloc[j]
    return (
        float(r.get('open')) if pd.notna(r.get('open')) else None,
        float(r.get('close')) if pd.notna(r.get('close')) else None,
        float(r.get('pct_chg')) if pd.notna(r.get('pct_chg')) else None,
        float(r.get('pcto')) if pd.notna(r.get('pcto')) else None,
        float(r.get('amplitude')) if pd.notna(r.get('amplitude')) else None,
        float(r.get('turnover_rate')) if pd.notna(r.get('turnover_rate')) else None,
        float(r.get('amount')) if pd.notna(r.get('amount')) else None,
    )

def fill_stock_window_from_ref(ref_df: pd.DataFrame):
    with engine.begin() as conn:
        upsert = text("""
            INSERT INTO stock_window (
              sec_code, pick_date,
              o_m3,c_m3,pctc_m3,pcto_m3,amp_m3,turn_m3,amt_m3,
              o_m2,c_m2,pctc_m2,pcto_m2,amp_m2,turn_m2,amt_m2,
              o_m1,c_m1,pctc_m1,pcto_m1,amp_m1,turn_m1,amt_m1,
              o_d0,c_d0,pctc_d0,pcto_d0,amp_d0,amt_d0,turn_d0,
              o_p1,c_p1,pctc_p1,pcto_p1,amp_p1,turn_p1,amt_p1,
              o_p2,c_p2,pctc_p2,pcto_p2,amp_p2,turn_p2,amt_p2,
              o_p3,c_p3,pctc_p3,pcto_p3,amp_p3,turn_p3,amt_p3
            ) VALUES (
              :sec, :t,
              :o_m3,:c_m3,:pctc_m3,:pcto_m3,:amp_m3,:turn_m3,:amt_m3,
              :o_m2,:c_m2,:pctc_m2,:pcto_m2,:amp_m2,:turn_m2,:amt_m2,
              :o_m1,:c_m1,:pctc_m1,:pcto_m1,:amp_m1,:turn_m1,:amt_m1,
              :o_d0,:c_d0,:pctc_d0,:pcto_d0,:amp_d0,:amt_d0,:turn_d0,
              :o_p1,:c_p1,:pctc_p1,:pcto_p1,:amp_p1,:turn_p1,:amt_p1,
              :o_p2,:c_p2,:pctc_p2,:pcto_p2,:amp_p2,:turn_p2,:amt_p2,
              :o_p3,:c_p3,:pctc_p3,:pcto_p3,:amp_p3,:turn_p3,:amt_p3
            )
            ON CONFLICT (sec_code, pick_date) DO UPDATE SET
              o_m3=EXCLUDED.o_m3, c_m3=EXCLUDED.c_m3, pctc_m3=EXCLUDED.pctc_m3, pcto_m3=EXCLUDED.pcto_m3, amp_m3=EXCLUDED.amp_m3, turn_m3=EXCLUDED.turn_m3, amt_m3=EXCLUDED.amt_m3,
              o_m2=EXCLUDED.o_m2, c_m2=EXCLUDED.c_m2, pctc_m2=EXCLUDED.pctc_m2, pcto_m2=EXCLUDED.pcto_m2, amp_m2=EXCLUDED.amp_m2, turn_m2=EXCLUDED.turn_m2, amt_m2=EXCLUDED.amt_m2,
              o_m1=EXCLUDED.o_m1, c_m1=EXCLUDED.c_m1, pctc_m1=EXCLUDED.pctc_m1, pcto_m1=EXCLUDED.pcto_m1, amp_m1=EXCLUDED.amp_m1, turn_m1=EXCLUDED.turn_m1, amt_m1=EXCLUDED.amt_m1,
              o_d0=EXCLUDED.o_d0, c_d0=EXCLUDED.c_d0, pctc_d0=EXCLUDED.pctc_d0, pcto_d0=EXCLUDED.pcto_d0, amp_d0=EXCLUDED.amp_d0, amt_d0=EXCLUDED.amt_d0, turn_d0=EXCLUDED.turn_d0,
              o_p1=EXCLUDED.o_p1, c_p1=EXCLUDED.c_p1, pctc_p1=EXCLUDED.pctc_p1, pcto_p1=EXCLUDED.pcto_p1, amp_p1=EXCLUDED.amp_p1, turn_p1=EXCLUDED.turn_p1, amt_p1=EXCLUDED.amt_p1,
              o_p2=EXCLUDED.o_p2, c_p2=EXCLUDED.c_p2, pctc_p2=EXCLUDED.pctc_p2, pcto_p2=EXCLUDED.pcto_p2, amp_p2=EXCLUDED.amp_p2, turn_p2=EXCLUDED.turn_p2, amt_p2=EXCLUDED.amt_p2,
              o_p3=EXCLUDED.o_p3, c_p3=EXCLUDED.c_p3, pctc_p3=EXCLUDED.pctc_p3, pcto_p3=EXCLUDED.pcto_p3, amp_p3=EXCLUDED.amp_p3, turn_p3=EXCLUDED.turn_p3, amt_p3=EXCLUDED.amt_p3,
              updated_at = NOW();
        """)

        for _, r in ref_df.iterrows():
            sec = r['sec_code']
            ep_starts = [dt.datetime.strptime(x, "%Y-%m-%d").date()
                         for x in str(r['pick_dates']).split(',') if x.strip()]
            if not ep_starts:
                continue

            start = min(ep_starts) - dt.timedelta(days=30)
            end   = max(ep_starts) + dt.timedelta(days=30)
            hist = fetch_hist_df(sec, start, end)
            if hist.empty:
                print(f"[{sec}] 无历史数据，跳过")
                continue

            for t in ep_starts:
                payload = {'sec': sec, 't': t}
                # 用“交易日偏移”方式抓 T-3~T+3（自动跳过周末/休市）
                for suf, delta in [('m3',-3),('m2',-2),('m1',-1),('d0',0),('p1',1),('p2',2),('p3',3)]:
                    o,c,pctc,pcto,amp,turn,amt = pick_by_trade_offset(hist, t, delta)
                    payload[f"o_{suf}"]=o; payload[f"c_{suf}"]=c
                    payload[f"pctc_{suf}"]=pctc; payload[f"pcto_{suf}"]=pcto
                    payload[f"amp_{suf}"]=amp; payload[f"turn_{suf}"]=turn; payload[f"amt_{suf}"]=amt
                conn.execute(upsert, payload)
    print("✅ 已写入 C表 stock_window（按交易日索引）")

# ============ 运行后校验 & 名称补齐 ============
def validate_and_retry(ref_df: pd.DataFrame):
    # Excel 去重后的股票只数
    excel_unique = ref_df['sec_code'].nunique()
    with engine.begin() as conn:
        db_unique = conn.execute(text("SELECT COUNT(*) FROM ref_list")).scalar()
    ok = (excel_unique == db_unique)
    print(f"🔎 校验 A表股票只数：Excel去重={excel_unique} vs A表={db_unique} -> {'OK' if ok else 'MISMATCH'}")

    # 查找 A表中名称为空或空串的股票
    with engine.begin() as conn:
        df_missing = pd.read_sql(
            text("SELECT sec_code FROM ref_list WHERE sec_name IS NULL OR sec_name=''"),
            conn
        )
    if df_missing.empty:
        print("🔎 A表名称列：无空数据 ✅")
        return

    print(f"🔁 A表名称列：发现 {len(df_missing)} 条为空，尝试重试拉取并回填……")
    # 重试获取并回填 A/B 表
    to_fix = df_missing['sec_code'].tolist()
    with engine.begin() as conn:
        upsert_b = text("""
            INSERT INTO stock_info (sec_code, sec_name, float_mktcap_100m)
            VALUES (:sec, :name, :cap)
            ON CONFLICT (sec_code) DO UPDATE
            SET sec_name = COALESCE(EXCLUDED.sec_name, stock_info.sec_name),
                float_mktcap_100m = COALESCE(EXCLUDED.float_mktcap_100m, stock_info.float_mktcap_100m),
                updated_at = NOW();
        """)
        update_a = text("""
            UPDATE ref_list
               SET sec_name = :name, updated_at = NOW()
             WHERE sec_code = :sec;
        """)

        for sec in to_fix:
            name, cap = fetch_name_mktcap(ak_symbol(sec), max_retry=4, sleep_sec=0.8)
            if name:
                conn.execute(upsert_b, {'sec': sec, 'name': name, 'cap': cap})
                conn.execute(update_a, {'sec': sec, 'name': name})

    # 再次确认
    with engine.begin() as conn:
        left = conn.execute(text("SELECT COUNT(*) FROM ref_list WHERE sec_name IS NULL OR sec_name=''")).scalar()
    print(f"🔎 名称补齐后剩余空值：{left} 条")

# ============ 主流程 ============
def main():
    if len(sys.argv) < 2:
        print("用法：python reset_and_load.py <你的Excel或CSV路径>")
        print("示例：python reset_and_load.py guruList.xlsx")
        sys.exit(1)

    excel_path = sys.argv[1]
    ensure_tables()

    # 读取与规范化
    df_picks = read_picks(excel_path)

    # 分段 -> A 表
    ref_df = build_ref_list(df_picks)

    # 填 B 表（名称&市值），名称回填 A 表
    ref_df = fill_names_and_mktcap(ref_df)

    # 写 A 表
    save_ref_list(ref_df)

    # 用 A 表“每段首日”生成 C 表窗口（按交易日索引）
    fill_stock_window_from_ref(ref_df)

    # 运行完成：自检 & 名称重试补齐
    validate_and_retry(ref_df)

    # 最后给个总体统计
    with engine.begin() as conn:
        n_a = conn.execute(text("SELECT COUNT(*) FROM ref_list")).scalar()
        n_b = conn.execute(text("SELECT COUNT(*) FROM stock_info")).scalar()
        n_c = conn.execute(text("SELECT COUNT(*) FROM stock_window")).scalar()
    print(f"📦 导入完成：A(ref_list)={n_a} 只，B(stock_info)={n_b} 条，C(stock_window)={n_c} 行")

if __name__ == "__main__":
    main()
