import os
import sys
import re
import time
import argparse
import datetime as dt
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from tqdm import tqdm

# 需要：akshare, pandas, sqlalchemy, psycopg2-binary, python-dotenv, openpyxl, tqdm
import akshare as ak

# ================== 环境与连接 ==================
load_dotenv()
DB_URL = os.getenv("DB_URL")
if not DB_URL:
    print("请在 .env 里设置 DB_URL")
    sys.exit(1)

engine = create_engine(DB_URL, future=True)

# ================== 工具：代码与日期 ==================
DIGITS6 = re.compile(r"^\d{1,6}$")

def fix_code_to_6digits(raw: str) -> str | None:
    """
    统一成 6 位数字：000001/300XXX/603XXX/688XXX/8XXXXX 等
    """
    s = str(raw).strip().upper().replace('.SZSE','').replace('.SSE','').replace('.SH','').replace('.SZ','').replace('.BJ','')
    digits = re.sub(r"\D", "", s)
    if not digits or not DIGITS6.match(digits):
        return None
    return digits.zfill(6)

def parse_date(s: str) -> dt.date:
    s = str(s).strip().replace('/','-').replace('.','-')
    return dt.datetime.strptime(s, "%Y-%m-%d").date()

def board_from_code(code6: str) -> str:
    """
    交易板块分类（不区分沪/深主板）：
      主板：600/601/603/605/000/001/002（含原中小板）
      创业板：300
      科创板：688
      北交所：4xxxxx/8xxxxx（常见以 83/87/88 开头，统一归“北交所”）
    """
    if code6.startswith(("300",)): return "创业板"
    if code6.startswith(("688",)): return "科创板"
    if code6[0] in ("4","8"):      return "北交所"
    if code6.startswith(("600","601","603","605","000","001","002")): return "主板"
    # 其它非常见前缀，保守按主板处理
    return "主板"

def exch_suffix(code6: str) -> str:
    """把 6 位代码映射成 .SH/.SZ/.BJ 后缀（用于行情接口 fallback）"""
    if code6.startswith('6'):       return f"{code6}.SH"
    if code6.startswith(('0','3')): return f"{code6}.SZ"
    if code6.startswith(('4','8')): return f"{code6}.BJ"
    return f"{code6}.SZ"

# ================== 建表 DDL ==================
DDL = """
CREATE TABLE IF NOT EXISTS ref_list (
  sec_code   VARCHAR(6)  PRIMARY KEY,  -- 纯6位代码
  board      VARCHAR(10),              -- 主板/创业板/科创板/北交所
  sec_name   VARCHAR(64),
  pick_dates TEXT NOT NULL,            -- 每段“首个入选日”，逗号分隔
  streaks    TEXT,                     -- 与 pick_dates 对应的段内连续天数
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS stock_info (
  sec_code            VARCHAR(6) PRIMARY KEY,
  sec_name            VARCHAR(64) NOT NULL,
  industry            VARCHAR(64),
  float_mktcap_100m   NUMERIC(20,2),
  updated_at          TIMESTAMP DEFAULT NOW()
);

-- 特征表：入选日前 T-3~-1 + 入选当日 T0 （共4天*5指标）
CREATE TABLE IF NOT EXISTS stock_pre (
  sec_code  VARCHAR(6) NOT NULL,
  pick_date DATE NOT NULL,
  pctc_m3 NUMERIC(10,4), pcto_m3 NUMERIC(10,4), amp_m3 NUMERIC(10,4), turn_m3 NUMERIC(10,4), amt_m3 NUMERIC(20,2),
  pctc_m2 NUMERIC(10,4), pcto_m2 NUMERIC(10,4), amp_m2 NUMERIC(10,4), turn_m2 NUMERIC(10,4), amt_m2 NUMERIC(20,2),
  pctc_m1 NUMERIC(10,4), pcto_m1 NUMERIC(10,4), amp_m1 NUMERIC(10,4), turn_m1 NUMERIC(10,4), amt_m1 NUMERIC(20,2),
  pctc_d0 NUMERIC(10,4), pcto_d0 NUMERIC(10,4), amp_d0 NUMERIC(10,4), turn_d0 NUMERIC(10,4), amt_d0 NUMERIC(20,2),
  updated_at TIMESTAMP DEFAULT NOW(),
  PRIMARY KEY (sec_code, pick_date)
);

CREATE TABLE IF NOT EXISTS stock_post (
  sec_code  VARCHAR(6) NOT NULL,
  pick_date DATE NOT NULL,
  pctc_p1 NUMERIC(10,4), pcto_p1 NUMERIC(10,4), amp_p1 NUMERIC(10,4),
  pctc_p2 NUMERIC(10,4), pcto_p2 NUMERIC(10,4), amp_p2 NUMERIC(10,4),
  pctc_p3 NUMERIC(10,4), pcto_p3 NUMERIC(10,4), amp_p3 NUMERIC(10,4),
  pctc_p4 NUMERIC(10,4), pcto_p4 NUMERIC(10,4), amp_p4 NUMERIC(10,4),
  pctc_p5 NUMERIC(10,4), pcto_p5 NUMERIC(10,4), amp_p5 NUMERIC(10,4),
  pctc_p6 NUMERIC(10,4), pcto_p6 NUMERIC(10,4), amp_p6 NUMERIC(10,4),
  pctc_p7 NUMERIC(10,4), pcto_p7 NUMERIC(10,4), amp_p7 NUMERIC(10,4),
  updated_at TIMESTAMP DEFAULT NOW(),
  PRIMARY KEY (sec_code, pick_date)
);

-- 用户关注指数热度表
CREATE TABLE IF NOT EXISTS stock_heat (
  sec_code  VARCHAR(6) NOT NULL,
  pick_date DATE NOT NULL,
  heat_m5 NUMERIC(10,2), heat_m4 NUMERIC(10,2), heat_m3 NUMERIC(10,2), 
  heat_m2 NUMERIC(10,2), heat_m1 NUMERIC(10,2), heat_d0 NUMERIC(10,2),
  updated_at TIMESTAMP DEFAULT NOW(),
  PRIMARY KEY (sec_code, pick_date)
);

-- 散户比例表
CREATE TABLE IF NOT EXISTS stock_retail (
  sec_code  VARCHAR(6) NOT NULL,
  pick_date DATE NOT NULL,
  retail_m5 NUMERIC(10,4), retail_m4 NUMERIC(10,4), retail_m3 NUMERIC(10,4),
  retail_m2 NUMERIC(10,4), retail_m1 NUMERIC(10,4), retail_d0 NUMERIC(10,4),
  updated_at TIMESTAMP DEFAULT NOW(),
  PRIMARY KEY (sec_code, pick_date)
);

CREATE INDEX IF NOT EXISTS idx_pre_date  ON stock_pre(pick_date);
CREATE INDEX IF NOT EXISTS idx_post_date ON stock_post(pick_date);
CREATE INDEX IF NOT EXISTS idx_heat_date ON stock_heat(pick_date);
CREATE INDEX IF NOT EXISTS idx_retail_date ON stock_retail(pick_date);
"""

def drop_all_tables():
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE;"))
        conn.execute(text("CREATE SCHEMA public;"))
        conn.execute(text("GRANT ALL ON SCHEMA public TO public;"))
        conn.execute(text(DDL))
    print("🧨 已销毁并重建所有表结构")

def truncate_all_data():
    with engine.begin() as conn:
        for t in ["stock_pre","stock_post","stock_info","ref_list","stock_heat","stock_retail"]:
            conn.execute(text(f"TRUNCATE TABLE {t} RESTART IDENTITY CASCADE;"))
    print("🧹 已清空所有表数据（保留结构）")

def ensure_tables():
    with engine.begin() as conn:
        conn.execute(text(DDL))

# ================== 读取 Excel & 分段 ==================
def read_picks(excel_path: str) -> pd.DataFrame:
    if excel_path.lower().endswith('.csv'):
        df = pd.read_csv(excel_path)
    else:
        df = pd.read_excel(excel_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if not {'pick_date','code'}.issubset(df.columns):
        raise ValueError("Excel/CSV 需包含列：pick_date, code")

    df['sec_code']  = df['code'].apply(fix_code_to_6digits)
    df['pick_date'] = df['pick_date'].apply(parse_date)

    bad = df[df['sec_code'].isna()]
    if not bad.empty:
        print("⚠️ 发现无效代码（已跳过）：", len(bad))
        print("示例：", bad.head(5).to_dict(orient='records'))

    df = df[df['sec_code'].notna()]
    df = df[['sec_code','pick_date']].drop_duplicates().sort_values(['sec_code','pick_date'])
    return df

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
    rows = []
    for sec, g in df_picks.groupby('sec_code'):
        ds = sorted(g['pick_date'].tolist())
        starts, streaks = episodes_from_dates(ds)
        rows.append({
            'sec_code': sec,
            'board': board_from_code(sec),
            'sec_name': None,
            'pick_dates': ','.join(d.strftime('%Y-%m-%d') for d in starts),
            'streaks': ','.join(str(x) for x in streaks)
        })
    return pd.DataFrame(rows)

# ================== 获取热度/机构参与度/行业 ==================
def fetch_heat_data(sec_code: str, max_retry: int = 3, sleep_sec: float = 0.6) -> pd.DataFrame:
    """获取用户关注指数热度数据"""
    last_err = None
    for _ in range(max_retry):
        try:
            df = ak.stock_comment_detail_scrd_focus_em(symbol=sec_code)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df = df.copy()
                df.columns = ['trade_date', 'heat_value']
                df['trade_date'] = pd.to_datetime(df['trade_date']).dt.date
                return df.sort_values('trade_date')
            return pd.DataFrame()
        except Exception as e:
            last_err = e
            time.sleep(sleep_sec)
    if last_err:
        print(f"[{sec_code}] 获取热度数据失败：{last_err}")
    return pd.DataFrame()

def fetch_retail_data(sec_code: str, max_retry: int = 3, sleep_sec: float = 0.6) -> pd.DataFrame:
    """获取机构参与度数据并计算散户比例"""
    last_err = None
    for _ in range(max_retry):
        try:
            df = ak.stock_comment_detail_zlkp_jgcyd_em(symbol=sec_code)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df = df.copy()
                df.columns = ['trade_date', 'institution_rate']
                df['trade_date'] = pd.to_datetime(df['trade_date']).dt.date
                # 计算散户比例 = 100 - 机构参与度
                df['retail_rate'] = 100.0 - df['institution_rate']
                return df.sort_values('trade_date')
            return pd.DataFrame()
        except Exception as e:
            last_err = e
            time.sleep(sleep_sec)
    if last_err:
        print(f"[{sec_code}] 获取机构参与度数据失败：{last_err}")
    return pd.DataFrame()

def fetch_industry_info(sec_code: str, max_retry: int = 3, sleep_sec: float = 0.6) -> str | None:
    """获取行业信息"""
    last_err = None
    for _ in range(max_retry):
        try:
            df = ak.stock_individual_info_em(sec_code)
            if isinstance(df, pd.DataFrame) and not df.empty:
                item_col, val_col = df.columns[:2]
                row = df[df[item_col].astype(str).str.contains("行业", na=False)]
                if not row.empty:
                    industry = str(row.iloc[0, 1]).strip()
                    return industry if industry and industry != "None" else None
            return None
        except Exception as e:
            last_err = e
            time.sleep(sleep_sec)
    if last_err:
        print(f"[{sec_code}] 获取行业信息失败：{last_err}")
    return None

# ================== 名称/市值（带重试） ==================
def fetch_name_mktcap(sym: str, max_retry: int = 3, sleep_sec: float = 0.6):
    last_err = None
    for _ in range(max_retry):
        try:
            info = ak.stock_individual_info_em(sym)  # sym = '603123'
            if isinstance(info, pd.DataFrame) and not info.empty and info.shape[1] >= 2:
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

def upsert_stock_info(sec_code: str, sec_name: str | None, cap100m: float | None, industry: str | None = None):
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO stock_info (sec_code, sec_name, industry, float_mktcap_100m)
            VALUES (:sec, :name, :industry, :cap)
            ON CONFLICT (sec_code) DO UPDATE
            SET sec_name = COALESCE(EXCLUDED.sec_name, stock_info.sec_name),
                industry = COALESCE(EXCLUDED.industry, stock_info.industry),
                float_mktcap_100m = COALESCE(EXCLUDED.float_mktcap_100m, stock_info.float_mktcap_100m),
                updated_at = NOW();
        """), {'sec': sec_code, 'name': (sec_name or ''), 'industry': industry, 'cap': cap100m})

def fill_names_and_mktcap(ref_df: pd.DataFrame) -> pd.DataFrame:
    names, caps, industries = [], [], []
    print("📊 正在获取股票名称、市值和行业信息...")
    for sec in tqdm(ref_df['sec_code'], desc="获取股票信息", unit="只"):
        name, cap = fetch_name_mktcap(sec)  # 传 6位数字即可
        industry = fetch_industry_info(sec)  # 获取行业信息
        names.append(name); caps.append(cap); industries.append(industry)
        upsert_stock_info(sec, name, cap, industry)
    ref_df = ref_df.copy()
    ref_df['sec_name'] = names
    return ref_df

def save_ref_list(ref_df: pd.DataFrame):
    with engine.begin() as conn:
        upsert = text("""
            INSERT INTO ref_list (sec_code, board, sec_name, pick_dates, streaks)
            VALUES (:sec, :board, :name, :dates, :streaks)
            ON CONFLICT (sec_code) DO UPDATE
            SET board = EXCLUDED.board,
                sec_name = COALESCE(EXCLUDED.sec_name, ref_list.sec_name),
                pick_dates = EXCLUDED.pick_dates,
                streaks = EXCLUDED.streaks,
                updated_at = NOW();
        """)
        for _, r in ref_df.iterrows():
            conn.execute(upsert, {
                'sec': r['sec_code'],
                'board': r['board'],
                'name': r['sec_name'] if pd.notna(r['sec_name']) else None,
                'dates': r['pick_dates'],
                'streaks': r['streaks']
            })
    print(f"✅ 已写入 A表 ref_list：{len(ref_df)} 只")

# ================== 行情：交易日索引 ==================
def fetch_hist_df(sec_code: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    使用akshare获取股票历史数据
    尝试多种方法：
      1) ak.stock_zh_a_daily(不复权版本，稳定可用)
      2) ak.stock_zh_a_hist(6位数字，备选)
    """
    s = start.strftime("%Y%m%d")
    e = end.strftime("%Y%m%d")

    # 路线1：stock_zh_a_daily（新版本推荐，最稳定）
    try:
        # 生成正确的symbol格式
        if sec_code.startswith('6'):
            symbol = f"sh{sec_code}"  # 沪市
        elif sec_code.startswith(('0', '3')):
            symbol = f"sz{sec_code}"  # 深市
        elif sec_code.startswith(('4', '8')):
            symbol = f"bj{sec_code}"  # 北交所（如果支持）
        else:
            symbol = f"sz{sec_code}"  # 默认深市
            
        df = ak.stock_zh_a_daily(symbol=symbol, start_date=s, end_date=e, adjust="")
        if isinstance(df, pd.DataFrame) and not df.empty:
            # 成功获取数据，跳转到数据处理部分
            pass
        else:
            df = None
    except Exception as e:
        # print(f"[{sec_code}] stock_zh_a_daily失败: {e}")
        df = None

    # 路线2：stock_zh_a_hist（6位数字，备选）
    if df is None:
        try:
            df = ak.stock_zh_a_hist(symbol=sec_code, period="daily", start_date=s, end_date=e, adjust="qfq")
            if isinstance(df, pd.DataFrame) and not df.empty:
                pass
            else:
                df = None
        except Exception as e:
            # print(f"[{sec_code}] stock_zh_a_hist失败: {e}")
            df = None

    # 如果所有方法都失败
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    # 统一列名处理
    mapping = {
        # stock_zh_a_daily的列名
        "date":"date","open":"open","high":"high","low":"low","close":"close",
        "volume":"volume","amount":"amount","outstanding_share":"outstanding_share","turnover":"turnover_rate",
        # stock_zh_a_hist的中文列名
        "日期":"date","开盘":"open","最高":"high","最低":"low","收盘":"close",
        "涨跌幅":"pct_chg","振幅":"amplitude","成交量":"volume","成交额":"amount",
        "换手率":"turnover_rate","量比":"volume_ratio","前收盘":"pre_close",
        # 其他可能的列名
        "change_pct":"pct_chg","preclose":"pre_close"
    }
    
    # 重命名列
    cols = {c: mapping.get(c, c) for c in df.columns}
    df = df.rename(columns=cols).copy()

    # 确保日期列格式正确
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df[pd.notna(df["date"])].sort_values("date")
    if df.empty: 
        return pd.DataFrame()

    # 计算必需的技术指标
    # 前收盘价
    if "pre_close" not in df.columns or df["pre_close"].isna().all():
        df["pre_close"] = df["close"].shift(1)
    
    # 涨跌幅（百分比）
    if "pct_chg" not in df.columns or df["pct_chg"].isna().all():
        df["pct_chg"] = (df["close"] / df["pre_close"] - 1.0) * 100
    
    # 振幅（百分比）
    if "amplitude" not in df.columns or df["amplitude"].isna().all():
        df["amplitude"] = (df["high"] - df["low"]) / df["pre_close"] * 100
    
    # 确保换手率列存在
    if "turnover_rate" not in df.columns:
        if "turnover" in df.columns:
            df["turnover_rate"] = df["turnover"]
        else:
            df["turnover_rate"] = None
    
    # 开盘涨跌幅
    df["pcto"] = (df["open"] / df["pre_close"] - 1.0) * 100
    
    return df

def pick_by_trade_offset(hist: pd.DataFrame, t: dt.date, delta: int):
    """
    在 hist 的交易日序列里：以“最近不晚于 t 的交易日”为基准，再取偏移 delta。
    这样 Excel 给到周末/节假日时不会整段失配。
    返回：pctc, pcto, amp, turn, amt
    """
    if hist is None or hist.empty:
        return (None, None, None, None, None)

    dates = hist["date"].tolist()
    # 找到 <= t 的最近一个交易日下标
    i = None
    # 快速二分也行，这里用线性回退易读
    for k in range(len(dates)-1, -1, -1):
        if dates[k] <= t:
            i = k
            break
    if i is None:
        return (None, None, None, None, None)

    j = i + delta
    if j < 0 or j >= len(dates):
        return (None, None, None, None, None)

    r = hist.iloc[j]
    return (
        float(r.get('pct_chg')) if pd.notna(r.get('pct_chg')) else None,
        float(r.get('pcto'))    if pd.notna(r.get('pcto'))    else None,
        float(r.get('amplitude')) if pd.notna(r.get('amplitude')) else None,
        float(r.get('turnover_rate')) if pd.notna(r.get('turnover_rate')) else None,
        float(r.get('amount'))  if pd.notna(r.get('amount'))  else None,
    )

def pick_by_date_offset(data_df: pd.DataFrame, target_date: dt.date, delta: int, value_col: str):
    """
    从时间序列数据中获取目标日期偏移delta天的值
    """
    if data_df is None or data_df.empty:
        return None
    
    dates = data_df["trade_date"].tolist()
    # 找到 <= target_date 的最近一个交易日下标
    i = None
    for k in range(len(dates)-1, -1, -1):
        if dates[k] <= target_date:
            i = k
            break
    if i is None:
        return None
    
    j = i + delta
    if j < 0 or j >= len(dates):
        return None
    
    r = data_df.iloc[j]
    val = r.get(value_col)
    return float(val) if pd.notna(val) else None

# ================== 写入：stock_pre / stock_post / stock_heat / stock_retail ==================
def upsert_pre_row(conn, sec: str, t: dt.date, hist: pd.DataFrame):
    payload = {'sec': sec, 't': t}
    for suf, delta in [('m3',-3),('m2',-2),('m1',-1),('d0',0)]:
        pctc, pcto, amp, turn, amt = pick_by_trade_offset(hist, t, delta)
        payload[f"pctc_{suf}"]=pctc; payload[f"pcto_{suf}"]=pcto
        payload[f"amp_{suf}"]=amp;   payload[f"turn_{suf}"]=turn; payload[f"amt_{suf}"]=amt

    conn.execute(text("""
        INSERT INTO stock_pre (
          sec_code, pick_date,
          pctc_m3,pcto_m3,amp_m3,turn_m3,amt_m3,
          pctc_m2,pcto_m2,amp_m2,turn_m2,amt_m2,
          pctc_m1,pcto_m1,amp_m1,turn_m1,amt_m1,
          pctc_d0,pcto_d0,amp_d0,turn_d0,amt_d0
        ) VALUES (
          :sec, :t,
          :pctc_m3,:pcto_m3,:amp_m3,:turn_m3,:amt_m3,
          :pctc_m2,:pcto_m2,:amp_m2,:turn_m2,:amt_m2,
          :pctc_m1,:pcto_m1,:amp_m1,:turn_m1,:amt_m1,
          :pctc_d0,:pcto_d0,:amp_d0,:turn_d0,:amt_d0
        )
        ON CONFLICT (sec_code, pick_date) DO UPDATE SET
          pctc_m3=EXCLUDED.pctc_m3, pcto_m3=EXCLUDED.pcto_m3, amp_m3=EXCLUDED.amp_m3, turn_m3=EXCLUDED.turn_m3, amt_m3=EXCLUDED.amt_m3,
          pctc_m2=EXCLUDED.pctc_m2, pcto_m2=EXCLUDED.pcto_m2, amp_m2=EXCLUDED.amp_m2, turn_m2=EXCLUDED.turn_m2, amt_m2=EXCLUDED.amt_m2,
          pctc_m1=EXCLUDED.pctc_m1, pcto_m1=EXCLUDED.pcto_m1, amp_m1=EXCLUDED.amp_m1, turn_m1=EXCLUDED.turn_m1, amt_m1=EXCLUDED.amt_m1,
          pctc_d0=EXCLUDED.pctc_d0, pcto_d0=EXCLUDED.pcto_d0, amp_d0=EXCLUDED.amp_d0, turn_d0=EXCLUDED.turn_d0, amt_d0=EXCLUDED.amt_d0,
          updated_at = NOW();
    """), payload)

def upsert_post_row(conn, sec: str, t: dt.date, hist: pd.DataFrame):
    payload = {'sec': sec, 't': t}
    for k in range(1, 8):  # p1..p7
        pctc, pcto, amp, _, _ = pick_by_trade_offset(hist, t, k)
        payload[f"pctc_p{k}"]=pctc; payload[f"pcto_p{k}"]=pcto; payload[f"amp_p{k}"]=amp

    conn.execute(text("""
        INSERT INTO stock_post (
          sec_code, pick_date,
          pctc_p1,pcto_p1,amp_p1,
          pctc_p2,pcto_p2,amp_p2,
          pctc_p3,pcto_p3,amp_p3,
          pctc_p4,pcto_p4,amp_p4,
          pctc_p5,pcto_p5,amp_p5,
          pctc_p6,pcto_p6,amp_p6,
          pctc_p7,pcto_p7,amp_p7
        ) VALUES (
          :sec, :t,
          :pctc_p1,:pcto_p1,:amp_p1,
          :pctc_p2,:pcto_p2,:amp_p2,
          :pctc_p3,:pcto_p3,:amp_p3,
          :pctc_p4,:pcto_p4,:amp_p4,
          :pctc_p5,:pcto_p5,:amp_p5,
          :pctc_p6,:pcto_p6,:amp_p6,
          :pctc_p7,:pcto_p7,:amp_p7
        )
        ON CONFLICT (sec_code, pick_date) DO UPDATE SET
          pctc_p1=EXCLUDED.pctc_p1, pcto_p1=EXCLUDED.pcto_p1, amp_p1=EXCLUDED.amp_p1,
          pctc_p2=EXCLUDED.pctc_p2, pcto_p2=EXCLUDED.pcto_p2, amp_p2=EXCLUDED.amp_p2,
          pctc_p3=EXCLUDED.pctc_p3, pcto_p3=EXCLUDED.pcto_p3, amp_p3=EXCLUDED.amp_p3,
          pctc_p4=EXCLUDED.pctc_p4, pcto_p4=EXCLUDED.pcto_p4, amp_p4=EXCLUDED.amp_p4,
          pctc_p5=EXCLUDED.pctc_p5, pcto_p5=EXCLUDED.pcto_p5, amp_p5=EXCLUDED.amp_p5,
          pctc_p6=EXCLUDED.pctc_p6, pcto_p6=EXCLUDED.pcto_p6, amp_p6=EXCLUDED.amp_p6,
          pctc_p7=EXCLUDED.pctc_p7, pcto_p7=EXCLUDED.pcto_p7, amp_p7=EXCLUDED.amp_p7,
          updated_at = NOW();
    """), payload)

def upsert_heat_row(conn, sec: str, t: dt.date, heat_df: pd.DataFrame):
    """写入热度数据：入选日当天和前5日"""
    payload = {'sec': sec, 't': t}
    for k in range(-5, 1):  # m5, m4, m3, m2, m1, d0
        suf = f"m{abs(k)}" if k < 0 else "d0"
        heat_val = pick_by_date_offset(heat_df, t, k, 'heat_value')
        payload[f"heat_{suf}"] = heat_val
    
    conn.execute(text("""
        INSERT INTO stock_heat (
          sec_code, pick_date, heat_m5, heat_m4, heat_m3, heat_m2, heat_m1, heat_d0
        ) VALUES (
          :sec, :t, :heat_m5, :heat_m4, :heat_m3, :heat_m2, :heat_m1, :heat_d0
        )
        ON CONFLICT (sec_code, pick_date) DO UPDATE SET
          heat_m5=EXCLUDED.heat_m5, heat_m4=EXCLUDED.heat_m4, heat_m3=EXCLUDED.heat_m3,
          heat_m2=EXCLUDED.heat_m2, heat_m1=EXCLUDED.heat_m1, heat_d0=EXCLUDED.heat_d0,
          updated_at = NOW();
    """), payload)

def upsert_retail_row(conn, sec: str, t: dt.date, retail_df: pd.DataFrame):
    """写入散户比例数据：入选日当天和前5日"""
    payload = {'sec': sec, 't': t}
    for k in range(-5, 1):  # m5, m4, m3, m2, m1, d0
        suf = f"m{abs(k)}" if k < 0 else "d0"
        retail_val = pick_by_date_offset(retail_df, t, k, 'retail_rate')
        payload[f"retail_{suf}"] = retail_val
    
    conn.execute(text("""
        INSERT INTO stock_retail (
          sec_code, pick_date, retail_m5, retail_m4, retail_m3, retail_m2, retail_m1, retail_d0
        ) VALUES (
          :sec, :t, :retail_m5, :retail_m4, :retail_m3, :retail_m2, :retail_m1, :retail_d0
        )
        ON CONFLICT (sec_code, pick_date) DO UPDATE SET
          retail_m5=EXCLUDED.retail_m5, retail_m4=EXCLUDED.retail_m4, retail_m3=EXCLUDED.retail_m3,
          retail_m2=EXCLUDED.retail_m2, retail_m1=EXCLUDED.retail_m1, retail_d0=EXCLUDED.retail_d0,
          updated_at = NOW();
    """), payload)

def fill_windows_for(ref_df: pd.DataFrame, mode: str):
    """
    仅对"每段首日 pick_date"写入 stock_pre/stock_post。
    在 append 模式下，只对"未存在的 (sec_code,pick_date)" 追加；其他模式全量覆盖。
    """
    print("🔄 正在准备窗口数据...")
    
    # 从 ref_df（Excel 计算）取得目标集
    targets = []
    for _, r in ref_df.iterrows():
        sec = r['sec_code']
        ep_starts = [dt.datetime.strptime(x.strip(), "%Y-%m-%d").date()
                     for x in str(r['pick_dates']).split(',') if x.strip()]
        for t in ep_starts:
            targets.append((sec, t))
    targets = sorted(set(targets))

    # append 模式：过滤掉 DB 已有的 (sec,t)
    if mode == "append":
        with engine.begin() as conn:
            exist_pre = pd.read_sql(text("SELECT sec_code, pick_date FROM stock_pre"), conn)
            exist_post= pd.read_sql(text("SELECT sec_code, pick_date FROM stock_post"), conn)
            exist_heat = pd.read_sql(text("SELECT sec_code, pick_date FROM stock_heat"), conn)
            exist_retail = pd.read_sql(text("SELECT sec_code, pick_date FROM stock_retail"), conn)
        existed = set(map(tuple, exist_pre.values.tolist())) | set(map(tuple, exist_post.values.tolist())) | \
                 set(map(tuple, exist_heat.values.tolist())) | set(map(tuple, exist_retail.values.tolist()))
        targets = [x for x in targets if x not in existed]
        print(f"➕ 追加模式：需要新增窗口 {len(targets)} 行")

    if not targets:
        print("📭 无需写入窗口数据")
        return

    # 预取每只股票历史（按最小/最大段首日扩 40 天，足够覆盖 m3~p7）
    groups = {}
    for sec, t in targets:
        info = groups.get(sec, {'min':t, 'max':t})
        info['min'] = min(info['min'], t)
        info['max'] = max(info['max'], t)
        groups[sec] = info

    print(f"📈 正在获取 {len(groups)} 只股票的历史数据并写入窗口...")
    
    with engine.begin() as conn:
        for sec, span in tqdm(groups.items(), desc="处理股票历史数据", unit="只"):
            start = span['min'] - dt.timedelta(days=40)
            end   = span['max'] + dt.timedelta(days=40)
            
            # 获取历史行情数据
            hist = fetch_hist_df(sec, start, end)
            
            # 获取热度数据
            heat_df = fetch_heat_data(sec)
            
            # 获取散户比例数据
            retail_df = fetch_retail_data(sec)
            
            if hist.empty:
                tqdm.write(f"[{sec}] 无历史数据，跳过")
                continue

            # 针对该 sec 的所有目标日写入
            target_dates = [x for x in targets if x[0]==sec]
            for (s, t) in target_dates:
                upsert_pre_row(conn, sec, t, hist)
                upsert_post_row(conn, sec, t, hist)
                upsert_heat_row(conn, sec, t, heat_df)
                upsert_retail_row(conn, sec, t, retail_df)

    print("✅ 已写入 stock_pre / stock_post / stock_heat / stock_retail")

# ================== 运行后校验 & 名称补齐 ==================
def validate_and_retry(ref_df: pd.DataFrame):
    print("🔍 正在校验数据完整性...")
    excel_unique = ref_df['sec_code'].nunique()
    with engine.begin() as conn:
        db_unique = conn.execute(text("SELECT COUNT(*) FROM ref_list")).scalar()
    print(f"🔎 校验 A表股票只数：Excel去重={excel_unique} vs A表={db_unique} -> {'OK' if excel_unique==db_unique else 'MISMATCH'}")

    with engine.begin() as conn:
        df_missing = pd.read_sql(text("SELECT sec_code FROM ref_list WHERE sec_name IS NULL OR sec_name=''"), conn)
    if df_missing.empty:
        print("🔎 A表名称列：无空数据 ✅")
        return

    print(f"🔁 A表名称列：发现 {len(df_missing)} 条为空，重试回填……")
    with engine.begin() as conn:
        update_a = text("UPDATE ref_list SET sec_name=:name, updated_at=NOW() WHERE sec_code=:sec;")
        for sec in tqdm(df_missing['sec_code'].tolist(), desc="补齐股票名称", unit="只"):
            name, cap = fetch_name_mktcap(sec, max_retry=4, sleep_sec=0.8)
            if name:
                upsert_stock_info(sec, name, cap)
                conn.execute(update_a, {'sec': sec, 'name': name})

    with engine.begin() as conn:
        left = conn.execute(text("SELECT COUNT(*) FROM ref_list WHERE sec_name IS NULL OR sec_name=''")).scalar()
    print(f"🔎 名称补齐后剩余空值：{left} 条")
    
    # 补齐行业信息
    with engine.begin() as conn:
        df_missing_industry = pd.read_sql(text("SELECT sec_code FROM stock_info WHERE industry IS NULL OR industry=''"), conn)
    if not df_missing_industry.empty:
        print(f"🔁 B表行业列：发现 {len(df_missing_industry)} 条为空，重试回填……")
        with engine.begin() as conn:
            update_industry = text("UPDATE stock_info SET industry=:industry, updated_at=NOW() WHERE sec_code=:sec;")
            for sec in tqdm(df_missing_industry['sec_code'].tolist(), desc="补齐行业信息", unit="只"):
                industry = fetch_industry_info(sec, max_retry=4, sleep_sec=0.8)
                if industry:
                    conn.execute(update_industry, {'sec': sec, 'industry': industry})
    else:
        print("🔎 B表行业列：无空数据 ✅")


# ================== 主流程 ==================
def main():
    parser = argparse.ArgumentParser(description="Load GuruList into PostgreSQL")
    parser.add_argument("excel", help="Excel/CSV path, must contain columns: code, pick_date")
    parser.add_argument("--mode", choices=["drop","truncate","append"], default="append",
                        help="drop: 销毁并重建结构后全量导入；truncate: 清空数据后全量导入；append: 只追加Excel中的新增段首日（默认）")
    args = parser.parse_args()

    print("🚀 开始执行 GuruList 数据加载...")
    print(f"📄 数据文件: {args.excel}")
    print(f"⚙️  执行模式: {args.mode}")
    print("-" * 50)

    # 步骤1: 准备数据库表结构
    print("📋 步骤 1/5: 准备数据库表结构")
    if args.mode == "drop":
        drop_all_tables()
    else:
        ensure_tables()
        if args.mode == "truncate":
            truncate_all_data()

    # 步骤2: 读取和处理Excel数据
    print("📋 步骤 2/5: 读取和处理Excel数据")
    df_picks = read_picks(args.excel)
    ref_df   = build_ref_list(df_picks)
    print(f"✅ 处理完成，共 {len(ref_df)} 只股票")

    # 步骤3: 获取股票基本信息
    print("📋 步骤 3/5: 获取股票基本信息")
    ref_df   = fill_names_and_mktcap(ref_df)
    save_ref_list(ref_df)

    # 步骤4: 获取历史行情数据
    print("📋 步骤 4/5: 获取历史行情数据")
    fill_windows_for(ref_df, mode=args.mode)

    # 步骤5: 数据校验和补齐
    print("📋 步骤 5/5: 数据校验和补齐")
    validate_and_retry(ref_df)

    # 汇总报告
    print("-" * 50)
    print("📊 数据加载完成汇总:")
    with engine.begin() as conn:
        n_a = conn.execute(text("SELECT COUNT(*) FROM ref_list")).scalar()
        n_b = conn.execute(text("SELECT COUNT(*) FROM stock_info")).scalar()
        n_pre = conn.execute(text("SELECT COUNT(*) FROM stock_pre")).scalar()
        n_post= conn.execute(text("SELECT COUNT(*) FROM stock_post")).scalar()
        n_heat = conn.execute(text("SELECT COUNT(*) FROM stock_heat")).scalar()
        n_retail = conn.execute(text("SELECT COUNT(*) FROM stock_retail")).scalar()
    print(f"📦 A表(ref_list): {n_a} 只股票")
    print(f"📦 B表(stock_info): {n_b} 条记录")
    print(f"📦 PRE表(特征数据): {n_pre} 行")
    print(f"📦 POST表(后续数据): {n_post} 行")
    print(f"📦 HEAT表(热度数据): {n_heat} 行")
    print(f"📦 RETAIL表(散户比例): {n_retail} 行")
    print("🎉 全部完成！")

if __name__ == "__main__":
    main()
