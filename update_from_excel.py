import os, sys, datetime as dt, pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import akshare as ak

load_dotenv()
DB_URL = os.getenv("DB_URL")
if not DB_URL:
    print("缺少 .env 里的 DB_URL"); sys.exit(1)
engine = create_engine(DB_URL, future=True)

def norm_code(c:str)->str:
    c=str(c).strip().upper().replace('.SZSE','').replace('.SSE','')
    if '.' in c: return c
    if c.startswith('6'): return f"{c}.SH"
    if c.startswith(('0','3')): return f"{c}.SZ"
    if c.startswith(('8','4')): return f"{c}.BJ"
    return c

def ak_symbol(sec_code:str)->str: return sec_code.split('.')[0]
def parse_date(s:str)->dt.date:
    s=str(s).strip().replace('/','-').replace('.','-')
    return dt.datetime.strptime(s, "%Y-%m-%d").date()

def ensure_tables():
    ddl = """
    CREATE TABLE IF NOT EXISTS ref_list (
      sec_code VARCHAR(12) PRIMARY KEY,
      sec_name VARCHAR(64),
      pick_dates TEXT NOT NULL,
      streaks TEXT,
      updated_at TIMESTAMP DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS stock_info (
      sec_code VARCHAR(12) PRIMARY KEY,
      sec_name VARCHAR(64) NOT NULL,
      float_mktcap_100m NUMERIC(20,2),
      updated_at TIMESTAMP DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS stock_window (
      sec_code VARCHAR(12) NOT NULL,
      pick_date DATE NOT NULL,
      o_m3 NUMERIC(18,4), c_m3 NUMERIC(18,4), pctc_m3 NUMERIC(10,4), pcto_m3 NUMERIC(10,4), amp_m3 NUMERIC(10,4), turn_m3 NUMERIC(10,4), amt_m3 NUMERIC(20,2),
      o_m2 NUMERIC(18,4), c_m2 NUMERIC(18,4), pctc_m2 NUMERIC(10,4), pcto_m2 NUMERIC(10,4), amp_m2 NUMERIC(10,4), turn_m2 NUMERIC(10,4), amt_m2 NUMERIC(20,2),
      o_m1 NUMERIC(18,4), c_m1 NUMERIC(18,4), pctc_m1 NUMERIC(10,4), pcto_m1 NUMERIC(10,4), amp_m1 NUMERIC(10,4), turn_m1 NUMERIC(10,4), amt_m1 NUMERIC(20,2),
      o_d0 NUMERIC(18,4), c_d0 NUMERIC(18,4), pctc_d0 NUMERIC(10,4), pcto_d0 NUMERIC(10,4), amp_d0 NUMERIC(10,4), turn_d0 NUMERIC(10,4), amt_d0 NUMERIC(20,2),
      o_p1 NUMERIC(18,4), c_p1 NUMERIC(18,4), pctc_p1 NUMERIC(10,4), pcto_p1 NUMERIC(10,4), amp_p1 NUMERIC(10,4), turn_p1 NUMERIC(10,4), amt_p1 NUMERIC(20,2),
      o_p2 NUMERIC(18,4), c_p2 NUMERIC(18,4), pctc_p2 NUMERIC(10,4), pcto_p2 NUMERIC(10,4), amp_p2 NUMERIC(10,4), turn_p2 NUMERIC(10,4), amt_p2 NUMERIC(20,2),
      o_p3 NUMERIC(18,4), c_p3 NUMERIC(18,4), pctc_p3 NUMERIC(10,4), pcto_p3 NUMERIC(10,4), amp_p3 NUMERIC(10,4), turn_p3 NUMERIC(10,4), amt_p3 NUMERIC(20,2),
      updated_at TIMESTAMP DEFAULT NOW(),
      PRIMARY KEY (sec_code, pick_date)
    );
    CREATE INDEX IF NOT EXISTS idx_window_date ON stock_window(pick_date);
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))

def read_picks(path:str)->pd.DataFrame:
    df = pd.read_excel(path) if path.lower().endswith('.xlsx') else pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if not {'pick_date','code'}.issubset(df.columns):
        raise ValueError("Excel/CSV 需包含列：pick_date, code")
    df['sec_code'] = df['code'].apply(norm_code)
    df['pick_date'] = df['pick_date'].apply(parse_date)
    return df[['sec_code','pick_date']].drop_duplicates()

def union_dates_and_streaks(existing_dates:str|None, new_dates:list[dt.date])->tuple[str,str]:
    dates = set()
    if existing_dates:
        for s in existing_dates.split(','):
            s=s.strip()
            if not s: continue
            dates.add(parse_date(s))
    dates.update(new_dates)
    dates = sorted(dates)
    # 7 天内算一段
    streaks=[]; cur=[dates[0]] if dates else []
    for d in dates[1:]:
        if (d-cur[-1]).days<=7: cur.append(d)
        else: streaks.append(len(cur)); cur=[d]
    if cur: streaks.append(len(cur))
    return (','.join(d.strftime('%Y-%m-%d') for d in dates),
            ','.join(str(x) for x in streaks))

def upsert_ref_and_info(df:pd.DataFrame):
    with engine.begin() as conn:
        # 取已有 ref_list
        exist = pd.read_sql(text("SELECT sec_code, pick_dates FROM ref_list"), conn)
        pick_map = dict(zip(exist['sec_code'], exist['pick_dates'])) if not exist.empty else {}

        # 逐股票合并 pick_dates
        agg = df.groupby('sec_code')['pick_date'].apply(list).reset_index()
        for _, row in agg.iterrows():
            sec = row['sec_code']; new_list = row['pick_date']
            merged_dates, streaks = union_dates_and_streaks(pick_map.get(sec), new_list)

            # 名称/市值
            sec_name=None; float_cap=None
            try:
                info = ak.stock_individual_info_em(ak_symbol(sec))
                if info is not None and not info.empty:
                    info.columns=['item','value']
                    kv=dict(zip(info['item'], info['value']))
                    sec_name = kv.get('证券简称') or kv.get('股票简称') or kv.get('简称')
                    float_cap = kv.get('流通市值(元)') or kv.get('流通市值')
                    if isinstance(float_cap, str):
                        float_cap=float(float_cap.replace(',','').replace('元','').strip() or 0)
                    float_cap = (float_cap/1e8) if float_cap else None
            except Exception:
                pass

            # A 表 upsert
            conn.execute(text("""
                INSERT INTO ref_list (sec_code, sec_name, pick_dates, streaks)
                VALUES (:sec, :name, :dates, :streaks)
                ON CONFLICT (sec_code) DO UPDATE
                SET sec_name=COALESCE(EXCLUDED.sec_name, ref_list.sec_name),
                    pick_dates=EXCLUDED.pick_dates,
                    streaks=EXCLUDED.streaks,
                    updated_at=NOW();
            """), {'sec':sec,'name':sec_name,'dates':merged_dates,'streaks':streaks})

            # B 表 upsert
            if sec_name or float_cap is not None:
                conn.execute(text("""
                    INSERT INTO stock_info (sec_code, sec_name, float_mktcap_100m)
                    VALUES (:sec,:name,:cap)
                    ON CONFLICT (sec_code) DO UPDATE
                    SET sec_name=COALESCE(EXCLUDED.sec_name, stock_info.sec_name),
                        float_mktcap_100m=COALESCE(EXCLUDED.float_mktcap_100m, stock_info.float_mktcap_100m),
                        updated_at=NOW();
                """), {'sec':sec,'name':sec_name or '', 'cap':float_cap})

def fetch_hist_df(sec_code:str, start:dt.date, end:dt.date)->pd.DataFrame:
    df = ak.stock_zh_a_hist(symbol=ak_symbol(sec_code), period="daily",
                            start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"), adjust="qfq")
    if df is None or df.empty: return pd.DataFrame()
    mp={"日期":"date","开盘":"open","最高":"high","最低":"low","收盘":"close",
        "涨跌幅":"pct_chg","振幅":"amplitude","成交量":"volume","成交额":"amount",
        "换手率":"turnover_rate","量比":"volume_ratio","前收盘":"pre_close"}
    for cn,en in mp.items():
        if cn in df.columns: df.rename(columns={cn:en}, inplace=True)
    if "date" not in df.columns and "日期" in df.columns: df.rename(columns={"日期":"date"}, inplace=True)
    df["date"]=pd.to_datetime(df["date"]).dt.date
    df.sort_values("date", inplace=True)
    if "pre_close" not in df.columns or df["pre_close"].isna().all(): df["pre_close"]=df["close"].shift(1)
    if "pct_chg" not in df.columns or df["pct_chg"].isna().all(): df["pct_chg"]=(df["close"]/df["pre_close"]-1)*100
    if "amplitude" not in df.columns or df["amplitude"].isna().all(): df["amplitude"]=(df["high"]-df["low"])/df["pre_close"]*100
    df["pcto"]=(df["open"]/df["pre_close"]-1)*100
    return df

def upsert_windows(df:pd.DataFrame):
    with engine.begin() as conn:
        # 找出 stock_window 里已有的键，避免重复
        exists = pd.read_sql(text("SELECT sec_code, pick_date FROM stock_window"), conn)
        exist_set = set((r['sec_code'], r['pick_date']) for _, r in exists.iterrows()) if not exists.empty else set()

        upsert = text("""
        INSERT INTO stock_window (
          sec_code, pick_date,
          o_m3,c_m3,pctc_m3,pcto_m3,amp_m3,turn_m3,amt_m3,
          o_m2,c_m2,pctc_m2,pcto_m2,amp_m2,turn_m2,amt_m2,
          o_m1,c_m1,pctc_m1,pcto_m1,amp_m1,turn_m1,amt_m1,
          o_d0,c_d0,pctc_d0,pcto_d0,amp_d0,turn_d0,amt_d0,
          o_p1,c_p1,pctc_p1,pcto_p1,amp_p1,turn_p1,amt_p1,
          o_p2,c_p2,pctc_p2,pcto_p2,amp_p2,turn_p2,amt_p2,
          o_p3,c_p3,pctc_p3,pcto_p3,amp_p3,turn_p3,amt_p3
        ) VALUES (
          :sec, :t,
          :o_m3,:c_m3,:pctc_m3,:pcto_m3,:amp_m3,:turn_m3,:amt_m3,
          :o_m2,:c_m2,:pctc_m2,:pcto_m2,:amp_m2,:turn_m2,:amt_m2,
          :o_m1,:c_m1,:pctc_m1,:pcto_m1,:amp_m1,:turn_m1,:amt_m1,
          :o_d0,:c_d0,:pctc_d0,:pcto_d0,:amp_d0,:turn_d0,:amt_d0,
          :o_p1,:c_p1,:pctc_p1,:pcto_p1,:amp_p1,:turn_p1,:amt_p1,
          :o_p2,:c_p2,:pctc_p2,:pcto_p2,:amp_p2,:turn_p2,:amt_p2,
          :o_p3,:c_p3,:pctc_p3,:pcto_p3,:amp_p3,:turn_p3,:amt_p3
        )
        ON CONFLICT (sec_code, pick_date) DO NOTHING;
        """)

        for sec, g in df.groupby('sec_code'):
            dates = sorted(set(g['pick_date'].tolist()))
            # 只抓没入库过的 pick_date
            new_dates = [d for d in dates if (sec, d) not in exist_set]
            if not new_dates: continue

            start = new_dates[0] - dt.timedelta(days=10)
            end   = new_dates[-1] + dt.timedelta(days=10)
            hist = fetch_hist_df(sec, start, end)
            if hist.empty: continue
            by_date = {d: row for d, row in hist.set_index('date').iterrows()}

            def pick(d):
                r = by_date.get(d)
                if r is None: return (None,)*7
                return (float(r['open']) if pd.notna(r['open']) else None,
                        float(r['close']) if pd.notna(r['close']) else None,
                        float(r['pct_chg']) if pd.notna(r['pct_chg']) else None,
                        float(r['pcto']) if pd.notna(r['pcto']) else None,
                        float(r['amplitude']) if pd.notna(r['amplitude']) else None,
                        float(r['turnover_rate']) if pd.notna(r['turnover_rate']) else None,
                        float(r['amount']) if pd.notna(r['amount']) else None)

            for t in new_dates:
                payload={'sec':sec,'t':t}
                for suf, delta in [('m3',-3),('m2',-2),('m1',-1),('d0',0),('p1',1),('p2',2),('p3',3)]:
                    o,c,pctc,pcto,amp,turn,amt = pick(t + dt.timedelta(days=delta))
                    payload[f"o_{suf}"]=o; payload[f"c_{suf}"]=c
                    payload[f"pctc_{suf}"]=pctc; payload[f"pcto_{suf}"]=pcto
                    payload[f"amp_{suf}"]=amp; payload[f"turn_{suf}"]=turn; payload[f"amt_{suf}"]=amt
                conn.execute(upsert, payload)

def main():
    if len(sys.argv)<2:
        print("用法: python update_from_excel.py <guruList.xlsx|csv>"); sys.exit(1)
    path=sys.argv[1]
    ensure_tables()
    df = read_picks(path)
    upsert_ref_and_info(df)
    upsert_windows(df)
    print("✅ 增量更新完成")

if __name__ == "__main__":
    main()
