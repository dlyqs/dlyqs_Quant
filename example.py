import datetime as dt
from typing import Tuple, Dict, Any
import pandas as pd
import akshare as ak

# ========== 工具函数 ==========
def parse_date(s: str) -> dt.date:
    return dt.datetime.strptime(s, "%Y-%m-%d").date()

def date_str(d: dt.date) -> str:
    return d.strftime("%Y%m%d")

def normalize_sec_code(code: str) -> str:
    code = code.strip().upper()
    if '.' in code:
        return code
    if code.startswith('6'):
        return f"{code}.SH"
    if code.startswith('0') or code.startswith('3'):
        return f"{code}.SZ"
    if code.startswith('8') or code.startswith('4'):
        return f"{code}.BJ"
    return code

def ak_symbol(sec_code: str) -> str:
    return sec_code.split('.')[0]

# ========== 1) 拉 T-3 ~ T+3 的日线，并计算“近似量比、均线关系、一周后涨跌” ==========
def fetch_window_kline(sec_code: str, pick_date: dt.date, pad_before: int = 120) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    返回：
      df_win: 覆盖 [T-3, T+3] 的窗口数据（含 近似量比LR5/LR20、是否>MA5/10/20）
      stats:  一些衍生统计（入选一周后相对入选日涨跌）
    """
    sym = ak_symbol(sec_code)
    start = pick_date - dt.timedelta(days=pad_before)
    end = pick_date + dt.timedelta(days=7)

    df = ak.stock_zh_a_hist(symbol=sym, period="daily", start_date=date_str(start), end_date=date_str(end), adjust="qfq")
    if df is None or df.empty:
        raise RuntimeError(f"无法获取 {sec_code} 的日线数据")

    # 兼容列名
    mapping = {
        "日期":"date","开盘":"open","最高":"high","最低":"low","收盘":"close",
        "涨跌幅":"pct_chg","振幅":"amplitude","成交量":"volume","成交额":"amount",
        "换手率":"turnover_rate","量比":"volume_ratio","前收盘":"pre_close"
    }
    for cn,en in mapping.items():
        if cn in df.columns: df.rename(columns={cn:en}, inplace=True)
    if "date" not in df.columns:
        # 有些版本日期列就是“日期”
        df.rename(columns={"日期":"date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df.sort_values("date", inplace=True)

    # 近似量比（历史回测口径）
    df["vol_ma5"] = df["volume"].rolling(5).mean()
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["lr5"] = df["volume"] / df["vol_ma5"]   # 量比近似-5日
    df["lr20"] = df["volume"] / df["vol_ma20"] # 量比近似-20日

    # 均线关系
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["above_ma5"]  = (df["close"] > df["ma5"]).astype(int)
    df["above_ma10"] = (df["close"] > df["ma10"]).astype(int)
    df["above_ma20"] = (df["close"] > df["ma20"]).astype(int)
    df["ma_stack"] = df["above_ma5"] + df["above_ma10"] + df["above_ma20"]

    # 提取窗口 [T-3, T+3]
    t_minus3 = pick_date - dt.timedelta(days=3)
    t_plus3  = pick_date + dt.timedelta(days=3)
    win = df[(df["date"] >= t_minus3) & (df["date"] <= t_plus3)].copy()

    # 入选一周后相对入选日涨跌
    # 以 T 日收盘为基准，取 T+5（自然日向后取到下一个有效交易日）
    close_T = df.loc[df["date"] == pick_date, "close"]
    ret_1w = None
    if not close_T.empty:
        base = float(close_T.iloc[0])
        # 找到 T 后最近的第 5 个交易日
        after = df[df["date"] > pick_date].head(5)
        if not after.empty and len(after) >= 5:
            close_T5 = float(after.iloc[-1]["close"])
            ret_1w = close_T5 / base - 1.0

    stats = {
        "one_week_return": ret_1w,
    }
    return win, stats

# ========== 2) 拿 财务与估值指标（ROE、EPS、PE、PB、营收/利润增速等） ==========
def fetch_fundamental_and_valuation(sec_code: str) -> Dict[str, Any]:
    sym = ak_symbol(sec_code)
    out = {}

    # 主要指标（按报告期）—— ROE、EPS、营收/净利同比、PE、PB 等
    try:
        ind = ak.stock_financial_analysis_indicator(symbol=sym)
        # 兼容列名：不同版本表头可能是中文，如 “报告期”、“净资产收益率ROE(%)”、“每股收益EPS(元)” 等
        # 我们取最近一期（第一行或最后一行，视接口排序而定）
        if not ind.empty:
            # 尝试让最近期在第一行
            if "报告期" in ind.columns:
                ind["报告期"] = pd.to_datetime(ind["报告期"], errors="coerce")
                ind.sort_values("报告期", ascending=False, inplace=True)
            row = ind.iloc[0].to_dict()
            # 兜底提取
            def pick(*names):
                for n in names:
                    if n in row and pd.notna(row[n]): return row[n]
                return None

            out.update({
                "report_date": pick("报告期","date"),
                "roe": pick("净资产收益率ROE(%)","ROE","净资产收益率(%)"),
                "eps": pick("每股收益EPS(元)","EPS","每股收益(元)"),
                "pe":  pick("市盈率PE","PE(TTM)","市盈率(倍)"),
                "pb":  pick("市净率PB","PB","市净率(倍)"),
                "revenue_yoy": pick("营业总收入同比增长率(%)","营收同比(%)"),
                "profit_yoy":  pick("净利润同比增长率(%)","净利同比(%)"),
            })
    except Exception as e:
        print(f"[{sec_code}] 财务主要指标获取失败：{e}")

    # 个股档案（当前口径的总/流通市值等）
    try:
        info = ak.stock_individual_info_em(sym)
        # 返回通常是两列：item/value
        if info is not None and not info.empty:
            info.columns = ["item","value"]
            info = info.set_index("item")["value"].to_dict()
            # 兼容键名
            def gv(*keys):
                for k in keys:
                    if k in info: return info[k]
                return None
            out.update({
                "total_mktcap_now": gv("总市值","总市值(元)"),
                "float_mktcap_now": gv("流通市值","流通市值(元)"),
            })
    except Exception as e:
        print(f"[{sec_code}] 个股档案获取失败：{e}")

    return out

# ========== 3) 拿利空事件：限售解禁 & 减持 ==========
def fetch_negative_events(sec_code: str, around: Tuple[dt.date, dt.date]) -> Dict[str, Any]:
    sym = ak_symbol(sec_code)
    d1, d2 = around
    neg = {
        "has_unlock_in_window": False,
        "unlock_rows": 0,
        "has_reduction_in_window": False,
        "reduction_rows": 0,
    }

    # 限售解禁
    try:
        unlock = ak.stock_restricted_release_em()
        # 常见列：'代码','名称','解禁日期','解禁数量(亿股)','解禁市值(亿元)'...
        if unlock is not None and not unlock.empty:
            # 统一代码为 6位（东财常用 600519）
            unlock = unlock.copy()
            unlock["代码"] = unlock["代码"].astype(str).str.zfill(6)
            mask = (unlock["代码"] == sym) & (pd.to_datetime(unlock["解禁日期"]).dt.date.between(d1, d2))
            sel = unlock[mask]
            neg["has_unlock_in_window"] = (len(sel) > 0)
            neg["unlock_rows"] = int(len(sel))
    except Exception as e:
        print(f"[{sec_code}] 解禁数据获取失败：{e}")

    # 减持公告（不同 AK 版本有不同接口；此处尝试 detail 接口，失败则跳过）
    try:
        # 如果你的 AK 版本没有该接口，会抛异常，被 except 捕获
        reduce_df = ak.stock_reduce_hold_detail_em(symbol=sym)
        # 常见列：'公告日期','股东名称','减持数量','减持比例(%)'...
        if reduce_df is not None and not reduce_df.empty:
            reduce_df["公告日期"] = pd.to_datetime(reduce_df["公告日期"], errors="coerce")
            sel = reduce_df[reduce_df["公告日期"].dt.date.between(d1, d2)]
            neg["has_reduction_in_window"] = (len(sel) > 0)
            neg["reduction_rows"] = int(len(sel))
    except Exception as e:
        print(f"[{sec_code}] 减持数据获取失败（接口可能缺失或变更）：{e}")

    return neg

# ========== 4) 将以上整合成一个 demo 接口 ==========
def demo_one_stock(sec_code: str, pick_date_str: str = "2025-08-21"):
    sec_code = normalize_sec_code(sec_code)
    pick_date = parse_date(pick_date_str)

    print(f"=== 标的: {sec_code} | 入选日: {pick_date} ===")
    # K线与近似量比、均线关系
    win, stats = fetch_window_kline(sec_code, pick_date)

    print("\n--- T-3 ~ T+3 交易数据（关键列） ---")
    cols = ["date","open","high","low","close","pct_chg","turnover_rate","amplitude","lr5","lr20","ma_stack"]
    exist_cols = [c for c in cols if c in win.columns]
    print(win[exist_cols].to_string(index=False))

    # 一周后相对涨跌
    print("\n--- 入选后一周相对涨跌（以入选日收盘为基准）---")
    print(f"{stats['one_week_return']*100:.2f}% (若为 None 表示无法计算)")

    # 财务与估值
    fund = fetch_fundamental_and_valuation(sec_code)
    print("\n--- 财务与估值（最近报告期/当前口径） ---")
    for k,v in fund.items():
        print(f"{k}: {v}")

    # 入选日前后 30 天的利空窗口
    neg = fetch_negative_events(sec_code, around=(pick_date-dt.timedelta(days=30), pick_date+dt.timedelta(days=30)))
    print("\n--- 负面事件（±30日） ---")
    print(neg)


if __name__ == "__main__":
    # 示例：贵州茅台
    demo_one_stock("60312.SH", "2025-08-18")
    # 你也可以换成你名单里的任意一只：
    # demo_one_stock("000001.SZ", "2025-08-20")
