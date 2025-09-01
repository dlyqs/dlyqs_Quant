# dlyqs_Quant
dlyqs 的量化项目，跟踪大佬的列表起步

## 1. 项目背景
- 我关注的投资大佬每天都会提供一个 **股票候选名单**（Excel 形式，含股票代码和入选日期）。  
- 观察发现，这些股票在入选后的 **1～3 个交易日内经常涨停/大涨**。  
- **目标：**
  1. 把这些名单导入数据库，形成可追溯的数据仓库；  
  2. 补充股票的基本面信息（名称、市值等）和入选前后若干交易日的行情数据；  
  3. 为后续量化分析（如板块效应、连续入选效应、筛选最可能大涨的票）提供干净的基础数据。  

---

## 2. 环境依赖
项目需要安装以下Python包：
```bash
python -m pip install -r requirements.txt
```

同时需要在项目根目录创建 `.env` 文件，配置数据库连接：
```
DB_URL=postgresql://username:password@host:port/database
```

---

## 3. 数据库表结构
目前设计 **四张表**：`ref_list`、`stock_info`、`stock_pre`、`stock_post`。

### A. `ref_list`（候选名单索引表）
- 每只股票在候选名单中的历史记录，按"**7 天为一个分段**"聚合。
- **字段：**
  - `sec_code`：股票代码（6位数字，如 `000001`）  
  - `board`：交易板块（主板/创业板/科创板/北交所）
  - `sec_name`：股票名称  
  - `pick_dates`：逗号分隔的每段**首个入选日**（例 `2025-08-18,2025-08-27`）  
  - `streaks`：与 `pick_dates` 一一对应，记录段内连续天数（例 `3,2`）  
  - `updated_at`：更新时间  

**示例：**
```
000001,主板,平安银行,2025-08-18,2
```

---

### B. `stock_info`（公司基本面表）
- 每只股票的基本信息，目前只存储市值，后续可扩展。
- **字段：**
  - `sec_code`：股票代码（6位数字）  
  - `sec_name`：股票名称  
  - `float_mktcap_100m`：流通市值（单位：亿元）  
  - `updated_at`：更新时间  

**示例：**
```
000001,平安银行,1234.56
```

---

### C. `stock_pre`（入选前特征表）
- 记录每只股票在 **入选日前 T-3 ~ T0（入选当日）** 的关键交易数据。
- **字段：**
  - `sec_code`：股票代码（6位数字）
  - `pick_date`：入选日期
  - **T-3日指标：** `pctc_m3`, `pcto_m3`, `amp_m3`, `turn_m3`, `amt_m3`
  - **T-2日指标：** `pctc_m2`, `pcto_m2`, `amp_m2`, `turn_m2`, `amt_m2`
  - **T-1日指标：** `pctc_m1`, `pcto_m1`, `amp_m1`, `turn_m1`, `amt_m1`
  - **T0日指标：** `pctc_d0`, `pcto_d0`, `amp_d0`, `turn_d0`, `amt_d0`
  - `updated_at`：更新时间

**指标说明：**
- `pctc_*`：收盘涨跌幅（%）
- `pcto_*`：开盘涨跌幅（%）
- `amp_*`：振幅（%）
- `turn_*`：换手率（%）
- `amt_*`：成交额（元）

---

### D. `stock_post`（入选后表现表）
- 记录每只股票在 **入选日后 T+1 ~ T+7** 的关键表现数据。
- **字段：**
  - `sec_code`：股票代码（6位数字）
  - `pick_date`：入选日期
  - **T+1~T+7日指标：** `pctc_p1~p7`, `pcto_p1~p7`, `amp_p1~p7`
  - `updated_at`：更新时间

**指标说明：**
- `pctc_p*`：收盘涨跌幅（%）
- `pcto_p*`：开盘涨跌幅（%）
- `amp_p*`：振幅（%）

---

## 4. 使用方式

### 数据文件格式要求
Excel/CSV 文件必须包含以下列：
- `code`：股票代码（支持多种格式，如 `000001`, `000001.SZ`, `600000.SH` 等）
- `pick_date`：入选日期（支持 `YYYY-MM-DD`, `YYYY/MM/DD`, `YYYY.MM.DD` 格式）

### 运行模式

#### 1) 完全重新开始（销毁表和数据）
```bash
python reset_and_load.py guruList.xlsx --mode drop
```

#### 2) 不销毁表，只清空数据重新获取
```bash
python reset_and_load.py guruList.xlsx --mode truncate
```

#### 3) 仅追加新数据（默认）
```bash
python reset_and_load.py guruList.xlsx
# 或明确指定
python reset_and_load.py guruList.xlsx --mode append
```

---

## 5. AkShare 接口使用说明

项目使用以下 AkShare 接口获取股票数据：

### 5.1 股票基本信息
```python
# 获取股票名称和流通市值
info = ak.stock_individual_info_em(symbol='000001')
```
- **参数：** `symbol` - 6位股票代码
- **返回：** DataFrame，包含股票名称、流通市值等基本信息
- **字段映射：**
  - `证券简称/股票简称/简称` → 股票名称
  - `流通市值(元)/流通市值` → 流通市值（元）

### 5.2 股票历史行情数据

#### 方法一：stock_zh_a_daily（推荐，最稳定）
```python
# 获取股票日线数据
df = ak.stock_zh_a_daily(symbol='sh000001', start_date='20240101', end_date='20241231', adjust="")
```
- **参数：**
  - `symbol` - 股票代码（带市场前缀：`sh`沪市, `sz`深市, `bj`北交所）
  - `start_date` - 开始日期（YYYYMMDD格式）
  - `end_date` - 结束日期（YYYYMMDD格式）
  - `adjust` - 复权类型（""不复权）

#### 方法二：stock_zh_a_hist（备选）
```python
# 备选方案
df = ak.stock_zh_a_hist(symbol='000001', period='daily', start_date='20240101', end_date='20241231', adjust='qfq')
```
- **参数：**
  - `symbol` - 6位股票代码
  - `period` - 周期（'daily'日线）
  - `adjust` - 复权类型（'qfq'前复权）

### 5.3 数据字段映射
程序会自动处理不同接口返回的列名差异：
```python
# 列名映射表
mapping = {
    # stock_zh_a_daily的列名
    "date":"date", "open":"open", "high":"high", "low":"low", "close":"close",
    "volume":"volume", "amount":"amount", "turnover":"turnover_rate",
    
    # stock_zh_a_hist的中文列名
    "日期":"date", "开盘":"open", "最高":"high", "最低":"low", "收盘":"close",
    "涨跌幅":"pct_chg", "振幅":"amplitude", "成交量":"volume", "成交额":"amount",
    "换手率":"turnover_rate", "前收盘":"pre_close"
}
```

### 5.4 股票代码规范化
程序会自动将各种格式的股票代码统一为6位数字：
- 输入：`000001.SZ`, `600000.SH`, `300001`, `688001.SZSE`
- 输出：`000001`, `600000`, `300001`, `688001`

### 5.5 交易板块分类
根据股票代码自动分类：
- **主板：** `600/601/603/605`（沪市）+ `000/001/002`（深市，含原中小板）
- **创业板：** `300` 开头
- **科创板：** `688` 开头  
- **北交所：** `4xxxxx/8xxxxx` 开头

### 5.6 容错机制
- **重试机制：** 每个接口调用失败时会自动重试（默认3次）
- **接口降级：** 优先使用 `stock_zh_a_daily`，失败时自动切换到 `stock_zh_a_hist`
- **数据校验：** 自动计算缺失的技术指标（涨跌幅、振幅、开盘涨跌幅等）
- **容错处理：** 遇到无效股票代码或接口异常时会跳过并记录日志

---

## 6. 执行流程

程序按以下步骤执行：

1. **准备数据库表结构** - 根据模式创建或清理表
2. **读取和处理Excel数据** - 解析股票代码和日期，按7天分段
3. **获取股票基本信息** - 调用AkShare获取股票名称和市值
4. **获取历史行情数据** - 获取入选前后的交易数据
5. **数据校验和补齐** - 验证数据完整性，补齐缺失信息

每个步骤都有详细的进度提示和错误处理。