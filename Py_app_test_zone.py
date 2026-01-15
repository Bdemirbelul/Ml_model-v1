import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# Load final dataset (365d + club perf)
# -----------------------------
df = pd.read_parquet("outputs/training_final_365d_with_club_perf.parquet").copy()
df["date"] = pd.to_datetime(df["date"])

# -----------------------------
# Load transfers
# -----------------------------
tr = pd.read_csv("data/transfers.csv", parse_dates=["transfer_date"])

# normalize
tr["player_id"] = pd.to_numeric(tr["player_id"], errors="coerce")
tr = tr.dropna(subset=["player_id", "transfer_date"]).copy()
tr["player_id"] = tr["player_id"].astype("int64")
tr["transfer_fee"] = pd.to_numeric(tr["transfer_fee"], errors="coerce")

# Keep only needed
tr = tr[["player_id", "transfer_date", "transfer_fee", "from_club_id", "to_club_id"]].copy()

# -----------------------------
# Build 6-month transfer features per snapshot
# -----------------------------
# We'll compute per player with searchsorted (no merge_asof)
df = df.sort_values(["player_id", "date"]).reset_index(drop=True)
tr = tr.sort_values(["player_id", "transfer_date"]).reset_index(drop=True)

tr_groups = dict(tuple(tr.groupby("player_id", sort=False)))

will_transfer = np.zeros(len(df), dtype="float64")
transfer_count = np.zeros(len(df), dtype="float64")
fee_sum = np.zeros(len(df), dtype="float64")
club_change = np.zeros(len(df), dtype="float64")

for pid, idx in df.groupby("player_id", sort=False).groups.items():
    rows = np.array(list(idx), dtype=int)
    snap_dates = df.loc[rows, "date"].to_numpy(dtype="datetime64[ns]")
    end_dates = snap_dates + np.timedelta64(180, "D")  # 6 months ~ 180 days

    gtr = tr_groups.get(pid)
    if gtr is None or len(gtr) == 0:
        continue

    tdates = gtr["transfer_date"].to_numpy(dtype="datetime64[ns]")
    tfee = gtr["transfer_fee"].to_numpy(dtype="float64")
    tfee = np.nan_to_num(tfee, nan=0.0)

    fclub = pd.to_numeric(gtr["from_club_id"], errors="coerce").to_numpy(dtype="float64")
    tclub = pd.to_numeric(gtr["to_club_id"], errors="coerce").to_numpy(dtype="float64")

    # window: [snapshot_date, snapshot_date+180d)
    left = np.searchsorted(tdates, snap_dates, side="left")
    right = np.searchsorted(tdates, end_dates, side="left")  # exclusive

    # counts and fee sum via prefix
    ccount = np.cumsum(np.ones(len(tdates), dtype="float64"))
    cfee = np.cumsum(tfee)

    def pref(arr, i):
        out = np.zeros_like(i, dtype="float64")
        m = i > 0
        out[m] = arr[i[m]-1]
        return out

    cnt = pref(ccount, right) - pref(ccount, left)
    fsum = pref(cfee, right) - pref(cfee, left)

    # club_change: any transfer where from != to inside window
    # We do a simple scan only for snapshots that have transfers (usually sparse)
    ch = np.zeros(len(rows), dtype="float64")
    for k in range(len(rows)):
        if cnt[k] <= 0:
            continue
        a, b = left[k], right[k]
        # any non-null change?
        sub_from = fclub[a:b]
        sub_to = tclub[a:b]
        mask = np.isfinite(sub_from) & np.isfinite(sub_to)
        if mask.any():
            ch[k] = float(np.any(sub_from[mask] != sub_to[mask]))

    will_transfer[rows] = (cnt > 0).astype(float)
    transfer_count[rows] = cnt
    fee_sum[rows] = fsum
    club_change[rows] = ch

df["will_transfer_6m"] = will_transfer
df["transfer_count_6m"] = transfer_count
df["transfer_fee_sum_6m"] = fee_sum
df["transfer_fee_log_6m"] = np.log1p(df["transfer_fee_sum_6m"])
df["club_change_6m"] = club_change

# -----------------------------
# Train (same time split + log target)
# -----------------------------
TRAIN_END_YEAR = 2019
VALID_START_YEAR = 2020
VALID_END_YEAR = 2021
TEST_START_YEAR = 2022

train = df[df["year"] <= TRAIN_END_YEAR].copy()
valid = df[(df["year"] >= VALID_START_YEAR) & (df["year"] <= VALID_END_YEAR)].copy()
test  = df[df["year"] >= TEST_START_YEAR].copy()

y_train = train["y_log_change"].astype(float)
y_valid_raw = valid["y_value_change_6m"].astype(float)
y_test_raw  = test["y_value_change_6m"].astype(float)

drop_cols = ["y_value_change_6m","y_log_change","year","date"]
X_train = train.drop(columns=drop_cols, errors="ignore")
X_valid = valid.drop(columns=drop_cols, errors="ignore")
X_test  = test.drop(columns=drop_cols, errors="ignore")

cat_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]
num_cols = [c for c in X_train.columns if c not in cat_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline(steps=[
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols),
    ],
    remainder="drop"
)

model = RandomForestRegressor(
    n_estimators=160,
    max_depth=20,
    min_samples_leaf=3,
    n_jobs=-1,
    random_state=42
)

pipe = Pipeline(steps=[("prep", preprocess), ("model", model)])
pipe.fit(X_train, y_train)

def evaluate(name, X, y_raw):
    pred_log = pipe.predict(X)
    pred_raw = np.expm1(pred_log)
    mae = mean_absolute_error(y_raw, pred_raw)
    rmse = np.sqrt(mean_squared_error(y_raw, pred_raw))
    dir_acc = (np.sign(pred_raw) == np.sign(y_raw)).mean()
    print(f"{name}: MAE={mae:.4f} RMSE={rmse:.4f} DirectionAcc={dir_acc:.3f}")

print("\n--- RESULTS (365d + club perf + TRANSFERS) ---")
evaluate("VALID", X_valid, y_valid_raw)
evaluate("TEST",  X_test,  y_test_raw)

print("\nTransfer feature sanity:")
print("will_transfer_6m rate:", df["will_transfer_6m"].mean())
print("avg transfer_count_6m:", df["transfer_count_6m"].mean())
print("nonzero fee rate:", (df["transfer_fee_sum_6m"] > 0).mean())
