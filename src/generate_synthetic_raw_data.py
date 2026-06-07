
"""
Synthetic Citizens Bank raw dataset generator.

Purpose
-------
Create a privacy-safe fake raw dataset with the same schema, dtypes, row count,
missingness pattern, and approximate distributional trends as the project raw data.

This is NOT a row-level perturbation of the original data. It is fully synthetic.

Output schema:
- 17,765 rows by default
- 49 columns
- dtypes: 30 float64, 11 int64, 8 object
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


RAW_COLUMN_ORDER = [
    "masked_dep_acct_num",
    "masked_bank_num",
    "masked_account_type",
    "masked_id",
    "masked_product_code",
    "bucket_days_since_open",
    "number_of_owners",
    "total_deposit_amount",
    "item_amt",
    "deposit_dt",
    "channel",
    "relationship_balance",
    "oao_flg",
    "onus_ind",
    "treasury_check_ind",
    "heloc_ind",
    "rdis",
    "max_deposit_amount30d",
    "total_deposit_item_count",
    "prevtran1",
    "prevtran2",
    "prevtran3",
    "prevtran4",
    "prevtran5",
    "prevtran6",
    "prevtran7",
    "prevtran8",
    "prevtran9",
    "prevtran10",
    "prevtrandate1",
    "prevtrandate2",
    "prevtrandate3",
    "prevtrandate4",
    "prevtrandate5",
    "prevtrandate6",
    "prevtrandate7",
    "prevtrandate8",
    "prevtrandate9",
    "prevtrandate10",
    "drawee_sum",
    "drawee_cnt",
    "drawee_avg",
    "drawee_max",
    "drawee_min",
    "RDI_DT",
    "RETURN_REASON",
    "return_target",
    "over_draft_amount",
    "month_num",
]


def _shuffle_values(values, rng: np.random.Generator) -> np.ndarray:
    """Return a shuffled numpy array."""
    values = np.asarray(values, dtype=object)
    return values[rng.permutation(len(values))]


def _values_from_counts(counts: dict, rng: np.random.Generator) -> np.ndarray:
    """
    Build an array from exact value counts and shuffle it.

    Example
    -------
    {0: 3, 1: 2} -> shuffled array of [0, 0, 0, 1, 1]
    """
    values = []
    for value, count in counts.items():
        values.extend([value] * int(count))
    return _shuffle_values(values, rng)


def _sample_unique_ints(
    n: int,
    low: int,
    high: int,
    rng: np.random.Generator,
    force_min_max: bool = True,
) -> np.ndarray:
    """
    Sample n unique integers from [low, high].

    The original masked identifiers are unique and range over a wider integer
    space than the row count. This mimics that behavior.
    """
    if high - low + 1 < n:
        raise ValueError("Range is too small for unique sampling.")

    if force_min_max and n >= 2:
        middle = rng.choice(
            np.arange(low + 1, high),
            size=n - 2,
            replace=False,
        )
        values = np.concatenate([[low, high], middle])
    else:
        values = rng.choice(np.arange(low, high + 1), size=n, replace=False)

    return _shuffle_values(values, rng).astype("int64")


def _sample_lognormal_amount(
    n: int,
    rng: np.random.Generator,
    median: float = 700.0,
    sigma: float = 1.15,
    min_value: float = 100.0,
    max_value: float = 2_800_000.0,
    outlier_rate: float = 0.018,
) -> np.ndarray:
    """
    Generate right-skewed positive dollar amounts.

    The real amount fields are highly skewed, have many moderate transactions,
    and contain a small number of very large outliers.
    """
    base = rng.lognormal(mean=np.log(median), sigma=sigma, size=n)
    base = np.maximum(base, min_value)

    outlier_mask = rng.random(n) < outlier_rate
    if outlier_mask.any():
        # Pareto tail creates occasional large transactions.
        tail = min_value * (1 + rng.pareto(a=1.35, size=outlier_mask.sum())) * 35
        base[outlier_mask] = np.maximum(base[outlier_mask], tail)

    base = np.clip(base, min_value, max_value)
    return np.round(base, 2).astype("float64")


def _assign_target(
    masked_bank_num: np.ndarray,
    masked_product_code: np.ndarray,
    rng: np.random.Generator,
    n_positive: int = 5954,
) -> np.ndarray:
    """
    Create return_target with approximately realistic risk structure.

    Target rate is calibrated to 5,954 positives out of 17,765 rows, matching the
    non-null RDI_DT / RETURN_REASON count provided by the project EDA.

    Risk is made mildly dependent on bank number and product code so the synthetic
    data preserves useful model signal.
    """
    bank_effect = {
        0: 0.32,
        1: -0.35,
        2: -0.65,
        3: -0.05,
        4: -0.55,
        5: -0.25,
        6: 0.35,
        7: -0.75,
    }
    product_risky = {1, 0, 4, 5, 6}

    logit = np.full(len(masked_bank_num), -0.72)
    logit += np.array([bank_effect.get(int(x), 0.0) for x in masked_bank_num])
    logit += np.array([0.18 if int(x) in product_risky else -0.10 for x in masked_product_code])
    logit += rng.normal(0, 0.25, size=len(masked_bank_num))

    # Select top-risk rows to hit the exact target count.
    cutoff_idx = np.argsort(logit)[-n_positive:]
    y = np.zeros(len(masked_bank_num), dtype="int64")
    y[cutoff_idx] = 1
    return y


def _generate_deposit_dates(
    n: int,
    rng: np.random.Generator,
    start: str = "2024-02-07",
    end: str = "2024-11-29",
) -> pd.Series:
    """
    Generate deposit dates within the observed date range.

    Weekdays receive more mass than weekends, and later months receive slightly
    more activity. This creates a realistic but synthetic monthly trend.
    """
    dates = pd.date_range(start=start, end=end, freq="D")
    weights = np.ones(len(dates), dtype=float)

    # Lower weekend activity.
    weights[dates.weekday >= 5] *= 0.35

    # Month-level seasonality; active from February through November.
    month_weight = {
        2: 0.85,
        3: 0.95,
        4: 1.00,
        5: 1.08,
        6: 0.95,
        7: 1.02,
        8: 1.06,
        9: 1.08,
        10: 1.12,
        11: 1.18,
    }
    weights *= np.array([month_weight[d.month] for d in dates])

    weights = weights / weights.sum()
    sampled = rng.choice(dates.to_numpy(), size=n, replace=True, p=weights)
    sampled = pd.to_datetime(sampled).strftime("%Y-%m-%d")
    return pd.Series(sampled, dtype="object")


def _generate_relationship_balance(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate relationship_balance as int64.

    Target shape:
    - no missing
    - median around 10k
    - many positive small/medium balances
    - 169 negative balances
    - 68 sentinel-like -99,999,999 values
    - one very large positive outlier near 2.7B
    """
    balance = rng.lognormal(mean=np.log(10_000), sigma=1.45, size=n)
    balance = np.round(balance).astype("int64")

    # Insert large positive outliers.
    large_idx = rng.choice(n, size=8, replace=False)
    large_values = np.array([
        2_745_575_018,
        18_520_601,
        9_375_121,
        6_647_064,
        6_607_629,
        4_500_000,
        3_200_000,
        2_100_000,
    ], dtype="int64")
    balance[large_idx] = large_values

    # 169 negative balances; 68 are sentinel-like values.
    neg_idx = rng.choice(np.setdiff1d(np.arange(n), large_idx), size=169, replace=False)
    sentinel_idx = neg_idx[:68]
    small_neg_idx = neg_idx[68:]

    balance[sentinel_idx] = -99_999_999
    balance[small_neg_idx] = -rng.choice(
        np.arange(1, 2_200),
        size=len(small_neg_idx),
        replace=True,
    )

    return balance.astype("int64")


def _generate_total_deposit_item_count(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate total_deposit_item_count with median 1 and a long right tail.
    """
    counts = rng.geometric(p=0.58, size=n)
    tail_mask = rng.random(n) < 0.018
    counts[tail_mask] += rng.zipf(a=1.65, size=tail_mask.sum()) * 3
    counts = np.clip(counts, 1, 907)

    # Force one observed-style maximum.
    counts[rng.integers(0, n)] = 907
    return counts.astype("int64")


def _generate_prev_transactions(
    n: int,
    item_amt: np.ndarray,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate prevtran1-10 and prevtrandate1-10 with exact non-null counts.

    Non-null counts:
    prevtran1      13378
    prevtran2      11152
    ...
    prevtran10      4996

    The date fields use the same structural missingness as the amount fields.
    """
    # Exact number of rows with history_count == h.
    # Derived from cumulative non-null counts.
    history_count_distribution = {
        0: 4387,
        1: 2226,
        2: 1474,
        3: 1024,
        4: 918,
        5: 708,
        6: 604,
        7: 560,
        8: 475,
        9: 393,
        10: 4996,
    }
    history_count = _values_from_counts(history_count_distribution, rng)
    if len(history_count) != n:
        raise ValueError("History count distribution does not sum to n.")

    prevtran = np.full((n, 10), np.nan, dtype="float64")
    prevdate = np.full((n, 10), np.nan, dtype="float64")

    for row in range(n):
        h = int(history_count[row])
        if h == 0:
            continue

        # Previous transaction amounts correlate weakly with current item amount.
        baseline = max(float(item_amt[row]), 100.0)
        amounts = _sample_lognormal_amount(
            h,
            rng,
            median=max(250.0, min(baseline * 0.75, 2_500.0)),
            sigma=1.15,
            min_value=100.0,
            max_value=475_000.0,
            outlier_rate=0.015,
        )
        prevtran[row, :h] = amounts

        # Days ago should generally increase for older transactions.
        # First transaction often happens same-day or within a week.
        first_gap = rng.choice(
            [0, 1, 2, 3, 4, 5, 7, 14, 21, 28],
            p=[0.22, 0.10, 0.08, 0.07, 0.05, 0.05, 0.16, 0.10, 0.08, 0.09],
        )
        if h == 1:
            days = np.array([first_gap])
        else:
            increments = rng.gamma(shape=1.55, scale=8.0, size=h - 1).round().astype(int)
            increments = np.maximum(increments, 1)
            days = np.concatenate([[first_gap], first_gap + np.cumsum(increments)])
        days = np.clip(days, 0, 292)
        prevdate[row, :h] = days.astype("float64")

    prevtran_df = pd.DataFrame(prevtran, columns=[f"prevtran{i}" for i in range(1, 11)])
    prevdate_df = pd.DataFrame(prevdate, columns=[f"prevtrandate{i}" for i in range(1, 11)])
    return prevtran_df, prevdate_df


def _generate_drawee_features(
    n: int,
    rng: np.random.Generator,
    non_null_count: int = 5529,
) -> pd.DataFrame:
    """
    Generate drawee_sum/cnt/avg/max/min with exact structural missingness.

    All five drawee variables share the same non-null mask.
    """
    data = np.full((n, 5), np.nan, dtype="float64")
    idx = rng.choice(n, size=non_null_count, replace=False)

    cnt = rng.geometric(p=0.48, size=non_null_count).astype(float)
    rare_tail = rng.random(non_null_count) < 0.01
    cnt[rare_tail] += rng.zipf(a=1.8, size=rare_tail.sum())
    cnt = np.clip(cnt, 1, 135).astype(float)

    avg = _sample_lognormal_amount(
        non_null_count,
        rng,
        median=820.0,
        sigma=1.05,
        min_value=101.0,
        max_value=475_000.0,
        outlier_rate=0.012,
    )

    spread = rng.lognormal(mean=np.log(1.12), sigma=0.25, size=non_null_count)
    max_amt = np.maximum(avg, avg * spread)
    min_amt = np.minimum(avg, avg / spread)
    min_amt = np.maximum(min_amt, 100.2)

    # Some cnt == 1 rows naturally have min=max=avg.
    single = cnt == 1
    max_amt[single] = avg[single]
    min_amt[single] = avg[single]

    drawee_sum = avg * cnt

    data[idx, 0] = np.round(drawee_sum, 2)
    data[idx, 1] = cnt
    data[idx, 2] = np.round(avg, 4)
    data[idx, 3] = np.round(np.clip(max_amt, 101.0, 475_000.0), 2)
    data[idx, 4] = np.round(np.clip(min_amt, 100.2, 475_000.0), 2)

    return pd.DataFrame(
        data,
        columns=["drawee_sum", "drawee_cnt", "drawee_avg", "drawee_max", "drawee_min"],
    )


def _generate_rdis(
    n: int,
    return_target: np.ndarray,
    rng: np.random.Generator,
    non_null_count: int = 5273,
) -> np.ndarray:
    """
    Generate rdis with exact non-null count.

    rdis is prior return deposit item count. It is more likely to be present for
    rows with return_target = 1, but not identical to the target.
    """
    rdis = np.full(n, np.nan, dtype="float64")

    pos_idx = np.where(return_target == 1)[0]
    neg_idx = np.where(return_target == 0)[0]

    n_from_pos = min(int(non_null_count * 0.60), len(pos_idx))
    selected_pos = rng.choice(pos_idx, size=n_from_pos, replace=False)
    selected_neg = rng.choice(neg_idx, size=non_null_count - n_from_pos, replace=False)
    idx = np.concatenate([selected_pos, selected_neg])
    rng.shuffle(idx)

    vals = rng.poisson(lam=1.2, size=non_null_count).astype(float)
    vals += (rng.random(non_null_count) < 0.10) * rng.integers(2, 8, size=non_null_count)
    vals = np.clip(vals, 0, 25).astype(float)

    rdis[idx] = vals
    return rdis


def _generate_rdi_dt_and_reason(
    deposit_dt: pd.Series,
    return_target: np.ndarray,
    rng: np.random.Generator,
) -> tuple[pd.Series, np.ndarray]:
    """
    Generate RDI_DT and RETURN_REASON.

    These fields are only populated when return_target == 1. The non-null count
    will therefore be exactly equal to the number of positive target rows.
    """
    n = len(return_target)
    rdi_dt = pd.Series([np.nan] * n, dtype="object")
    reason = np.full(n, np.nan, dtype="float64")

    pos_idx = np.where(return_target == 1)[0]
    deposit_as_dt = pd.to_datetime(deposit_dt)

    # Returned deposits occur shortly after deposit date.
    delay_days = rng.choice(
        np.arange(1, 31),
        size=len(pos_idx),
        replace=True,
        p=np.array(
            [0.10, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.05, 0.04,
             0.035, 0.03, 0.03, 0.025, 0.025, 0.02, 0.018, 0.016, 0.014, 0.012,
             0.010, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001]
        ) / np.array(
            [0.10, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.05, 0.04,
             0.035, 0.03, 0.03, 0.025, 0.025, 0.02, 0.018, 0.016, 0.014, 0.012,
             0.010, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001]
        ).sum(),
    )
    returned_dates = deposit_as_dt.iloc[pos_idx] + pd.to_timedelta(delay_days, unit="D")
    rdi_dt.iloc[pos_idx] = returned_dates.dt.strftime("%Y-%m-%d").to_numpy()

    # Exact RETURN_REASON distribution from EDA.
    reason_counts = {
        1.0: 3201,
        10.0: 1110,
        6.0: 801,
        16.0: 293,
        9.0: 152,
        2.0: 113,
        34.0: 110,
        12.0: 82,
        11.0: 36,
        25.0: 27,
        28.0: 5,
        29.0: 5,
        14.0: 4,
        33.0: 4,
        27.0: 3,
        8.0: 2,
        19.0: 1,
        31.0: 1,
        7.0: 1,
        38.0: 1,
        26.0: 1,
        32.0: 1,
    }
    reason_values = _values_from_counts(reason_counts, rng).astype("float64")
    if len(reason_values) != len(pos_idx):
        raise ValueError("RETURN_REASON counts must equal number of positive target rows.")
    reason[pos_idx] = reason_values

    return rdi_dt, reason


def generate_synthetic_citizens_raw_data(
    n_rows: int = 17_765,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic raw Citizens-style dataset.

    Parameters
    ----------
    n_rows:
        Number of rows. The default 17,765 exactly matches the project raw data.
        The exact EDA-based count constraints are calibrated for 17,765 rows.
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Synthetic raw dataset with the same 49 columns and intended dtypes.
    """
    if n_rows != 17_765:
        raise ValueError(
            "This generator is calibrated to exact count constraints for n_rows=17,765. "
            "Use n_rows=17_765 to preserve the original missingness/count trends."
        )

    rng = np.random.default_rng(random_state)
    n = n_rows

    # -----------------------------
    # Customer & account variables
    # -----------------------------
    masked_dep_acct_num = _sample_unique_ints(n, 0, 24_690, rng, force_min_max=True)
    masked_id = _sample_unique_ints(n, 0, 31_651, rng, force_min_max=True)

    masked_bank_num = _values_from_counts(
        {
            0: 15207,
            3: 797,
            1: 706,
            6: 420,
            4: 324,
            2: 214,
            5: 65,
            7: 32,
        },
        rng,
    ).astype("int64")

    masked_account_type = _values_from_counts(
        {
            0: 14610,
            1: 1733,
            2: 1257,
            3: 140,
            4: 25,
        },
        rng,
    ).astype("int64")

    masked_product_code = _values_from_counts(
        {
            1: 7855,
            0: 4110,
            2: 1182,
            4: 1147,
            5: 1069,
            6: 461,
            7: 423,
            9: 395,
            8: 338,
            12: 245,
            15: 131,
            11: 71,
            17: 70,
            13: 59,
            10: 56,
            19: 49,
            21: 34,
            16: 21,
            18: 16,
            20: 10,
            14: 8,
            22: 6,
            3: 5,
            23: 4,
        },
        rng,
    ).astype("int64")

    bucket_values = (
        ["5000+"] * 8411
        + ["2000-5000"] * 3949
        + ["0-1000"] * 3674
        + ["1000-2000"] * 1704
        + [np.nan] * 27
    )
    bucket_days_since_open = pd.Series(_shuffle_values(bucket_values, rng), dtype="object")

    number_of_owners = _values_from_counts(
        {
            1: 8713,
            2: 6411,
            0: 1972,
            3: 557,
            4: 97,
            5: 8,
            6: 3,
            8: 2,
            7: 2,
        },
        rng,
    ).astype("int64")

    # -----------------------------
    # Transaction variables
    # -----------------------------
    deposit_dt = _generate_deposit_dates(n, rng)
    month_num = pd.to_datetime(deposit_dt).dt.month.astype("int64").to_numpy()

    item_amt = _sample_lognormal_amount(
        n,
        rng,
        median=725.0,
        sigma=1.20,
        min_value=100.01,
        max_value=2_800_000.0,
        outlier_rate=0.025,
    )

    # Force exact-ish observed range.
    item_amt[rng.integers(0, n)] = 2_800_000.00
    item_amt[rng.integers(0, n)] = 100.01

    add_on = rng.gamma(shape=1.1, scale=650.0, size=n)
    same_item_mask = rng.random(n) < 0.58
    total_deposit_amount = np.where(same_item_mask, item_amt, item_amt + add_on)
    total_deposit_amount = np.maximum(total_deposit_amount, 100.56)
    total_deposit_amount = np.clip(total_deposit_amount, 100.56, 2_800_000.0)
    total_deposit_amount = np.round(total_deposit_amount, 2).astype("float64")
    total_deposit_amount[rng.integers(0, n)] = 2_800_000.00
    total_deposit_amount[rng.integers(0, n)] = 100.56

    channel = pd.Series(["TELLER"] * n, dtype="object")
    relationship_balance = _generate_relationship_balance(n, rng)

    oao_flg = pd.Series(_values_from_counts({"N": 17527, "Y": 238}, rng), dtype="object")
    onus_ind = pd.Series(_values_from_counts({"F": 14771, "T": 2994}, rng), dtype="object")
    treasury_check_ind = pd.Series(_values_from_counts({"N": 17673, "Y": 92}, rng), dtype="object")
    heloc_ind = pd.Series(_values_from_counts({"N": 17733, "Y": 32}, rng), dtype="object")

    max_deposit_amount30d = np.maximum(
        item_amt,
        _sample_lognormal_amount(
            n,
            rng,
            median=1_100.0,
            sigma=1.20,
            min_value=100.0,
            max_value=2_800_000.0,
            outlier_rate=0.02,
        ),
    )
    max_deposit_amount30d = np.round(max_deposit_amount30d, 2).astype("float64")

    total_deposit_item_count = _generate_total_deposit_item_count(n, rng)

    # -----------------------------
    # Target first, then dependent leakage/context variables
    # -----------------------------
    return_target = _assign_target(
        masked_bank_num=masked_bank_num,
        masked_product_code=masked_product_code,
        rng=rng,
        n_positive=5954,
    )

    rdis = _generate_rdis(n, return_target, rng, non_null_count=5273)

    prevtran_df, prevdate_df = _generate_prev_transactions(n, item_amt, rng)
    drawee_df = _generate_drawee_features(n, rng, non_null_count=5529)
    RDI_DT, RETURN_REASON = _generate_rdi_dt_and_reason(deposit_dt, return_target, rng)

    over_draft_amount = _values_from_counts(
        {
            0: 16669,
            1: 390,
            2: 208,
            3: 100,
            4: 69,
            5: 64,
            7: 37,
            6: 34,
            8: 27,
            9: 23,
            12: 21,
            10: 20,
            17: 13,
            15: 13,
            11: 13,
            14: 10,
            13: 10,
            21: 9,
            19: 9,
            18: 7,
            20: 7,
            16: 7,
            22: 5,
        },
        rng,
    ).astype("int64")

    # -----------------------------
    # Assemble
    # -----------------------------
    df = pd.DataFrame(
        {
            "masked_dep_acct_num": masked_dep_acct_num,
            "masked_bank_num": masked_bank_num,
            "masked_account_type": masked_account_type,
            "masked_id": masked_id,
            "masked_product_code": masked_product_code,
            "bucket_days_since_open": bucket_days_since_open,
            "number_of_owners": number_of_owners,
            "total_deposit_amount": total_deposit_amount,
            "item_amt": item_amt,
            "deposit_dt": deposit_dt,
            "channel": channel,
            "relationship_balance": relationship_balance,
            "oao_flg": oao_flg,
            "onus_ind": onus_ind,
            "treasury_check_ind": treasury_check_ind,
            "heloc_ind": heloc_ind,
            "rdis": rdis,
            "max_deposit_amount30d": max_deposit_amount30d,
            "total_deposit_item_count": total_deposit_item_count,
            "RDI_DT": RDI_DT,
            "RETURN_REASON": RETURN_REASON,
            "return_target": return_target,
            "over_draft_amount": over_draft_amount,
            "month_num": month_num,
        }
    )

    # Insert grouped dataframes.
    for col in prevtran_df.columns:
        df[col] = prevtran_df[col]
    for col in prevdate_df.columns:
        df[col] = prevdate_df[col]
    for col in drawee_df.columns:
        df[col] = drawee_df[col]

    df = df[RAW_COLUMN_ORDER].copy()

    # Enforce dtypes.
    int_cols = [
        "masked_dep_acct_num",
        "masked_bank_num",
        "masked_account_type",
        "masked_id",
        "masked_product_code",
        "number_of_owners",
        "relationship_balance",
        "total_deposit_item_count",
        "return_target",
        "over_draft_amount",
        "month_num",
    ]
    float_cols = [
        "total_deposit_amount",
        "item_amt",
        "rdis",
        "max_deposit_amount30d",
        *[f"prevtran{i}" for i in range(1, 11)],
        *[f"prevtrandate{i}" for i in range(1, 11)],
        "drawee_sum",
        "drawee_cnt",
        "drawee_avg",
        "drawee_max",
        "drawee_min",
        "RETURN_REASON",
    ]
    object_cols = [
        "bucket_days_since_open",
        "deposit_dt",
        "channel",
        "oao_flg",
        "onus_ind",
        "treasury_check_ind",
        "heloc_ind",
        "RDI_DT",
    ]

    for col in int_cols:
        df[col] = df[col].astype("int64")
    for col in float_cols:
        df[col] = df[col].astype("float64")
    for col in object_cols:
        df[col] = df[col].astype("object")

    return df


def validate_synthetic_citizens_raw_data(df: pd.DataFrame) -> None:
    """
    Print validation checks against the intended raw-data schema.
    """
    print("Shape:", df.shape)
    print("\nDtypes:")
    print(df.dtypes.value_counts())

    print("\nColumn order matches expected:", list(df.columns) == RAW_COLUMN_ORDER)

    print("\nNon-null counts:")
    print(df.info())

    print("\nMissing counts for variables with missingness:")
    missing = df.isna().sum()
    print(missing[missing > 0])

    print("\nSelected distribution checks:")
    print("masked_bank_num:")
    print(df["masked_bank_num"].value_counts().sort_index())

    print("\nmasked_account_type:")
    print(df["masked_account_type"].value_counts().sort_index())

    print("\nbucket_days_since_open:")
    print(df["bucket_days_since_open"].value_counts(dropna=False))

    print("\nreturn_target:")
    print(df["return_target"].value_counts().sort_index())

    print("\nRETURN_REASON:")
    print(df["RETURN_REASON"].value_counts().sort_index())

    print("\nover_draft_amount:")
    print(df["over_draft_amount"].value_counts().sort_index())

    print("\nAmount summaries:")
    print(df[["total_deposit_amount", "item_amt", "relationship_balance"]].describe())


if __name__ == "__main__":
    synthetic_df = generate_synthetic_citizens_raw_data(random_state=42)
    validate_synthetic_citizens_raw_data(synthetic_df)

    # Save synthetic raw data under the project raw-data directory.
    output_path = Path("data/raw/synthetic_citizens_raw_data.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    synthetic_df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
