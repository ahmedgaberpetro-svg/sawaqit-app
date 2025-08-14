
import streamlit as st
import math, time, random
from typing import List, Optional
import pandas as pd

st.set_page_config(page_title="توزيع السواقط", page_icon="💧", layout="wide")

# ===================== Pricing helpers =====================
def price_mills_for_units(u: int, p1: float, p2: float, p3: float, fee: float, step: float) -> int:
    """Return value in mills (0.001 EGP) for quantity 'u' units. One unit = `step` m3."""
    q = u * step
    # tiered pricing
    if q <= 30.0:
        v = q * p1
    elif q <= 60.0:
        v = 30.0 * p1 + (q - 30.0) * p2
    else:
        v = 30.0 * p1 + 30.0 * p2 + (q - 60.0) * p3
    v += fee  # monthly fee always counted
    return int(round(v * 1000))

def to_units(q: float, step: float) -> int:
    return int(round(q / step))

def from_units(u: int, step: float) -> float:
    return round(u * step, 1)  # show one decimal place

# ===================== Core search =====================
def random_split(total_units: int, n: int) -> List[int]:
    """Random integer split that sums to total_units across n buckets (can be zero)."""
    if n <= 0:
        return []
    # simple "stick breaking" approach using random weights
    w = [random.random() + 0.01 for _ in range(n)]
    s = sum(w)
    base = [int((wi / s) * total_units) for wi in w]
    rem = total_units - sum(base)
    # give remainder to largest fractional weights
    frac = [((wi / s) * total_units) - bi for wi, bi in zip(w, base)]
    order = sorted(range(n), key=lambda i: frac[i], reverse=True)
    for i in range(rem):
        base[order[i % n]] += 1
    return base

def greedy_improve(q, v, locks, target_mills, p1, p2, p3, fee, step, time_budget):
    """Move 1 unit at a time between months to get closer to target within time_budget seconds."""
    t0 = time.time()
    n = len(q)
    while time.time() - t0 <= time_budget:
        cur_sum = sum(v)
        need_up = cur_sum < target_mills

        # compute delta arrays
        very_big = 10**12
        gain = [-(very_big)] * n
        loss = [very_big] * n
        for i in range(n):
            # we can always try to add a unit to month i (even if locked, adding would violate lock)
            # but if locked quantity -> skip adjustments
            if locks[i] is None:
                gain[i] = price_mills_for_units(q[i] + 1, p1, p2, p3, fee, step) - v[i]
                loss[i] = v[i] - price_mills_for_units(max(q[i]-1, 0), p1, p2, p3, fee, step) if q[i] > 0 else very_big

        best_i = -1
        best_j = -1
        best_net = -(very_big) if need_up else very_big

        for i in range(n):
            if locks[i] is not None:
                continue
            for j in range(n):
                if i == j or locks[j] is not None:
                    continue
                if need_up:
                    if q[j] > 0:
                        net = gain[i] - loss[j]
                        if net > best_net:
                            best_net = net
                            best_i, best_j = i, j
                else:
                    if q[i] > 0:
                        net = gain[j] - loss[i]
                        if net < best_net:
                            best_net = net
                            best_i, best_j = j, i

        if best_i == -1 or best_j == -1:
            break  # no move improves

        # perform move
        q[best_i] += 1
        q[best_j] -= 1
        v[best_i] = price_mills_for_units(q[best_i], p1, p2, p3, fee, step)
        v[best_j] = price_mills_for_units(q[best_j], p1, p2, p3, fee, step)

        # small early stop if perfect
        if sum(v) == target_mills:
            break

def solve_distribution(total_q, target_value, p1, p2, p3, fee, months, locks_qty, step,
                       max_restarts, time_budget):
    """Search multiple random restarts; return best quantities/values and diff in mills."""
    target_mills = int(round(target_value * 1000))
    total_units = to_units(total_q, step)

    # Prepare locks (as units)
    locks = [None] * months
    locked_units = 0
    for i in range(months):
        v = locks_qty[i]
        if v is not None and v != "":
            try:
                qf = float(v)
            except:
                qf = None
            if qf is not None:
                u = to_units(max(qf, 0.0), step)
                locks[i] = u
                locked_units += u

    if locked_units > total_units:
        return None, None, None, "الكمية المقفولة أكبر من الإجمالي. قلل القيم المقفولة."

    best = None
    best_q = None
    best_v = None

    free_units_total = total_units - locked_units
    free_indices = [i for i in range(months) if locks[i] is None]

    for _ in range(max_restarts):
        # starting point
        q = [0] * months
        for i in range(months):
            q[i] = locks[i] if locks[i] is not None else 0
        if free_indices:
            alloc = random_split(free_units_total, len(free_indices))
            for idx, ai in zip(free_indices, alloc):
                q[idx] = ai

        v = [price_mills_for_units(qi, p1, p2, p3, fee, step) for qi in q]

        # local greedy
        greedy_improve(q, v, locks, target_mills, p1, p2, p3, fee, step, time_budget)

        diff = abs(sum(v) - target_mills)
        if (best is None) or (diff < best):
            best = diff
            best_q = q[:]
            best_v = v[:]
            if best == 0:
                break

    # build dataframe
    months_idx = list(range(1, months + 1))
    out = pd.DataFrame({
        "Month": months_idx,
        "Quantity": [from_units(u, step) for u in best_q],
        "Value": [round(m / 1000.0, 3) for m in best_v],
    })
    totals = pd.DataFrame({
        "Month": ["Totals"],
        "Quantity": [round(sum(out["Quantity"]), 1)],
        "Value": [round(sum(out["Value"]), 3)],
    })
    out = pd.concat([out, totals], ignore_index=True)

    return out, best / 1000.0, sum(best_v) / 1000.0, None

# ===================== UI =====================
st.title("💧 موزّع السواقط (نسخة ويب)")
st.caption("إدخال المعطيات → احسب → جرّب احتمال جديد لو عايز تركيبة مختلفة بنفس الشروط.")

with st.sidebar:
    st.header("المعطيات")
    col1, col2 = st.columns(2)
    with col1:
        total_q = st.number_input("إجمالي الكمية (م³)", value=392.9, step=0.1, format="%.1f")
        months = st.number_input("عدد الشهور", min_value=1, max_value=12, value=8, step=1)
        step = st.selectbox("دقة الخطوة (م³)", [0.05, 0.1], index=1, help="كل نقلة بين الشهور بالقيمة المختارة.")
    with col2:
        target_value = st.number_input("القيمة المطلوبة (جنيه)", value=1533.145, step=0.001, format="%.3f")
        fee = st.number_input("الرسوم الشهرية", value=17.5, step=0.1)
    p1 = st.number_input("سعر شريحة 1", value=3.036, step=0.001, format="%.3f")
    p2 = st.number_input("سعر شريحة 2", value=4.036, step=0.001, format="%.3f")
    p3 = st.number_input("سعر شريحة 3", value=5.036, step=0.001, format="%.3f")

    st.markdown("---")
    mode = st.radio("وضع التشغيل", ["سريع", "دقيق"], index=0, help="السريع للمعاينة، الدقيق يحاول يقرّب أكثر")
    if mode == "سريع":
        max_restarts = 120
        time_budget = 0.12
    else:
        max_restarts = 600
        time_budget = 2.5

    st.markdown("---")
    st.write("**تثبيت شهور معيّنة (اختياري):** اكتب كمية م³ في الخانة المقابلة للشهر علشان تتثبت.")
    locks_df = pd.DataFrame({"Month": list(range(1, int(months)+1)), "Lock_Quantity": [None]*int(months)})
    locks_df = st.data_editor(
        locks_df,
        hide_index=True,
        num_rows="fixed",
        column_config={
            "Lock_Quantity": st.column_config.NumberColumn("Lock Quantity (m³)", format="%.1f", step=0.1)
        },
        key="locks_editor",
    )

# buttons row
b1, b2, b3 = st.columns([1,1,2])
with b1:
    run = st.button("احسب", type="primary")
with b2:
    try_new = st.button("جرّب احتمال جديد")

if "seed" not in st.session_state:
    st.session_state.seed = None

if run or try_new:
    if try_new:
        st.session_state.seed = random.randint(1, 10**9)
        random.seed(st.session_state.seed)
    else:
        # fresh seed each run
        st.session_state.seed = int(time.time() * 1000) % (10**9)
        random.seed(st.session_state.seed)

    locks_list = [row["Lock_Quantity"] if not (row["Lock_Quantity"] is None) else None for _, row in locks_df.iterrows()]

    with st.spinner("جارٍ البحث عن أفضل توزيع..."):
        df, diff, achieved, err = solve_distribution(
            total_q=total_q,
            target_value=target_value,
            p1=p1, p2=p2, p3=p3,
            fee=fee,
            months=int(months),
            locks_qty=locks_list,
            step=float(step),
            max_restarts=max_restarts,
            time_budget=time_budget
        )
    if err:
        st.error(err)
    else:
        st.subheader("النتيجة")
        st.dataframe(df, use_container_width=True)
        colx, coly, colz = st.columns(3)
        with colx:
            st.metric("إجمالي القيم الناتجة", f"{achieved:.3f} جنيه")
        with coly:
            st.metric("الفرق عن الهدف", f"{diff:.3f} جنيه")
        with colz:
            st.write(f"Seed: `{st.session_state.seed}`  |  Restarts: {max_restarts}  |  Budget: {time_budget}s")

        # Download CSV
        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("تحميل CSV", csv, file_name="result.csv", mime="text/csv")
else:
    st.info("ادخل القيم من الشريط الجانبي ثم اضغط **احسب**.")

