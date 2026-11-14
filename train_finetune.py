"""
ARGUS: дообучение готовой нейросети на новых данных + выгрузка весов для ESP32-C3.

Что делает:
  1. Берёт готовые argus_ppm_model.joblib и argus_scaler.joblib (ваши веса).
  2. Скейлер НЕ пересчитывается (иначе старые веса станут бессмысленными).
  3. Дообучает сеть на НОВЫХ данных с лабораторными метками (титрование)
     вместе со СТАРЫМИ данными (чтобы сеть не "забыла" прошлое).
  4. Проверяет на тестовой части: сохраняет результат только если стало не хуже.
  5. Пишет argus_weights.h для скетча и сверяет C-формулу с sklearn.

Запуск:
    python train_finetune.py
    python train_finetune.py --new new_data.csv --lr 1e-4 --epochs 400

Формат new_data.csv (первая строка - заголовки):
    gas_raw_clean,temp_c,humidity_pct,lab_molar_mol_l
    62.4,25.8,55.1,6.02e-4
  gas_raw_clean : значение в шкале Nano 0-1023 (на ESP это ADC10eq из Serial)
  lab_molar_mol_l : результат титрования, моль/л
  вместо lab_molar_mol_l можно дать log_ppm (log10 от PPM).

ВАЖНО: колонки ppm / molar_mol_l из argus_live_dataset.csv - это ответы самой
нейросети, а не лаборатории. Обучать на них нельзя.
"""
import argparse
import os
import shutil
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

# ---- Константы (те же, что в live_argus.py и скетче) ----
R_L = 10.0
R_0 = 76.8
V_CC = 5.0
ADC_MAX = 1023.0
P_ATM = 101325.0
R_GAS = 8.314
FEATURES = ["Xlog", "temp_c", "humidity_pct"]

OLD_FILES = ["egg_decay_dataset.csv", "combined_clean.csv"]
MODEL_FILE = "argus_ppm_model.joblib"
SCALER_FILE = "argus_scaler.joblib"
HEADER_FILE = "argus_weights.h"


def molar_to_log_ppm(c_m, temp_c):
    """Обратная формула к calculate_molar_concentration из live_argus.py."""
    temp_k = temp_c + 273.15
    ppm = c_m * 1000.0 * R_GAS * temp_k / (P_ATM * 1e-6)
    return np.log10(ppm)


def add_xlog(df):
    vout = np.clip(df["gas_raw"] / ADC_MAX * V_CC, 1e-3, V_CC - 1e-3)
    rs = R_L * (V_CC - vout) / vout
    df["Xlog"] = np.log10(rs / R_0)
    return df


def load_csv(path, need_label):
    df = pd.read_csv(path)
    gas_col = "gas_raw_clean" if "gas_raw_clean" in df.columns else "gas_raw"
    if gas_col not in df.columns:
        raise ValueError(f"{path}: нет колонки gas_raw_clean / gas_raw")
    for c in [gas_col, "temp_c", "humidity_pct"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[gas_col, "temp_c", "humidity_pct"])
    # строки с нулевым климатом - это сбой DHT22, их отбрасываем
    df = df[(df["humidity_pct"] > 0) & (df["temp_c"] > 0)].copy()
    df["gas_raw"] = df[gas_col]
    df = add_xlog(df)

    if "log_ppm" in df.columns:
        df["target_y"] = pd.to_numeric(df["log_ppm"], errors="coerce")
    elif "lab_molar_mol_l" in df.columns:
        c_m = pd.to_numeric(df["lab_molar_mol_l"], errors="coerce")
        df["target_y"] = molar_to_log_ppm(c_m, df["temp_c"])
    elif need_label:
        raise ValueError(
            f"{path}: нет лабораторной метки (lab_molar_mol_l или log_ppm). "
            "Колонки ppm/molar_mol_l из live-датасета не подходят: это ответы "
            "самой нейросети, а не титрование."
        )
    else:
        # как в argus_train.py: запасная калибровка для старых файлов без метки
        cf = np.maximum(
            1.0 - 0.010 * (df["temp_c"] - 20.0) + 0.003 * (df["humidity_pct"] - 65.0), 0.1)
        df["target_y"] = -1.20 * (df["Xlog"] - np.log10(cf)) + 1.50

    df = df.dropna(subset=["target_y"])
    return df[FEATURES + ["target_y"]].reset_index(drop=True)


def numpy_forward(model, scaler, X):
    """Та же математика, что в скетче ESP32 (для самопроверки)."""
    a = (X - scaler.mean_) / scaler.scale_
    for i, (W, b) in enumerate(zip(model.coefs_, model.intercepts_)):
        a = a @ W + b
        if i < len(model.coefs_) - 1:
            a = np.maximum(a, 0)
    return a.ravel()


def export_header(model, scaler, path):
    def arr(name, a):
        flat = ", ".join(f"{v:.8f}f" for v in np.ravel(a))
        return f"const float {name}[{np.size(a)}] = {{{flat}}};\n"

    with open(path, "w", encoding="utf-8") as f:
        f.write("#pragma once\n// Сгенерировано train_finetune.py\n")
        f.write(arr("W1", model.coefs_[0]))
        f.write(arr("B1", model.intercepts_[0]))
        f.write(arr("W2", model.coefs_[1]))
        f.write(arr("B2", model.intercepts_[1]))
        f.write(arr("W3", model.coefs_[2]))
        f.write(arr("B3", model.intercepts_[2]))
        f.write(arr("SC_MEAN", scaler.mean_))
        f.write(arr("SC_SCALE", scaler.scale_))


def set_learning_rate(model, lr):
    model.learning_rate_init = lr
    opt = getattr(model, "_optimizer", None)
    if opt is not None and hasattr(opt, "learning_rate_init"):
        opt.learning_rate_init = lr
        if hasattr(opt, "learning_rate"):
            opt.learning_rate = lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", default="new_data.csv", help="новые данные с титрованием")
    ap.add_argument("--old", nargs="*", default=OLD_FILES, help="старые датасеты (повтор)")
    ap.add_argument("--lr", type=float, default=1e-4, help="шаг дообучения (мал, чтобы не ломать веса)")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--new_weight", type=int, default=3,
                    help="во сколько раз повторить новые строки (они важнее)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for f in (MODEL_FILE, SCALER_FILE):
        if not os.path.exists(f):
            sys.exit(f"Нет файла {f}. Положите рядом готовые веса.")
    if not os.path.exists(args.new):
        sys.exit(f"Нет файла новых данных {args.new} (формат смотри в начале скрипта).")

    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)

    # ---- данные ----
    new_df = load_csv(args.new, need_label=True)
    old_parts = []
    for p in args.old:
        if os.path.exists(p):
            old_parts.append(load_csv(p, need_label=False))
        else:
            print(f"[нет старого файла] {p} - пропускаю")
    old_df = pd.concat(old_parts, ignore_index=True) if old_parts else pd.DataFrame(columns=FEATURES + ["target_y"])
    print(f"Новых строк: {len(new_df)}, старых строк: {len(old_df)}")
    if len(new_df) < 10:
        print("[ВНИМАНИЕ] Новых данных мало (<10). Результат дообучения будет ненадёжным.")

    # ---- разбиение: тест берём и из новых, и из старых данных ----
    rng = np.random.RandomState(args.seed)

    def split(df, frac=0.2):
        idx = rng.permutation(len(df))
        n_te = max(1, int(len(df) * frac)) if len(df) > 4 else 0
        return df.iloc[idx[n_te:]], df.iloc[idx[:n_te]]

    new_tr, new_te = split(new_df)
    old_tr, old_te = split(old_df) if len(old_df) else (old_df, old_df)

    train = pd.concat([old_tr] + [new_tr] * args.new_weight, ignore_index=True)
    X_tr = scaler.transform(train[FEATURES].values)   # скейлер старый, не fit!
    y_tr = train["target_y"].values

    def mae(df):
        if len(df) == 0:
            return float("nan")
        return mean_absolute_error(df["target_y"].values,
                                   model.predict(scaler.transform(df[FEATURES].values)))

    before_new, before_old = mae(new_te), mae(old_te)
    print(f"\nДО дообучения (MAE по log10 PPM):  новые тест = {before_new:.3f}   старые тест = {before_old:.3f}")

    # ---- резервная копия готовых весов ----
    shutil.copy(MODEL_FILE, "argus_ppm_model_backup.joblib")
    shutil.copy(SCALER_FILE, "argus_scaler_backup.joblib")
    best_state = ([w.copy() for w in model.coefs_], [b.copy() for b in model.intercepts_])

    def score():
        s_new, s_old = mae(new_te), mae(old_te)
        vals = [v for v in (s_new, s_old) if not np.isnan(v)]
        return float(np.mean(vals)), s_new, s_old

    best, _, _ = score()
    bad = 0

    # ---- дообучение ----
    model.early_stopping = False   # partial_fit не работает с ним; свой останов ниже
    if getattr(model, "best_loss_", None) is None:
        model.best_loss_ = np.inf  # у моделей с early_stopping это поле пустое
    set_learning_rate(model, args.lr)
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(X_tr))
        for start in range(0, len(order), 32):
            b = order[start:start + 32]
            model.partial_fit(X_tr[b], y_tr[b])
        cur, _, _ = score()
        if cur < best - 1e-4:
            best, bad = cur, 0
            best_state = ([w.copy() for w in model.coefs_], [b.copy() for b in model.intercepts_])
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Ранняя остановка на эпохе {epoch}")
                break

    model.coefs_, model.intercepts_ = best_state
    _, after_new, after_old = score()
    print(f"ПОСЛЕ дообучения (MAE по log10 PPM): новые тест = {after_new:.3f}   старые тест = {after_old:.3f}")

    # ---- не сохраняем, если стало хуже на старых данных ----
    forgot = (not np.isnan(before_old)) and after_old > before_old + 0.15
    improved = np.isnan(before_new) or after_new <= before_new
    if forgot or not improved:
        print("\n[ОТКАТ] Дообучение не улучшило модель или испортило старые результаты.")
        print("Готовые веса оставлены без изменений. Попробуйте --lr 5e-5 или больше новых данных.")
        export_header(joblib.load("argus_ppm_model_backup.joblib"), scaler, HEADER_FILE)
        print(f"{HEADER_FILE} записан из СТАРЫХ весов.")
        return

    # ---- сохранение ----
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    export_header(model, scaler, HEADER_FILE)

    # самопроверка: формула для ESP совпадает с sklearn
    Xc = scaler.transform(new_df[FEATURES].values) * scaler.scale_ + scaler.mean_
    diff = np.max(np.abs(numpy_forward(model, scaler, Xc) - model.predict(scaler.transform(Xc))))
    print(f"\nСамопроверка формулы для ESP: макс. расхождение = {diff:.2e} (должно быть ~1e-6)")
    print(f"Готово: {MODEL_FILE}, {SCALER_FILE}, {HEADER_FILE}")
    print("Копию старых весов смотри в *_backup.joblib")
    print(f"Скопируйте {HEADER_FILE} в папку argus_esp и перепрошейте плату.")


if __name__ == "__main__":
    main()
