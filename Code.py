import os

os.environ["TF_DETERMINISTIC_OPS"] = "1"

import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Dense, Dropout, Softmax, Multiply, Lambda

tf.random.set_seed(42)
np.random.seed(42)

INPUT_FILE = "runoff_dataset.xlsx"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXPECTED_COLS = [
    "data",
    "snowmelt",
    "pet",
    "precipitation",
    "surface_sensible_heat_flux",
    "snowfall",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "runoff",
]

DATE_COL = "data"
CLIMATE_COLS = [
    "snowmelt", "pet", "precipitation", "surface_sensible_heat_flux",
    "snowfall", "temperature", "u_component_of_wind", "v_component_of_wind",
]
RESID_COL = "runoff"

TRAIN_START, TRAIN_END = 1980, 1996
VAL_START, VAL_END = 1997, 2000
SIM_START, SIM_END = 2001, 2020

N_IN = 6
N_OUT = 3
STRIDE = 1

STRICT_COLS = True


def parse_dates(series):
    s = series.astype(str).str.strip()
    zh = s.str.replace("年", "-", regex=False).str.replace("月", "", regex=False)
    dt = pd.to_datetime(zh, format="%Y-%m", errors="coerce")
    if dt.isna().any():
        dt = dt.fillna(pd.to_datetime(s, errors="coerce"))
    return dt


def create_supervised_sequences(feature_arr, target_arr, n_in, n_out, stride=1):
    feature_arr = np.asarray(feature_arr, dtype=np.float32)
    target_arr = np.asarray(target_arr, dtype=np.float32).reshape(-1)
    n = len(feature_arr)
    max_i = n - n_in - n_out + 1
    if max_i <= 0:
        return (np.empty((0, n_in, feature_arr.shape[1]), np.float32),
                np.empty((0, n_out), np.float32))
    X, Y = [], []
    for i in range(0, max_i, stride):
        X.append(feature_arr[i:i + n_in, :])
        Y.append(target_arr[i + n_in:i + n_in + n_out])
    return np.stack(X), np.stack(Y)


def build_lstm_attention_model(n_in, n_features, n_out):
    inputs = Input(shape=(n_in, n_features))
    x = LSTM(64, return_sequences=True)(inputs)

    score = Dense(1)(x)
    weights = Softmax(axis=1)(score)
    weighted = Multiply()([x, weights])
    context = Lambda(lambda t: tf.reduce_sum(t, axis=1))(weighted)

    context = Dropout(0.3)(context)
    context = Dense(64, activation="relu")(context)
    outputs = Dense(n_out)(context)

    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model


dataset = pd.read_excel(INPUT_FILE)

if STRICT_COLS:
    actual = [str(c).strip() for c in dataset.columns[:len(EXPECTED_COLS)]]
    if actual != EXPECTED_COLS:
        raise ValueError(
            "Input columns (A..J) do not match the expected README names/order.\n"
            f"Expected: {EXPECTED_COLS}\n"
            f"Actual:   {actual}\n"
            "Set STRICT_COLS = False to map the first 10 columns by position instead."
        )
else:
    dataset = dataset.iloc[:, :len(EXPECTED_COLS)].copy()
    dataset.columns = EXPECTED_COLS

dataset[DATE_COL] = parse_dates(dataset[DATE_COL])
dataset = dataset.sort_values(DATE_COL).reset_index(drop=True)
dataset["Year"] = dataset[DATE_COL].dt.year
dataset["Month"] = dataset[DATE_COL].dt.month

train_df = dataset[(dataset.Year >= TRAIN_START) & (dataset.Year <= TRAIN_END)].copy()
val_df = dataset[(dataset.Year >= VAL_START) & (dataset.Year <= VAL_END)].copy()
sim_df = dataset[(dataset.Year >= SIM_START) & (dataset.Year <= SIM_END)].copy()

train_df = train_df.dropna(subset=CLIMATE_COLS + [RESID_COL]).copy()
val_df = val_df.dropna(subset=CLIMATE_COLS + [RESID_COL]).copy()
sim_df = sim_df.dropna(subset=CLIMATE_COLS).reset_index(drop=True)

x_scaler = MinMaxScaler().fit(train_df[CLIMATE_COLS].values)
y_scaler = MinMaxScaler().fit(train_df[[RESID_COL]].values)


def scaled_xy(d):
    x = x_scaler.transform(d[CLIMATE_COLS].values).astype(np.float32)
    y = y_scaler.transform(d[[RESID_COL]].values).reshape(-1).astype(np.float32)
    return x, y


Xtr_m, ytr_m = scaled_xy(train_df)
Xva_m, yva_m = scaled_xy(val_df)

X_train, Y_train = create_supervised_sequences(Xtr_m, ytr_m, N_IN, N_OUT, STRIDE)
X_val, Y_val = create_supervised_sequences(Xva_m, yva_m, N_IN, N_OUT, STRIDE)

if X_train.shape[0] == 0 or X_val.shape[0] == 0:
    raise ValueError("Not enough data to build supervised samples; check periods / N_IN / N_OUT.")

n_features = X_train.shape[2]
print(f"Train: {X_train.shape}, Val: {X_val.shape}, n_features={n_features}")

model = build_lstm_attention_model(N_IN, n_features, N_OUT)
model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
]
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=150,
    batch_size=16,
    shuffle=True,
    verbose=1,
    callbacks=callbacks,
)

model_path = os.path.join(OUTPUT_DIR, "wh_dl_residual.keras")
model.save(model_path)
with open(os.path.join(OUTPUT_DIR, "x_scaler.pkl"), "wb") as f:
    pickle.dump(x_scaler, f)
with open(os.path.join(OUTPUT_DIR, "y_scaler.pkl"), "wb") as f:
    pickle.dump(y_scaler, f)

val_pred = y_scaler.inverse_transform(model.predict(X_val, verbose=0).reshape(-1, 1)).reshape(-1)
val_true = y_scaler.inverse_transform(Y_val.reshape(-1, 1)).reshape(-1)
vr = np.corrcoef(val_true, val_pred)[0, 1]
vrmse = float(np.sqrt(np.mean((val_pred - val_true) ** 2)))
print(f"Validation (1997-2000) residual: R2={vr ** 2:.3f}, RMSE={vrmse:.4f}")

sim_start_date = pd.Timestamp(f"{SIM_START}-01-01")
hist = dataset[dataset[DATE_COL] < sim_start_date].dropna(subset=CLIMATE_COLS).tail(N_IN)
if len(hist) < N_IN:
    raise ValueError("Not enough history before 2001 to build the initial input window.")

window = x_scaler.transform(hist[CLIMATE_COLS].values).astype(np.float32)
sim_climate_scaled = x_scaler.transform(sim_df[CLIMATE_COLS].values).astype(np.float32)
n_sim = len(sim_climate_scaled)

pred_scaled_all = []
t = 0
while len(pred_scaled_all) < n_sim:
    x_in = window[-N_IN:, :].reshape(1, N_IN, n_features)
    p = model.predict(x_in, verbose=0)[0]
    take = min(N_OUT, n_sim - len(pred_scaled_all))
    pred_scaled_all.extend(p[:take].tolist())

    new_clim = sim_climate_scaled[t:t + N_OUT]
    if len(new_clim) == 0:
        break
    window = np.vstack([window[len(new_clim):], new_clim])
    t += N_OUT

pred_resid = y_scaler.inverse_transform(
    np.array(pred_scaled_all[:n_sim]).reshape(-1, 1)
).reshape(-1)

sim_out = sim_df[[DATE_COL, "Year", "Month"]].copy()
sim_out["residual_pred"] = pred_resid

out_path = os.path.join(OUTPUT_DIR, "residual_prediction_2001_2020.xlsx")
sim_out.to_excel(out_path, index=False)

print("\nDone.")
print(f"Model:                {model_path}")
print(f"Predicted residuals:  {out_path}")
print("Next step (outside this script): corrected runoff  Q_sim = Q_wh + residual_pred")
