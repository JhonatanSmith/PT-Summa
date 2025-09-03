from pathlib import Path
import json, joblib, pandas as pd
import numpy as np

ROOT = Path(".").resolve()
# --- Artefactos esperados
clf_dir = ROOT/"src/artifacts/classifier"
sar_dir = ROOT/"src/artifacts/sarima"

req_files = [
    clf_dir/"model_best_pipeline.joblib",
    clf_dir/"model_threshold.txt",
    clf_dir/"eval_report.json",
    sar_dir/"model_sarima.joblib",
    sar_dir/"config_sarima.json",
    sar_dir/"backtest_metrics.json",
]
print("1) Existencia de artefactos:")
for f in req_files:
    print("   ", f, "OK" if f.exists() and f.stat().st_size>0 else "FALTA")

# --- Clasificador: cargar y probar una mini-predicción
print("\n2) Clasificador: carga + proba + umbral")
pipe = joblib.load(clf_dir/"model_best_pipeline.joblib")
thr  = float((clf_dir/"model_threshold.txt").read_text().strip())
rep  = json.loads((clf_dir/"eval_report.json").read_text())

print("   Umbral:", thr)
print("   Métricas:", {k:round(rep["metrics_test_at_threshold"][k],4) for k in ["roc_auc","balanced_accuracy","f1_positive","accuracy"]})

# construir una muestra mínima desde el raw para validar columnas
raw_clf = pd.read_csv(ROOT/"data/raw/dataset_alpha_betha.csv")
raw_clf.columns = [c.strip().lower().replace(" ","_") for c in raw_clf.columns]
target = "class"
drop_cols = [c for c in ["autoid", target] if c in raw_clf.columns]
X_sample = raw_clf.drop(columns=drop_cols).head(5)

proba_idx = list(pipe.classes_).index("Betha")
proba = pipe.predict_proba(X_sample)[:, proba_idx]
pred  = (proba >= thr).astype(int)
print("   proba[0:5] ~", np.round(proba,4).tolist(), "pred@thr ->", pred.tolist())

# --- SARIMA: carga + forecast corto
print("\n3) SARIMA: carga + forecast(3)")
sar = joblib.load(sar_dir/"model_sarima.joblib")
res = sar["model"]
fc  = res.get_forecast(steps=3)
mean = fc.predicted_mean
ci   = fc.conf_int(0.05)
print("   forecast len:", len(mean), "nulls:", int(mean.isna().sum()))
print("   conf_int cols:", ci.columns.tolist())
print("   sample:", mean.head().round(3).to_dict())

# --- Criterios mínimos de aceptación
ok = all(f.exists() and f.stat().st_size>0 for f in req_files) \
     and 0 <= proba.min() <= 1 and 0 <= proba.max() <= 1 \
     and len(mean)==3 and mean.isna().sum()==0
print("\nResultado final:", "OK ✅" if ok else "REVISAR ⚠️")
