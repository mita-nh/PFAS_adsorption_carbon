# PFAS adsorption prediction studio

It supports single-condition qe prediction, five-neighbor DDI screening, predicted concentration sweeps, Freundlich and Langmuir fits, and CSV export.

## Clean install on Windows

Use a fresh virtual environment so packages from other Python projects cannot conflict. In PowerShell, open this folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks environment activation, run this once in that PowerShell window, then activate the environment:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

The pinned NumPy and pandas versions are a compatible binary pair. If you already installed the unpinned requirements globally or into an existing environment and see `numpy.dtype size changed`, use a new `.venv` as above rather than installing over that environment.

## Upload model assets

In the app sidebar, upload the five assets:

- CatBoost model (`.pkl` or `.joblib`)
- Target scaler fitted to ln(Qe) (`.pkl` or `.joblib`)
- PFAS descriptor table (`.xlsx` or `.csv`), with `Name`, `SHsOH`, `SsOH`, `SddssS`, and `SsF` columns
- DDI feature scaler (`.pkl` or `.joblib`)
- Raw training feature CSV with the same 14 columns, in model feature order

## Model compatibility

The app was made from the supplied script, which specifies 14 model features and a DDI cutoff of 0.4. The DDI CSV should be unscaled; the uploaded DDI scaler is applied to it and to new inputs.

The workspace did not include model/data assets, so predictions become available only after you supply those files in the app.
