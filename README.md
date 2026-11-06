# PFAS adsorption prediction studio

This guide explains how to open and use the supplied `app.py` on a Windows 10/11 computer with an Intel or AMD 64-bit processor. The program runs locally in a web browser using Streamlit. You do not need a Python editor or Anaconda.

## 1. Collect the files from the original computer

Copy `app.py` and `requirements.txt` from the original GUI folder. Also obtain the following five assets from the "assets" folder; the program cannot make predictions without them.

| Asset | Required format and content |
|---|---|
| Trained ML model | `.pkl` or `.joblib`, trained for the 16 features listed below |
| Target scaler | `.pkl` or `.joblib`, fitted to the single target column ln(Qe) |
| PFAS descriptor table | `.xlsx` or `.csv`, with `Name`, `WPSA-3`, `AMW`, `SHsOH`, `SsOH`, `SddssS`, and `SsF` columns |
| DDI feature scaler | `.pkl` or `.joblib`, fitted for the same 16 input features |
| Raw DDI training features | A CSV with a header row, exactly 16 numeric columns in the order below, and at least five data rows; values must be unscaled |

Use the matching model, scalers, and data from the same training workflow. Only load trusted model/scaler files: loading pickle/joblib files can execute code. Descriptor names must be unique and descriptor values numeric.

*Do not transfer the old `.venv` folder. Create a new environment on the new computer.*

## 2. Create a folder on the new computer

In File Explorer, create `PFAS_Predictor` inside your user folder, for example `C:\Users\YourName\PFAS_Predictor`. Put the files inside as follows. The asset filenames below are examples; their actual names can differ.

```text
PFAS_Predictor
    app.py
    requirements.txt
    assets
        model.pkl
        target_scaler.pkl
        PFAS_descriptors.xlsx
        ddi_scaler.pkl
        ddi_training_features.csv
```

Make sure cloud-stored files are downloaded locally before copying them. The app lets you browse to the assets, so keeping them in `assets` is for convenience.

## 3. Install Python

Use **64-bit Python 3.11** for this guide because the supplied requirements pin older scientific packages. This is a setup choice, not a claim that the original model has been tested on Python 3.11. The existing environment records Python 3.9.7; serialized models can require the original training environment if compatibility problems occur.

Get Python from the [official Windows downloads page](https://www.python.org/downloads/windows/). If using a traditional installer, choose the 64-bit Windows installer for Python 3.11, enable the Python launcher and “Add python.exe to PATH” options when offered, and finish installation. If using the Python install manager, install the 3.11 runtime.

Open a **new Windows PowerShell** window from the Start menu and check:

```powershell
py -3.11 --version
```

Continue when it reports `Python 3.11.x`. If `py` is not recognized or 3.11 is unavailable, check the Python installation and reopen PowerShell.

## 4. Open the program folder in PowerShell

For the folder suggested in Step 2, run:

```powershell
cd "$env:USERPROFILE\PFAS_Predictor"
dir
```

Confirm that `app.py` and `requirements.txt` appear. If you chose another folder, **use its full path in quotation marks** instead.

## 5. Create the environment and install packages — first time only

Run each line below in the same PowerShell window. Wait for each command to finish before continuing. Internet access is needed to download the packages.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

The last command should report no broken requirements. These commands use the environment's Python directly, so you do not need to activate the environment or change PowerShell's execution policy.

The supplied requirements include Streamlit, NumPy 1.26.4, pandas 2.2.3, CatBoost 1.2.8, joblib, Plotly, SciPy 1.13.1, scikit-learn 1.5.2, and openpyxl. Keep the supplied `requirements.txt` rather than installing arbitrary newer versions.

## 6. Start the program

Run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Leave PowerShell open while using the program. A browser tab should open. If it does not, enter **http://localhost:8501** in your browser, or use the address shown in PowerShell. If Streamlit asks for an email address, you may leave it blank and press Enter.

You should see **PFAS Adsorption Prediction Studio** and a **Model assets** sidebar. A message asking for model assets is expected at this stage. Streamlit's [official launch documentation](https://docs.streamlit.io/develop/concepts/architecture/run-your-app) explains the Python module launch command used here.

## 7. Upload the five assets

Expand the sidebar if it is hidden. Find the required files in "assets" folder. Select each file in its matching uploader:

1. **model (.pkl)** → trained model.
2. **target_scaler (.pkl)** → scaler for ln(Qe).
3. **PFAS_descriptors (.xlsx or .csv)** → descriptor table.
4. **ddi_scaler (.pkl)** → fitted DDI scaler.
5. **ddi_training_features (.csv)** → unscaled training feature CSV.

The prediction form becomes available when all five files are loaded and pass validation. You may need to upload them again after restarting the program or starting a new browser session.

### Required training feature order

The current code contains **16 features**, in this exact order:

```text
WPSA-3
AMW
SHsOH
SsOH
SddssS
SsF
PFAS concentration (mg/L)
surface BET (m2/g)
Particle diameter (mm)
Batch dose (mg/L)
Salt (mol/L)
Temperature (??C)
pH
Stirring speed (rpm)
Duration (hours)
Single (0) or multicomponent (1)
```

Do not include an exported index column, PFAS name column, or target column in the DDI CSV. The program uses its column positions, so order matters.

## 8. Make a prediction

1. Select a PFAS from the dropdown. Its six descriptors are filled from the descriptor table.
2. Enter experimental conditions in the displayed units. In particular, adsorbent dosage is **mg/L**, particle diameter is **mm**, and contact duration is **hours**.
3. Select **Single-component** or **Multicomponent**.
4. Click **Predict qe**.
5. Check the predicted qe in **mg/g** and the applicability-domain DDI result.

For the first test, use a known example from the original computer and compare the qe and DDI values. A successful package installation alone does not verify that the correct trained assets were supplied.

## 9. Optional: generate an isotherm

Submit a single prediction first; the isotherm uses the conditions from that last submitted prediction.

1. Under **Predicted adsorption isotherm**, enter the minimum and maximum initial concentrations in mg/L. The maximum must exceed the minimum.
2. Choose 3–200 points.
3. Click **Calculate and fit isotherm**.
4. Review the plot and results table; use **Download sweep results (CSV)** to save results.

Fits require at least three physically valid points within the DDI threshold. If there are fewer, the app displays a warning and cannot provide the fits.

## 10. Stop and reopen the program

To stop: return to PowerShell and press **Ctrl+C**. Closing the browser tab alone does not stop the server.

To open it another day: open PowerShell and run only these two commands:

```powershell
cd "$env:USERPROFILE\PFAS_Predictor"
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Then upload the assets as needed. You do not need to reinstall packages each time.

## Troubleshooting

| Problem | Action |
|---|---|
| `app.py` or `requirements.txt` not found | Run `dir` and confirm you are in the folder containing both files. Check that Windows has not added `.txt` to `app.py`. |
| `.venv\Scripts\python.exe` not found | Return to the correct folder and complete the environment creation in Step 5. |
| `No module named ...` | Run the Step 5 installation command using `.venv\Scripts\python.exe`. Always launch with this same executable. |
| Installation tries to compile NumPy/SciPy or says no matching distribution | Confirm `py -3.11 --version` and that Python is 64-bit. The guide targets Intel/AMD Windows, not native Windows ARM. |
| NumPy binary incompatibility, such as `numpy.dtype size changed` | Close the app, rename the existing `.venv` folder as a backup, then repeat Step 5 to create a clean environment. |
| Browser cannot connect | Keep PowerShell running and look for errors. Use the port/address printed there. |
| Port 8501 is in use | Add `--server.port 8502` to the launch command and open `http://localhost:8502`. |
| Assets cannot load or a model/scaler version warning appears | Check that the files match one another. Obtain the original model-training package versions from the model owner and recreate that environment if necessary. Do not retrain or replace scalers just to bypass an error. |
| DDI feature-count error | Supply 16 numeric columns in the exact order above, with at least five rows. Both model and DDI scaler must also match this feature set. |
| Descriptor table error | Check all seven required headers, unique PFAS names, and numeric descriptor values. |

## Important discrepancies in the supplied program

The nearby README and several on-screen messages are outdated. The actual code uses **16 features, six descriptors, five nearest neighbors, and DDI ≤ 1.8** for the within-domain calculation. Some text still says 14 features, four descriptors, or cutoffs of 0.28/0.4. These are wording discrepancies; confirm the scientifically intended cutoff and feature set with the model owner before interpreting research results.

This guide is based on inspection of the supplied `app.py`, `requirements.txt`, README, and environment metadata. Installation on the new computer and predictions with the five assets have not been tested. The program was not modified.