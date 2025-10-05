Power BI / Report solutions on MacBook M1

Goal

Provide practical ways to produce a Power BI-like interactive report or an actual `.pbix` starting from the outputs in this repo, while working on an Apple Silicon (M1) MacBook.

Paths overview

1) Run Power BI Desktop on the Mac via Parallels + Windows (recommended for full .pbix)
2) Use Power BI Service (web) + Desktop on a remote Windows runner
3) Run Power BI in a Windows VM via UTM (experimental, slower)
4) Build an equivalent interactive report natively with Python (Dash/Voila/Streamlit/Plotly) — cross-platform and quick

Each path below includes steps, pros/cons, and quick commands.

1) Parallels Desktop + Windows for ARM (recommended to produce .pbix locally)

Pros: Full Power BI Desktop experience, can save `.pbix` locally. Best compatibility.
Cons: Requires a Windows ARM image and Parallels license; ARM Windows may have limitations for some Power BI features.

Steps:
- Install Parallels Desktop for Mac (supports Apple Silicon). https://www.parallels.com/
- Download Windows 11 ARM Insider Preview from Microsoft (ARM64 VHDX).
- Create a new VM in Parallels, boot the Windows ARM image.
- Install Power BI Desktop (Microsoft Store app or MSI if available for ARM). If Store version not available, use Power BI Desktop for Windows on ARM or use x64 emulation (Windows 11 supports x64 emulation) — test install.
- Transfer your repo into the Windows VM (shared folder or clone repo inside VM).
- Follow the earlier instructions to load `WPP2024_POWERBI_READY.csv`, build visuals, embed maps (hosted locally in the VM), and save `Tache_3_Nom_Prenom_pbix_JJMM.pbix`.

Notes:
- To host HTML maps inside the VM and load via HTML viewer: cd to maps folder and run `python -m http.server 8000` in the VM terminal.
- For Plotly PNG export inside VM, ensure Python environment used to generate the maps has `kaleido` installed.

2) Power BI Service + remote Windows runner (no local Desktop required)

Pros: No local Windows install; you can publish PBIX from a Windows machine and use the web for viewing.
Cons: To save `.pbix` you still need Power BI Desktop; publishing can require a Power BI Pro account for sharing.

Options:
- Use a CI runner or remote Windows machine with Power BI Desktop installed. I can build the `.pbix` there if you grant access or provide the runner.
- Or use Power BI web (app.powerbi.com) to connect to CSV and build reports — but some visuals and offline `.pbix` features are only in Desktop.

3) UTM VM (open-source) — experimental

Pros: Free, runs on Apple Silicon using QEMU/UTM.
Cons: Slower, setup more complex, may require Windows ARM image and more manual config.

4) Native macOS alternative: produce an equivalent interactive report (recommended if you don't require a `.pbix`)

Pros: Fast, reproducible, fully native. No Windows required. Easier to automate and share (host on a static server or Heroku/Render/Vercel for frontends). You can embed the Plotly HTML maps directly.
Cons: Not a `.pbix`. If challenge submission strictly requires `.pbix`, this is not a replacement.

Tools & recipes (Python Dash / Plotly / Streamlit)

- Dash (Plotly) — best for arbitrary layouts and embedding Plotly maps.
- Streamlit — quick and simple, fewer layout controls but very fast.
- Voila (Jupyter) — convert notebooks to interactive web apps.

Minimal Dash app skeleton (useful to serve the CSV + maps as a report)

1. Create a venv and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas plotly dash kaleido
```

2. Example `app.py` (place at repo root)

```python
import pandas as pd
from dash import Dash, dcc, html
import plotly.express as px

app = Dash(__name__)

df = pd.read_csv('outputs/tache2/WPP2024_DEMOGRAPHIC_ENRICHED.csv')

fig = px.choropleth(df[df['year']==2023], locations='iso3', color='population_total')

app.layout = html.Div([
    html.H1('WPP2024 — Dashboard'),
    dcc.Graph(figure=fig),
    dcc.Dropdown(id='year', options=[{'label':y,'value':y} for y in sorted(df['year'].unique())], value=2023),
    dcc.Graph(id='trend')
])

@app.callback(...)
def update(...):
    pass

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
```

3. Run:

```bash
python app.py
# Open http://127.0.0.1:8050
```

Embedding HTML maps locally (works on macOS)

From the maps folder:

```bash
cd outputs/tache3/maps
python3 -m http.server 8000
```

Then point an HTML viewer or Dash iframe to `http://127.0.0.1:8000/population_1950_2023.html`.

Extra notes & tips

- Kaleido: to write Plotly images (PNG) from Python, install `kaleido`:
  pip install -U kaleido

- If you want me to build the .pbix and you have no Windows machine, the smoothest path is to provide a small Windows runner (e.g., a cloud VM or a local machine you can grant temporary access to). I can then create the .pbix and return it.

Deliverables I can produce right now

- A ready-to-run Dash/Streamlit app that reproduces the Power BI report (hostable on Mac M1).
- Step-by-step Parallels + Windows ARM setup instructions (I already covered above).
- Automated scripts to host the HTML maps and produce PNG fallbacks.

I'll now mark the todo completed. If you want a concrete path, tell me which option you prefer (Parallels, remote runner, Dash/Streamlit) and I will implement the next steps (generate a Dash app or produce a detailed Parallels setup checklist with exact links and commands).