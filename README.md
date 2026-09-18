# Spaceship Titanic — Interfaz local de predicción

Formulario web local (Streamlit) para probar el modelo con un pasajero hipotético.

## Cómo correrla

La forma recomendada es ejecutar Streamlit como módulo de Python. Así no depende
de que el ejecutable `streamlit` esté agregado al `PATH` del sistema.

### Windows (PowerShell)

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Si `python` no está disponible, usa el lanzador de Python:

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

### macOS/Linux

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

Se abre automáticamente en el navegador en `http://localhost:8501`.

### Opción recomendada: entorno virtual

Para aislar las dependencias del proyecto, crea un entorno virtual y ejecuta
Streamlit usando directamente el Python de ese entorno.

En Windows:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

En macOS/Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

## Qué incluye

- `app.py` — la app (formulario + predicción).
- `modelo_final.joblib` — Random Forest (`n_estimators=200`, `min_samples_leaf=5`)
  entrenado con **todo** `train.csv`. Es la variante de Mario de la sección 5,
  que superó al modelo "oficial" del equipo en Accuracy, F1 y ROC-AUC — ver la
  observación del propio equipo en la sección 6.2 del notebook.
- `artifacts.joblib` — todo lo que el pipeline de preprocesamiento necesita
  para transformar un pasajero nuevo exactamente igual que en el primer avance:
  el orden ordinal de `Deck`, las columnas dummy de train, las medianas por
  `HomePlanet` (Age y gasto), el `RobustScaler` ya ajustado, y las columnas a
  las que se les aplicó `log1p`.

## Notas

- El formulario pide los mismos campos crudos que usa el pipeline
  (`selected_features` del primer avance), no las columnas ya procesadas.
- `GroupSize`/`TravelAlone` no existen para un pasajero nuevo que no viene del
  dataset, así que el formulario simplifica preguntando directamente si viaja
  solo y, si no, el tamaño de su grupo.
- Si cambian el modelo final (por ejemplo después del tuning con GridSearch/
  RandomizedSearch de la última entrega), solo hay que regenerar
  `modelo_final.joblib` con el modelo nuevo — `artifacts.joblib` no cambia
  mientras el pipeline de preprocesamiento sea el mismo.
