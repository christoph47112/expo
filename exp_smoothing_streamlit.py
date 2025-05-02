import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from matplotlib.figure import Figure

# Seitentitel und Konfiguration
st.set_page_config(page_title="Exponentielle Glättung 1. Ordnung", layout="wide")

# CSS für besseres Aussehen
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .metrics-container {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #e6f3ff;
        border-radius: 0.5rem;
        padding: 1rem;
        flex: 1 1 200px;
    }
    .forecast-box {
        background-color: #e6ffe6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Titel und Einführung
st.title("Exponentielle Glättung 1. Ordnung")
st.markdown("""
Diese App ermöglicht die Berechnung der exponentiellen Glättung 1. Ordnung für Zeitreihendaten.
Laden Sie eine CSV- oder Excel-Datei hoch und passen Sie die Parameter an, um Ihre Daten zu analysieren.
""")

# Funktionen für die exponentielle Glättung und Fehlermetriken
def exponential_smoothing(data, alpha):
    """
    Berechnet die exponentielle Glättung 1. Ordnung für eine Zeitreihe
    
    Parameters:
    -----------
    data : array-like
        Die Originaldaten der Zeitreihe
    alpha : float
        Der Glättungsfaktor (zwischen 0 und 1)
        
    Returns:
    --------
    smoothed : array-like
        Die geglätteten Werte
    """
    smoothed = np.zeros(len(data))
    smoothed[0] = data[0]  # Erster Wert bleibt gleich
    
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
    
    return smoothed

def calculate_error_metrics(actual, predicted):
    """
    Berechnet verschiedene Fehlermetriken
    
    Parameters:
    -----------
    actual : array-like
        Die tatsächlichen Werte
    predicted : array-like
        Die vorhergesagten Werte
        
    Returns:
    --------
    metrics : dict
        Ein Dictionary mit den berechneten Fehlermetriken
    """
    # Stellen Sie sicher, dass wir mit numpy arrays arbeiten
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Fehlerberechnung
    error = actual - predicted
    abs_error = np.abs(error)
    squared_error = error ** 2
    
    # MAE - Mean Absolute Error
    mae = np.mean(abs_error)
    
    # MSE - Mean Squared Error
    mse = np.mean(squared_error)
    
    # RMSE - Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # MAPE - Mean Absolute Percentage Error
    # Nur für Nicht-Null-Werte berechnen
    non_zero = actual != 0
    mape = np.mean(abs_error[non_zero] / np.abs(actual[non_zero])) * 100 if np.any(non_zero) else np.nan
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape
    }

def get_download_link(df, filename, text):
    """
    Erstellt einen Download-Link für ein DataFrame
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Sidebar für Datei-Upload und Parameter
with st.sidebar:
    st.header("Daten hochladen")
    uploaded_file = st.file_uploader("Zeitreihendaten hochladen", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            # Dateiformat erkennen und einlesen
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:  # Excel-Dateien
                df = pd.read_excel(uploaded_file)
            
            st.success(f"Datei erfolgreich geladen: {uploaded_file.name}")
            
            # Informationen über die geladenen Daten
            st.markdown("### Datenübersicht")
            st.write(f"Anzahl der Zeilen: {df.shape[0]}")
            st.write(f"Anzahl der Spalten: {df.shape[1]}")
            
            # Kleine Vorschau der Daten
            st.markdown("### Datenvorschau")
            st.dataframe(df.head(3))
        
        except Exception as e:
            st.error(f"Fehler beim Laden der Datei: {e}")
            df = None
    else:
        df = None
    
    # Parameter nur anzeigen, wenn Daten geladen wurden
    if df is not None:
        st.header("Parameter")
        
        # Spaltenauswahl für Werte
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        date_cols = [col for col in df.columns if any(word in col.lower() for word in ['date', 'zeit', 'jahr', 'monat', 'woche', 'period'])]
        
        value_col = st.selectbox(
            "Wertspalte auswählen",
            options=df.columns,
            index=numeric_cols[0] if numeric_cols else 0,
            help="Wählen Sie die Spalte mit den zu glättenden Werten aus."
        )
        
        # Optional: Spalte für Zeitperioden
        date_col = st.selectbox(
            "Datums-/Periodenspalte (optional)",
            options=['Keine (Index verwenden)'] + df.columns.tolist(),
            index=date_cols[0] if date_cols else 0,
            help="Wählen Sie optional eine Spalte für Zeitperioden/Datumsangaben aus."
        )
        
        if date_col == 'Keine (Index verwenden)':
            date_col = None
        
        # Glättungsparameter
        alpha = st.slider(
            "Glättungsfaktor (α)",
            min_value=0.01,
            max_value=0.99,
            value=0.2,
            step=0.01,
            help="Niedrigere Werte (nahe 0) glätten stärker, höhere Werte (nahe 1) folgen den Originaldaten enger."
        )
        
        # Optionen für das Diagramm
        st.header("Diagramm-Optionen")
        chart_height = st.slider("Diagrammhöhe", 300, 800, 500, 50)
        show_grid = st.checkbox("Gitter anzeigen", True)
        
        # Download-Optionen
        st.header("Download-Optionen")
        download_format = st.radio("Format", ["CSV", "Excel"])

# Hauptbereich - nur anzeigen, wenn Daten und Spalten ausgewählt wurden
if df is not None and value_col:
    try:
        # Daten vorbereiten
        if date_col:
            # Sortieren nach Datum/Periode, wenn ausgewählt
            df = df.sort_values(by=date_col)
        
        # Originaldaten extrahieren
        original_data = df[value_col].values
        
        # Prüfen, ob alle Werte numerisch sind
        if not np.issubdtype(original_data.dtype, np.number):
            # Versuchen, in numerische Werte zu konvertieren
            try:
                original_data = pd.to_numeric(original_data)
            except:
                st.error(f"Die Spalte '{value_col}' enthält nicht-numerische Werte, die nicht konvertiert werden können.")
                st.stop()
        
        # Exponentielle Glättung berechnen
        smoothed_data = exponential_smoothing(original_data, alpha)
        
        # Ergebnisse in DataFrame speichern
        result_df = pd.DataFrame()
        
        if date_col:
            result_df['Periode'] = df[date_col]
        else:
            result_df['Periode'] = range(1, len(original_data) + 1)
        
        result_df['Originalwert'] = original_data
        result_df['Geglätteter Wert'] = smoothed_data
        
        # Für Vorhersagefehler müssen wir die verschobenen Werte betrachten
        # (geglätteter Wert von t-1 als Vorhersage für t)
        one_step_forecast = np.zeros_like(smoothed_data)
        one_step_forecast[1:] = smoothed_data[:-1]  # Verschieben um 1
        one_step_forecast[0] = original_data[0]  # Erster Wert hat keine Vorhersage
        
        result_df['Vorhersage (t+1)'] = one_step_forecast
        
        # Fehler berechnen
        errors = calculate_error_metrics(original_data[1:], one_step_forecast[1:])
        
        # Layout mit Spalten
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Diagramm
            st.header("Visualisierung")
            
            fig, ax = plt.subplots(figsize=(10, chart_height/100))
            
            # Seaborn-Styling für besseres Aussehen
            sns.set_style("whitegrid" if show_grid else "white")
            
            # Originaldaten plotten
            ax.plot(result_df.index, original_data, 'o-', label='Originaldaten', color='#4285F4', markersize=4)
            
            # Geglättete Daten plotten
            ax.plot(result_df.index, smoothed_data, 'o-', label='Geglättete Werte', color='#34A853', markersize=4)
            
            # Optional: Vorhersagen plotten
            ax.plot(result_df.index, one_step_forecast, '--', label='Vorhersage (t+1)', color='#EA4335', alpha=0.7)
            
            # X-Achsenbeschriftungen
            if date_col:
                # Nur jeden n-ten Eintrag zeigen, um Überlappung zu vermeiden
                n = max(1, len(result_df) // 20)  # Maximal 20 Labels
                plt.xticks(result_df.index[::n], result_df['Periode'].iloc[::n], rotation=45)
            
            # Titel und Labels
            ax.set_title("Exponentielle Glättung 1. Ordnung", fontsize=14)
            ax.set_xlabel("Zeitperiode")
            ax.set_ylabel(value_col)
            
            # Legende
            ax.legend()
            
            # Grid
            ax.grid(show_grid)
            
            # Diagramm anzeigen
            plt.tight_layout()
            st.pyplot(fig)
            
            # Tabelle mit den Ergebnissen
            st.header("Ergebnistabelle")
            
            # Formatierung der geglätteten Werte auf 2 Dezimalstellen
            formatted_result_df = result_df.copy()
            formatted_result_df['Geglätteter Wert'] = formatted_result_df['Geglätteter Wert'].round(2)
            formatted_result_df['Vorhersage (t+1)'] = formatted_result_df['Vorhersage (t+1)'].round(2)
            
            # Tabelle anzeigen
            st.dataframe(formatted_result_df, height=400)
            
            # Download-Links
            st.markdown("### Ergebnisse herunterladen")
            
            if download_format == "CSV":
                st.markdown(get_download_link(formatted_result_df, 
                                             f"exponentielle_glaettung_alpha_{alpha}.csv",
                                             "Ergebnisse als CSV herunterladen"), 
                           unsafe_allow_html=True)
            else:  # Excel
                # Excel-Datei im Speicher erstellen
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    formatted_result_df.to_excel(writer, sheet_name="Ergebnisse", index=False)
                    
                    # Zweites Blatt mit Metriken
                    pd.DataFrame([errors]).to_excel(writer, sheet_name="Metriken", index=False)
                    
                    # Parameter-Blatt
                    params_df = pd.DataFrame({
                        'Parameter': ['Alpha', 'Wertspalte', 'Periodenspalte'],
                        'Wert': [alpha, value_col, date_col if date_col else 'Index']
                    })
                    params_df.to_excel(writer, sheet_name="Parameter", index=False)
                    
                output.seek(0)
                
                # Download-Button für Excel
                b64 = base64.b64encode(output.getvalue()).decode()
                href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="exponentielle_glaettung_alpha_{alpha}.xlsx">Ergebnisse als Excel herunterladen</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        with col2:
            # Fehlermetriken
            st.header("Fehlermetriken")
            
            for metric, value in errors.items():
                st.metric(
                    label=metric,
                    value=f"{value:.2f}" + ("%" if metric == "MAPE" else "")
                )
            
            # Prognose für den nächsten Zeitpunkt
            st.header("Prognose")
            
            st.markdown("""
            <div class="forecast-box">
                <p><strong>Vorhersage für den nächsten Zeitpunkt:</strong></p>
                <p style="font-size: 24px; font-weight: bold;">{:.2f}</p>
            </div>
            """.format(smoothed_data[-1]), unsafe_allow_html=True)
            
            # Erklärung der Metriken
            st.header("Erklärung")
            
            st.markdown("""
            <div class="info-box">
                <p><strong>Exponentielle Glättung 1. Ordnung</strong></p>
                <p>Formel: S<sub>t</sub> = α × Y<sub>t</sub> + (1 - α) × S<sub>t-1</sub></p>
                <p>Wobei:</p>
                <ul>
                    <li>S<sub>t</sub> = Geglätteter Wert zum Zeitpunkt t</li>
                    <li>Y<sub>t</sub> = Beobachteter Wert zum Zeitpunkt t</li>
                    <li>α = Glättungsfaktor (zwischen 0 und 1)</li>
                    <li>S<sub>t-1</sub> = Geglätteter Wert des vorherigen Zeitpunkts</li>
                </ul>
            </div>
            
            <div class="info-box">
                <p><strong>Fehlermetriken:</strong></p>
                <ul>
                    <li><strong>MAE</strong>: Mean Absolute Error - Durchschnittlicher absoluter Fehler</li>
                    <li><strong>MSE</strong>: Mean Squared Error - Durchschnittlicher quadratischer Fehler</li>
                    <li><strong>RMSE</strong>: Root Mean Squared Error - Wurzel aus dem MSE</li>
                    <li><strong>MAPE</strong>: Mean Absolute Percentage Error - Durchschnittlicher prozentualer Fehler</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Fehler bei der Berechnung: {e}")
        import traceback
        st.code(traceback.format_exc())

# Fügt ein README hinzu, wenn keine Daten geladen wurden
else:
    st.markdown("""
    ## Anleitung zur Verwendung
    
    Diese App berechnet die exponentielle Glättung 1. Ordnung für Zeitreihendaten und bietet:
    
    1. **Datenanalyse**: Laden Sie Ihre CSV- oder Excel-Datei hoch
    2. **Parameterkonfiguration**: Wählen Sie die relevanten Spalten und den Glättungsfaktor
    3. **Visualisierung**: Sehen Sie die Originaldaten und geglätteten Werte im Diagramm
    4. **Fehlermetriken**: Bewerten Sie die Qualität der Glättung anhand verschiedener Metriken
    5. **Prognose**: Erhalten Sie eine Vorhersage für den nächsten Zeitpunkt
    6. **Export**: Laden Sie die Ergebnisse als CSV oder Excel herunter
    
    ### Beispieldatenformat
    
    Ihre Daten sollten in folgendem Format vorliegen:
    
    | Datum/Periode | Wert |
    |--------------|------|
    | 2023-01      | 100  |
    | 2023-02      | 120  |
    | 2023-03      | 90   |
    | ...          | ...  |
    
    Oder:
    
    | Jahr | Woche | Menge |
    |------|-------|-------|
    | 2023 | 1     | 100   |
    | 2023 | 2     | 120   |
    | 2023 | 3     | 90    |
    | ...  | ...   | ...   |
    
    ### Über exponentielle Glättung
    
    Die exponentielle Glättung 1. Ordnung ist ein Verfahren zur Glättung von Zeitreihen und zur Vorhersage zukünftiger Werte. Sie gewichtet neuere Beobachtungen stärker als ältere, wobei der Glättungsfaktor α bestimmt, wie stark diese Gewichtung ausfällt.
    """)
    
    # Beispieldaten
    st.markdown("### Beispieldaten")
    
    # Einfache Beispieldaten generieren
    example_data = pd.DataFrame({
        'Periode': [f"2023-{i:02d}" for i in range(1, 13)],
        'Wert': [100, 120, 90, 110, 105, 130, 125, 140, 135, 145, 150, 155]
    })
    
    st.dataframe(example_data)
    
    # Beispielcode
    st.markdown("### Streamlit App auf GitHub hosten")
    
    st.code("""
# 1. Erstellen Sie ein neues GitHub-Repository

# 2. Laden Sie die Streamlit-App (app.py) hoch

# 3. Erstellen Sie eine requirements.txt-Datei mit folgenden Abhängigkeiten:
streamlit
pandas
numpy
matplotlib
seaborn
openpyxl
xlsxwriter

# 4. Stellen Sie die App auf Streamlit Cloud bereit:
# - Besuchen Sie https://streamlit.io/cloud
# - Melden Sie sich an und verbinden Sie Ihr GitHub-Konto
# - Wählen Sie Ihr Repository und die app.py-Datei aus
# - Klicken Sie auf "Deploy"
    """, language="bash")

# Füge eine Fußzeile hinzu
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    <p>Entwickelt mit Streamlit • GitHub-Integration verfügbar</p>
</div>
""", unsafe_allow_html=True)
