import streamlit as st
import pandas as pd
import psycopg2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import re
import matplotlib.dates as mdates
import matplotlib


# ----------------- Configuración de conexión -----------------
@st.cache_resource
def get_conn():
    return psycopg2.connect(
        dbname="postgres",
        user="erelis_admin",
        password="WQyS2HkgE7jRSi",
        host="erelis-prod.postgres.database.azure.com",
        port=5432
    )



# ----------------- Mapeo placa→cliente -----------------
CLIENTE_MAP = {
    '51UD2U':'NEOMEXICANA DE GNC SA PI DE CV', 
    '54US8S':'COMERCIAL Y TRANSPORTE GNC',
    '57UG7X':'NEOMEXICANA DE GNC SA PI DE CV',
    '60UG7X':'NEOMEXICANA DE GNC SA PI DE CV',
    'GENDISTA':'DISTASERMEX',
    'GENENCO':'ENCO',
    'GENSIMSA':'ENERGAS DE MEXICO',
    'NEOMEXICANA':'NEOMEXICANA DE GNC SA PI DE CV',
    'E5773':'Contenedor de GNC NATGAS',
    'E6713':'Contenedor de GNC NATGAS',
    '7HU2382':'Contenedor de GNC NATGAS',
    'E5772':'Contenedor de GNC NATGAS',
    'E6712':'Contenedor de GNC NATGAS',
    'E5771':'Contenedor de GNC NATGAS',
    'GENCIMAGAS':'CIMAGAS',
    '55AX8N':'COMERCIAL Y TRANSPORTE GNC',
    '537XT1':'ENERGAS DE MEXICO',
    'GENSIMA':'CIMAGAS',
    '08UH8N':'NEOMEXICANA DE GNC SA PI DE CV',
    '80UH8N':'NEOMEXICANA DE GNC SA PI DE CV',
    '03AN2H':'GAS NATURAL URUAPAN',
    '80UH8P':'NEOMEXICANA DE GNC SA PI DE CV',
    '70UD2S':'NEOMEXICANA DE GNC SA PI DE CV',
    'GEMSIMSA':'ENERGAS DE MEXICO',
    'GSIMSA':'ENERGAS DE MEXICO',
    '97UH6P':'NEOMEXICANA DE GNC SA PI DE CV',
    '17UJ4J':'NEOMEXICANA DE GNC SA PI DE CV',
    '18AU6X':'NEOMEXICANA DE GNC SA PI DE CV',
    '18UG6X':'NEOMEXICANA DE GNC SA PI DE CV',
    '7HU7608':'Contenedor de GNC NATGAS',
    '7HU7607':'Contenedor de GNC NATGAS',
    '07UC7G':'COMERCIAL Y TRANSPORTE GNC',
    '80UH8':'NEOMEXICANA DE GNC SA PI DE CV',
    '86UR6H':'NEOMEXICANA DE GNC SA PI DE CV',
    '02UR6M':'Ganamex',
    '02UC3X':'Ganamex',
    '03UR6M':'Ganamex',
    '51UR8M':'Ganamex',
    '551XA4':'NEOMEXICANA DE GNC SA PI DE CV',
    '555XA4':'Ganamex',
    '71HU7C':'COMERCIAL Y TRANSPORTE GNC',
    '71UH7C':'COMERCIAL Y TRANSPORTE GNC',
    '0711UC7G':'COMERCIAL Y TRANSPORTE GNC',
    '07UC76':'COMERCIAL Y TRANSPORTE GNC',
    '71UH7':'COMERCIAL Y TRANSPORTE GNC',
    'GENGREEN':'Green House',
    '35UD1K':'COMERCIAL Y TRANSPORTE GNC',
    '550XA4':'COMERCIAL Y TRANSPORTE GNC',
    '551XA':'COMERCIAL Y TRANSPORTE GNC',
    '8HU6323':'Contenedor de GNC NATGAS',
    '04UR6M':'Ganamex',
    '63UW1H':'Ganamex',
    '65UW1H':'Ganamex',
    '78UP7G':'Ganamex',
    '79UP7G':'Ganamex',
    '80UP7G':'Ganamex',
    '81UP7G':'Ganamex',
    '98UC2X':'Ganamex'

}

PLACAS = list(CLIENTE_MAP.keys())


clients = [
  "COMERCIAL Y TRANSPORTE GNC",
  "ENCO",
  "NEOMEXICANA DE GNC SA PI DE CV",
  "ENERGAS DE MEXICO",
  "Ganamex",
  "Green House"
]



# ----------------- Funciones de cálculo -----------------
def consumo_variaciones_semanales(n_semanas=4):
    """
    Retorna un DataFrame con las variaciones semanales de consumo para las últimas n_semanas.
    """
    today = datetime.now().date()
    # Último domingo como endpoint
    last_sunday = today - timedelta(days=(today.weekday()+1)%7)

    # Generar rangos semanales
    ranges = []
    end = last_sunday
    for _ in range(n_semanas):
        start = end - timedelta(days=6)
        ranges.append((start, end))
        end = start - timedelta(days=1)

    sql = """
      SELECT placa, SUM(cantidad) AS litros
      FROM erelis2_ventas_total
      WHERE fecha >= %s AND fecha < %s AND placa = ANY(%s)
      GROUP BY placa
    """

    semana_series = []
    # Ciclo con conexión manual para evitar recursividad
    for start, end in ranges:
        conn = psycopg2.connect(
            dbname="postgres",
            user="erelis_admin",
            password="WQyS2HkgE7jRSi",
            host="erelis-prod.postgres.database.azure.com",
            port=5432
        )
        try:
            dfw = pd.read_sql(
                sql,
                conn,
                params=(start, end + timedelta(days=1), PLACAS)
            )
        finally:
            conn.close()

        dfw['cliente'] = dfw['placa'].map(CLIENTE_MAP)
        serie = dfw.groupby('cliente')['litros'].sum()
        semana_series.append(serie)

    # Concatenar resultados
    df_weeks = pd.concat(semana_series, axis=1).fillna(0)

    # Etiquetas de columnas basado en ranges
    labels = [
        f"Semana {i+1}: {s[0].strftime('%d %b')}–{s[1].strftime('%d %b')}"
        for i, s in enumerate(ranges)
    ]
    df_weeks.columns = labels
    df_weeks.drop(index="Contenedor de GNC NATGAS", errors='ignore', inplace=True)

    # Calcular variaciones semana a semana
    for i in range(1, len(labels)):
        prev, curr = labels[i-1], labels[i]
        df_weeks[f"Var {i} (↓)"] = df_weeks[prev] - df_weeks[curr]

    df_weeks['Total Litros'] = df_weeks[labels].sum(axis=1)
    return df_weeks



# ----------------- Interfaz Streamlit -----------------
st.title("Análisis Semanal de Consumo GNC")

st.sidebar.header("Configuración")
n_semanas = st.sidebar.slider("Número de semanas", min_value=2, max_value=6, value=4)


# 1) Variaciones semanales
df_var = consumo_variaciones_semanales(n_semanas)
df_var_int = df_var.round(0).astype(int)
st.header(f"Variaciones de las últimas {n_semanas} semanas")
st.table(df_var_int)

st.subheader("Gráfica de descensos (Top 10)")
cols = [c for c in df_var_int.columns if c.startswith('Var')]
last = cols[-1]
chart_data = df_var_int[last].abs().sort_values(ascending=False).head(10)
plt.figure(figsize=(8,4))
plt.bar(chart_data.index, chart_data.values)
plt.xticks(rotation=45, ha='right')
plt.title('Top 10 caídas absolutas')
for i, val in enumerate(chart_data.values):
    plt.text(i, val, f"{int(val):,}", ha='center', va='bottom', fontsize=8)
st.pyplot(plt)

# ——————————————————————
# 2) Tendencia mensual normalizada (jun 2024 – may 2025) y forecast julio
# ——————————————————————
sql_full = """
  SELECT placa, fecha, cantidad
  FROM erelis2_ventas_total
  WHERE placa = ANY(%s)
"""
with get_conn() as conn:
    df_full = pd.read_sql(sql_full, conn, params=(PLACAS,))
# Normalizar a 30 días
df_full['fecha']    = pd.to_datetime(df_full['fecha'])
df_full['mes']      = df_full['fecha'].dt.to_period('M')
df_full['dias_mes'] = df_full['fecha'].dt.daysinmonth
df_full['lit_norm'] = df_full['cantidad'] / df_full['dias_mes'] * 30
df_full['cliente']  = df_full['placa'].map(CLIENTE_MAP)
# Pivot mensual
panel = df_full.groupby([df_full['mes'], 'cliente'])['lit_norm'].sum().reset_index()
df_tend = panel.pivot(index='mes', columns='cliente', values='lit_norm').fillna(0)
# Filtrar periodo
# Filtrar periodo dinámico: últimos 12 meses desde el actual
mes_actual = pd.Timestamp.now().to_period('M')
mes_inicio = mes_actual - 11
df_tend = df_tend.loc[mes_inicio:mes_actual]

# Excluir y renombrar
excl = ['Contenedor de GNC NATGAS','GAS NATURAL URUAPAN']
rename_map = {'NEOMEXICANA DE GNC SA PI DE CV':'Neomexicana','ENERGAS DE MEXICO':'ENERGAS'}

df_tend = df_tend.drop(columns=excl, errors='ignore').rename(columns=rename_map)

# Forecast Holt-Winters
df_trend = df_tend.copy()
# calcular listas de clientes sobre df_tend
df_clients = list(df_trend.columns)
forecasts = {}
for cli in df_clients:
    try:
        ts = df_trend[cli]
        fit = ExponentialSmoothing(ts, trend='add', seasonal=None,
                                   initialization_method='estimated').fit()
        forecasts[cli] = fit.forecast(1).iloc[0]
    except:
        forecasts[cli] = None

# Indices
try:
    idx_months = df_trend.index.to_timestamp()
except AttributeError:
    idx_months = pd.to_datetime(df_trend.index.astype(str))
# añadir forecast
next_month = idx_months.max() + pd.offsets.MonthBegin()
df_trend.loc[next_month] = pd.Series(forecasts)
idx_dates = idx_months.append(pd.DatetimeIndex([next_month]))

# ——————————————————————
# 2.5) Graficar por umbral y distinguir forecast con color histórico
# ——————————————————————
threshold = 200000
# separar high/low según df_tend (sin forecast)
high_clients = [c for c in df_clients if (df_tend[c] > threshold).any()]
low_clients  = [c for c in df_clients if c not in high_clients]

def plot_group(clients_list, title):
    plt.figure(figsize=(12,6))
    for cli in clients_list:
        vals = df_trend[cli]
        hist = vals[:-1]
        fc_val = vals.iloc[-1]
        fc_date = idx_dates[-1]
        # plot histórico y capturar color
        line, = plt.plot(idx_dates[:-1], hist, marker='o', label=cli)
        # plot forecast con mismo color
        plt.plot([fc_date], [fc_val], marker='X', linestyle='', color=line.get_color())
        # anotaciones
        for x, y in zip(idx_dates[:-1], hist):
            plt.text(x, y, f"{int(y):,}", ha='center', va='bottom', fontsize=8)
        plt.text(fc_date, fc_val, f"{int(fc_val):,}", ha='center', va='bottom', fontsize=8)
    plt.axvline(idx_dates[-2], color='gray', linestyle='--')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    plt.title(title)
    plt.ylabel('Litros normalizados (30 días)')
    plt.legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize=8)
    plt.tight_layout()
    st.pyplot(plt)


# 2.5a) Clientes con algún mes > threshold
st.subheader(f"forecast clientes gran volumen litros (base 30): > {threshold:,} L")
plot_group(high_clients, f"Tendencia (> {threshold:,} L)")

# 2.5b) Clientes sin superar threshold nunca
st.subheader(f"forecast clientes menor volumen litros (base 30): > {threshold:,} L")
plot_group(low_clients, f"Tendencia (≤ {threshold:,} L)")

# ——————————————————————
# X) Consumo Mensual Histórico Apilado
# ——————————————————————
import numpy as np

st.subheader("Consumo Mensual Histórico Apilado")

# 1) Consulta de datos
sql_full = """
  SELECT placa, fecha, cantidad
  FROM erelis2_ventas_total
  WHERE placa = ANY(%s)
"""
# No uses 'with' sobre get_conn() cacheada para evitar el error de reentrada
conn    = get_conn()
df_full = pd.read_sql(sql_full, conn, params=(PLACAS,))

# 2) Mapear y renombrar clientes a tus alias cortos
df_full['cliente'] = df_full['placa'].map(CLIENTE_MAP)
rename_map = {
    'COMERCIAL Y TRANSPORTE GNC':     'PERC',
    'NEOMEXICANA DE GNC SA PI DE CV': 'Neomexicana',
    'ENERGAS DE MEXICO':              'ENERGAS'
}
df_full['cliente'] = df_full['cliente'].replace(rename_map)

# 3) Preparar mes
df_full['fecha'] = pd.to_datetime(df_full['fecha'])
df_full['mes']   = df_full['fecha'].dt.to_period('M')

# 4) Filtrar sólo tus 6 clientes principales (alias cortos)
clients = ["PERC","ENCO","Neomexicana","ENERGAS","Ganamex","Green House"]
df_princ = df_full[df_full['cliente'].isin(clients)]

# 5) Agrupar y pivotar (unstack)
mensual = (
    df_princ
    .groupby(['mes','cliente'])['cantidad']
    .sum()
    .unstack(fill_value=0)
)

# 6) Forzar presencia de TODOS los meses y TODOS los clientes

# Filtrar periodo dinámico: últimos 12 meses desde el actual
mes_actual = pd.Timestamp.now().to_period('M')
primer_mes = mes_actual - 11
ultimo_mes   = mensual.index.max()
todos_meses  = pd.period_range(primer_mes, ultimo_mes, freq="M")

mensual_comp = mensual.reindex(
    index   = todos_meses,
    columns = clients,
    fill_value=0
)

# 7) Graficar stacked‐bar con etiquetas
labels = mensual_comp.index.strftime("%Y-%m")
x      = np.arange(len(labels))
colors = ["#8B4513","#002B49","#6B8E23","#082B29","#99C7EE","#444343"]

fig, ax = plt.subplots(figsize=(12,6))
bottom = np.zeros(len(x))

for idx, cli in enumerate(clients):
    vals = mensual_comp[cli].values
    ax.bar(x, vals, 0.8, bottom=bottom, color=colors[idx], label=cli)
    # etiqueta dentro de cada segmento
    for xi, yi, bi in zip(x, vals, bottom):
        if yi > 0:
            ax.text(xi, bi + yi/2, f"{yi:,.0f}", ha='center', va='center', fontsize=6)
    bottom += vals

# etiqueta total por mes
for xi, tot in zip(x, bottom):
    ax.text(xi, tot + max(bottom)*0.01,
            f"{int(tot):,}", ha='center', va='bottom',
            fontsize=8, fontweight='bold')

# formato ejes y leyenda
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)
ax.set_xlabel("Mes")
ax.set_ylabel("Litros cargados")
ax.set_title(f"Consumo Mensual Histórico Apilado (Jun 2024 – {labels[-1]})")
ax.legend(loc='center left', bbox_to_anchor=(1,0.5))

plt.tight_layout()
st.pyplot(fig)


# ——————————————————————
# 3.4) Comparativa mensual: Contenedor vs EDS Grupo CISA Monterrey
# ——————————————————————

# 1) Primero obtén el id de la EDS “GRUPO CISA MONTERREY”
with get_conn() as conn:
    df_id = pd.read_sql(
        "SELECT id_eds FROM erelis2_cat_eds WHERE desc_oasis = %s",
        conn, params=("GRUPO CISA MONTERREY",)
    )
    if df_id.empty:
        st.error("No se encontró 'GRUPO CISA MONTERREY' en el catálogo de EDS")
        st.stop()
    id_cisa = int(df_id.at[0, "id_eds"])

# 2a) Ventas normalizadas de tu Contenedor
start = datetime(2024, 8, 1)
end   = datetime.now() + timedelta(days=1)

with get_conn() as conn:
    df_cont = pd.read_sql(
        """
        SELECT placa, cantidad, fecha
          FROM erelis2_ventas_total
         WHERE placa = ANY(%s)
           AND fecha >= %s AND fecha < %s
        """,
        conn, params=(PLACAS, start, end)
    )

# --- FILTRO ESPECÍFICO PARA CONTENEDOR ---
df_cont = df_cont[df_cont['placa'].isin(['E5772', '7HU2382'])]


df_cont['fecha']    = pd.to_datetime(df_cont['fecha'])
df_cont['mes']      = df_cont['fecha'].dt.to_period('M')
df_cont['dias_mes'] = df_cont['fecha'].dt.daysinmonth
df_cont['lit_norm'] = df_cont['cantidad'] / df_cont['dias_mes'] * 30
df_cont['cliente']  = df_cont['placa'].map(CLIENTE_MAP)

pivot = (
    df_cont
    .groupby(['mes','cliente'])['lit_norm']
    .sum()
    .unstack(fill_value=0)
)

serie_container = pivot["Contenedor de GNC NATGAS"]

# 2b) Ventas normalizadas de EDS Grupo CISA
with get_conn() as conn:
    df_cisa = pd.read_sql(
        """
        SELECT cantidad, fecha
          FROM erelis2_ventas_total
         WHERE erelis2_id_eds = %s
           AND fecha >= %s AND fecha < %s
        """,
        conn, params=(id_cisa, start, end)
    )

df_cisa['fecha']    = pd.to_datetime(df_cisa['fecha'])
df_cisa['mes']      = df_cisa['fecha'].dt.to_period('M')
df_cisa['dias_mes'] = df_cisa['fecha'].dt.daysinmonth
df_cisa['lit_norm'] = df_cisa['cantidad'] / df_cisa['dias_mes'] * 30

serie_cisa = (
    df_cisa
    .groupby('mes')['lit_norm']
    .sum()
    .reindex(pivot.index, fill_value=0)
)



# 3) Graficar comparativa
x = pivot.index.to_timestamp()

plt.figure(figsize=(10,6))
plt.plot(x, serie_container, marker='o', label='Contenedor de GNC NATGAS')
plt.plot(x, serie_cisa,      marker='s', label='EDS Grupo CISA Monterrey')

for xi, y in zip(x, serie_container):
    plt.text(xi, y, f"{y:,.0f}", ha='center', va='bottom', fontsize=8)
for xi, y in zip(x, serie_cisa):
    plt.text(xi, y, f"{y:,.0f}", ha='center', va='bottom', fontsize=8)

plt.title('Comparativa mensual normalizada\nContenedor vs EDS Grupo CISA Monterrey')
plt.xlabel('Mes')
plt.ylabel('Litros normalizados (30 días)')
plt.xticks(rotation=45)
plt.legend(loc='center left', bbox_to_anchor=(1,0.5))
plt.tight_layout()
st.pyplot(plt)

# ----------------- Alerta placas no registradas -----------------
# Detectar cargas con manguera 13 o 15 de placas no registradas
sql_alert = """
    SELECT placa, erelis2_id_manguera, fecha, cantidad
    FROM erelis2_ventas_total
    WHERE erelis2_id_manguera IN (13,15)
      AND fecha >= %s
"""
today = datetime.now().date()
with get_conn() as conn:
    df_alert = pd.read_sql(sql_alert, conn, params=(today,))
# Filtrar placas no en CLIENTE_MAP
df_alert_no_map = df_alert[~df_alert['placa'].isin(PLACAS)]
if not df_alert_no_map.empty:
    st.error("Se detectaron cargas con placas no registradas en CLIENTE_MAP:")
    # Mostrar lista única de cargas
    st.table(df_alert_no_map[['placa','erelis2_id_manguera','fecha','cantidad']].drop_duplicates())
    # Calcular litros mensuales por placa
    df_alert_no_map['mes'] = pd.to_datetime(df_alert_no_map['fecha']).dt.to_period('M')
    sum_monthly = (
        df_alert_no_map
        .groupby(['placa','mes'])['cantidad']
        .sum()
        .reset_index()
        .rename(columns={'cantidad':'litros_totales_mes'})
    )
    # Mostrar consumo mensual
    st.subheader("Consumo mensual de placas no registradas")
    sum_monthly['mes'] = sum_monthly['mes'].astype(str)
    st.table(sum_monthly)
else:
    st.success("No hay cargas de manguera 13/15 de placas no registradas.")

# Resumen tipo correo
st.subheader("Resumen automático")
# Construir texto
lines = [f" total semana-a-semana: {int(df_var[last].sum()):,} L."]
for cli, row in df_var.iterrows():
    lines.append(f"- {cli}: Última variación: {int(row[last]):,} L.")
st.write("\n".join(lines))

st.info("Desarrollado por el equipo de planeación comercial.")