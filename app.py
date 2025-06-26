import streamlit as st
import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing

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

# ----------------- Funciones de cálculo -----------------
def consumo_variaciones_semanales(n_semanas=4):
    today = datetime.now().date()
    last_sunday = today - timedelta(days=(today.weekday()+1)%7)

    # Generar rangos semanales
    ranges, end = [], last_sunday
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

    PLACAS = list(CLIENTE_MAP.keys())

    # ----------- CAMBIO CLAVE -----------
    # Abrimos y cerramos la conexión EN CADA ITERACIÓN
    for start, end in ranges:
        with get_conn() as conn:
            dfw = pd.read_sql(
                sql,
                conn,
                params=(start, end + timedelta(days=1), PLACAS)
            )
        dfw['cliente'] = dfw['placa'].map(CLIENTE_MAP)
        serie = dfw.groupby('cliente')['litros'].sum()
        semana_series.append(serie)
    # ------------------------------------

    # Concatenar resultados
    df_weeks = pd.concat(semana_series, axis=1).fillna(0)

    # Etiquetas de columnas
    labels = [
        f"Semana {i+1}: {s[0].strftime('%d %b')}–{s[1].strftime('%d %b')}"
        for i, s in enumerate(ranges)
    ]
    df_weeks.columns = labels
    df_weeks.drop(index="Contenedor de GNC NATGAS", errors='ignore', inplace=True)

    # Cálculo de variaciones
    for i in range(1, len(labels)):
        prev, curr = labels[i-1], labels[i]
        df_weeks[f"Var {i} (↓)"] = df_weeks[prev] - df_weeks[curr]

    df_weeks['Total Litros'] = df_weeks[labels].sum(axis=1)
    return df_weeks


# ----------------- Mapeo placa→cliente -----------------
CLIENTE_MAP = {
    '51UD2U':'NEOMEXICANA DE GNC SA PI DE CV', '54US8S':'COMERCIAL Y TRANSPORTE GNC',
    '57UG7X':'NEOMEXICANA DE GNC SA PI DE CV','60UG7X':'NEOMEXICANA DE GNC SA PI DE CV',
    'GENDISTA':'DISTASERMEX','GENENCO':'ENCO','GENSIMSA':'ENERGAS DE MEXICO',
    'NEOMEXICANA':'NEOMEXICANA DE GNC SA PI DE CV','E5773':'Contenedor de GNC NATGAS',
    'E6713':'Contenedor de GNC NATGAS','7HU2382':'Contenedor de GNC NATGAS','E5772':'Contenedor de GNC NATGAS',
    'E6712':'Contenedor de GNC NATGAS','E5771':'Contenedor de GNC NATGAS','GENCIMAGAS':'CIMAGAS',
    '55AX8N':'COMERCIAL Y TRANSPORTE GNC','537XT1':'ENERGAS DE MEXICO','GENSIMA':'CIMAGAS',
    '08UH8N':'NEOMEXICANA DE GNC SA PI DE CV','80UH8N':'NEOMEXICANA DE GNC SA PI DE CV',
    '03AN2H':'GAS NATURAL URUAPAN','80UH8P':'NEOMEXICANA DE GNC SA PI DE CV','70UD2S':'NEOMEXICANA DE GNC SA PI DE CV',
    'GEMSIMSA':'ENERGAS DE MEXICO','GSIMSA':'ENERGAS DE MEXICO','97UH6P':'NEOMEXICANA DE GNC SA PI DE CV',
    '17UJ4J':'NEOMEXICANA DE GNC SA PI DE CV','18AU6X':'NEOMEXICANA DE GNC SA PI DE CV','18UG6X':'NEOMEXICANA DE GNC SA PI DE CV',
    '7HU7608':'Contenedor de GNC NATGAS','7HU7607':'Contenedor de GNC NATGAS','07UC7G':'COMERCIAL Y TRANSPORTE GNC',
    '80UH8':'NEOMEXICANA DE GNC SA PI DE CV','86UR6H':'NEOMEXICANA DE GNC SA PI DE CV','02UR6M':'Ganamex',
    '02UC3X':'Ganamex','03UR6M':'Ganamex','51UR8M':'Ganamex','551XA4':'NEOMEXICANA DE GNC SA PI DE CV',
    '555XA4':'Ganamex','71HU7C':'COMERCIAL Y TRANSPORTE GNC','71UH7C':'COMERCIAL Y TRANSPORTE GNC',
    '0711UC7G':'COMERCIAL Y TRANSPORTE GNC','07UC76':'COMERCIAL Y TRANSPORTE GNC','71UH7':'COMERCIAL Y TRANSPORTE GNC',
    'GENGREEN':'Green House','35UD1K':'COMERCIAL Y TRANSPORTE GNC','550XA4':'COMERCIAL Y TRANSPORTE GNC',
    '551XA':'COMERCIAL Y TRANSPORTE GNC','8HU6323':'Contenedor de GNC NATGAS','04UR6M':'Ganamex',
    '63UW1H':'Ganamex','65UW1H':'Ganamex','78UP7G':'Ganamex','79UP7G':'Ganamex',
    '80UP7G':'Ganamex','81UP7G':'Ganamex','98UC2X':'Ganamex'

}

PLACAS = list(CLIENTE_MAP.keys())

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
# Top clientes con mayor caída
cols = [c for c in df_var.columns if c.startswith('Var')]
last = cols[-2]  # última variación
chart_data = df_var_int[last].abs().sort_values(ascending=False).head(10)
plt.figure(figsize=(8,4))
plt.bar(chart_data.index, chart_data.values)
plt.xticks(rotation=45, ha='right')
plt.title('Top 10 caídas absolutas')
st.pyplot(plt)


# ——————————————————————
# 3) Tendencia mensual normalizada y comparativas
# ——————————————————————

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






# Resumen tipo correo
st.subheader("Resumen automático")
# Construir texto
lines = [f"Descrecimiento total semana-a-semana: {int(df_var[last].sum()):,} L."]
for cli, row in df_var.iterrows():
    lines.append(f"- {cli}: Última caída {int(row[last]):,} L.")
st.write("\n".join(lines))

st.info("Desarrollado por el equipo de planeación comercial.")

