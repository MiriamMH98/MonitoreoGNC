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

# 3.1. Traer todo el histórico de consumos normalizado a 30 días
sql_full = """
  SELECT placa, fecha, cantidad
  FROM erelis2_ventas_total
  WHERE placa = ANY(%s)
"""
with get_conn() as conn:
    df_full = pd.read_sql(sql_full, conn, params=(PLACAS,))

df_full['fecha']     = pd.to_datetime(df_full['fecha'])
df_full['mes']       = df_full['fecha'].dt.to_period('M').dt.to_timestamp()
df_full['dias_mes']  = df_full['fecha'].dt.daysinmonth
df_full['lit_norm']  = df_full['cantidad'] / df_full['dias_mes'] * 30

# 3.2. Agregar por mes y por placa, luego reagrupar en cliente
pm = (
    df_full
    .groupby([df_full['mes'], 'placa'])['lit_norm']
    .sum()
    .reset_index()
)
pm['cliente'] = pm['placa'].map(CLIENTE_MAP)

# Ahora sumamos litros por cliente
df_monthly = (
    pm
    .groupby(['mes', 'cliente'])['lit_norm']
    .sum()
    .reset_index()
)

# 3.3. Pivot para series y tablas (ya no hay duplicados)
df_tend = df_monthly.pivot(
    index='mes',
    columns='cliente',
    values='lit_norm'
).fillna(0)
df_tend_table = df_tend.round(0).astype(int)

# 3.4. Comparativa Contenedor vs EDS Grupo CISA
df_comparativa = df_tend[
    ['Contenedor de GNC NATGAS', 'EDS Grupo CISA']
].round(0).astype(int)


# 3.4. Comparativa Contenedor vs EDS Grupo CISA
df_comparativa = df_tend[['Contenedor de GNC NATGAS', 'EDS Grupo CISA']].round(0).astype(int)

# ——————————————————————
# Mostrar en Streamlit
# ——————————————————————
st.subheader("Tendencia Mensual Normalizada por Cliente")
st.line_chart(df_tend_table)

st.subheader("Tabla de Tendencias (Litros)")
st.table(df_tend_table)


# 4) Comparativa Contenedor vs EDS Grupo CISA
st.subheader("Comparativa: Contenedor GNC vs EDS Grupo CISA")
st.line_chart(df_comparativa.astype(int))



# Resumen tipo correo
st.subheader("Resumen automático")
# Construir texto
lines = [f"Descrecimiento total semana-a-semana: {int(df_var[last].sum()):,} L."]
for cli, row in df_var.iterrows():
    lines.append(f"- {cli}: Última caída {int(row[last]):,} L.")
st.write("\n".join(lines))

st.info("Desarrollado por el equipo de planeación comercial.")

