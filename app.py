# =====================================================================
# Sistema de Predicción de Fallas - Planta de Alquilación  (versión 2)
# Ejecutar con:  streamlit run app.py
# =====================================================================

import io
import os
import re
import warnings

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Configuración general (mismos valores que en el notebook)
# ---------------------------------------------------------------------
RUTA_MODELO = "modelo_random_forest.pkl"
HOJA_EXCEL = "Mantenimientos"
VENTANA_DIAS = 30

COLS_NUM = [
    "t_reparacion", "t_llvm", "periodicidad_meses",
    "n_eventos_previos", "n_fallas_previas", "n_correctivos_previos",
    "fallas_365d", "eventos_180d", "dias_desde_evento_previo",
    "dias_desde_ult_preventivo", "dias_desde_ult_falla",
    "t_reparacion_media_previa", "ratio_preventivo_vencido",
    "sin_preventivo_previo", "sin_falla_previa", "es_primer_evento",
    "falla", "mes",
]
COLS_CAT = ["tipo_mtto", "estado_inicial", "tipo_instrumento", "unidad"]
FEATURES = COLS_NUM + COLS_CAT

MAPEO_COLUMNAS = {
    "Fecha": "fecha",
    "Tag del instrumento": "tag",
    "Tipo de mantenimiento": "tipo_mtto",
    "Falla (Sí/No)": "falla_txt",
    "Descripción de la falla": "desc_falla",
    "Tiempo de reparación (h)": "t_reparacion",
    "Tiempo LLVM (h)": "t_llvm",
    "Estado inicial del equipo": "estado_inicial",
    "Descripción de la orden": "desc_orden",
    "ID reporte": "id_reporte",
    "Orden": "orden",
}
COLUMNAS_REQUERIDAS = ["fecha", "tag", "tipo_mtto", "falla_txt",
                       "t_reparacion", "t_llvm", "estado_inicial", "desc_orden"]

NIVELES = ["Alto", "Medio", "Bajo"]
ICONO_NIVEL = {"Alto": "🔴 Alto", "Medio": "🟠 Medio", "Bajo": "🟢 Bajo"}
COLOR_NIVEL = {"Alto": "#c0392b", "Medio": "#e08e0b", "Bajo": "#2e8b57"}

NOMBRES_VARIABLES = {
    "t_reparacion": "Tiempo de reparación (h)",
    "t_llvm": "Tiempo LLVM (h)",
    "periodicidad_meses": "Periodicidad del preventivo (meses)",
    "n_eventos_previos": "Eventos previos",
    "n_fallas_previas": "Fallas previas",
    "n_correctivos_previos": "Correctivos previos",
    "fallas_365d": "Fallas en el último año",
    "eventos_180d": "Eventos en los últimos 180 días",
    "dias_desde_evento_previo": "Días desde el evento anterior",
    "dias_desde_ult_preventivo": "Días desde el último preventivo",
    "dias_desde_ult_falla": "Días desde la última falla",
    "t_reparacion_media_previa": "Tiempo medio de reparación previo",
    "ratio_preventivo_vencido": "Ratio de preventivo vencido",
    "sin_preventivo_previo": "Sin preventivo previo",
    "sin_falla_previa": "Sin falla previa",
    "es_primer_evento": "Es el primer evento",
    "falla": "El evento actual es una falla",
    "mes": "Mes del evento",
}


# =====================================================================
# Lógica de datos (idéntica al notebook, celdas 3 y 4)
# =====================================================================
def leer_archivo(contenido, nombre):
    buffer = io.BytesIO(contenido)
    if nombre.lower().endswith(".csv"):
        try:
            return pd.read_csv(buffer, sep=None, engine="python")
        except UnicodeDecodeError:
            buffer.seek(0)
            return pd.read_csv(buffer, sep=None, engine="python", encoding="latin-1")
    hojas = pd.ExcelFile(buffer).sheet_names
    hoja = HOJA_EXCEL if HOJA_EXCEL in hojas else hojas[0]
    buffer.seek(0)
    return pd.read_excel(buffer, sheet_name=hoja)


def limpiar_tag(t):
    t = str(t).upper().strip()
    t = t.replace("(044)", "").strip()
    t = re.sub(r"\s+", "-", t)
    t = re.sub(r"^O44-", "044-", t)
    t = re.sub(r"^44-", "044-", t)
    return t


def limpiar_datos(df):
    df = df.rename(columns=MAPEO_COLUMNAS)
    faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in df.columns]
    if faltantes:
        inverso = {v: k for k, v in MAPEO_COLUMNAS.items()}
        raise ValueError(
            "Al archivo le faltan estas columnas: "
            + ", ".join(f"'{inverso.get(c, c)}'" for c in faltantes)
            + ". Descarga la plantilla en la barra lateral para ver los encabezados esperados."
        )

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["fecha", "tag"])
    df["tag"] = df["tag"].apply(limpiar_tag)
    df["es_instrumento"] = df["tag"].str.match(
        r"^(\d{3}-)?[A-Z]{1,4}-\d+|^SICI-WCFLD"
    ).astype(int)

    df = df.drop(columns=["id_reporte"], errors="ignore").drop_duplicates()
    df["falla"] = (df["falla_txt"].astype(str).str.strip().str.lower()
                   .isin(["sí", "si"])).astype(int)
    df["estado_inicial"] = df["estado_inicial"].fillna("Desconocido")
    df["desc_orden"] = df["desc_orden"].fillna("").astype(str)
    if "desc_falla" not in df.columns:
        df["desc_falla"] = ""
    df["desc_falla"] = df["desc_falla"].fillna("").astype(str)

    for col in ["t_reparacion", "t_llvm"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    return df.sort_values(["tag", "fecha"]).reset_index(drop=True)


def periodicidad_meses(texto):
    m = re.search(r"PREV\s+(\d+)\s*([MA])\b", texto.upper())
    if not m:
        return np.nan
    n = int(m.group(1))
    return n if m.group(2) == "M" else n * 12


def construir_fila(g):
    g = g.reset_index(drop=True)
    filas = []
    for i in range(len(g)):
        d = g.loc[i, "fecha"]
        pasado = g[g["fecha"] < d]
        fila = {"idx": g.loc[i, "idx_original"]}
        fila["n_eventos_previos"] = len(pasado)
        fila["n_fallas_previas"] = int(pasado["falla"].sum())
        fila["n_correctivos_previos"] = int((pasado["tipo_mtto"] == "Correctivo").sum())
        fila["fallas_365d"] = int(pasado[pasado["fecha"] >= d - pd.Timedelta(days=365)]["falla"].sum())
        fila["eventos_180d"] = int((pasado["fecha"] >= d - pd.Timedelta(days=180)).sum())
        fila["dias_desde_evento_previo"] = (d - pasado["fecha"].max()).days if len(pasado) else np.nan
        prev = pasado[pasado["tipo_mtto"] == "Preventivo"]
        fila["dias_desde_ult_preventivo"] = (d - prev["fecha"].max()).days if len(prev) else np.nan
        fall = pasado[pasado["falla"] == 1]
        fila["dias_desde_ult_falla"] = (d - fall["fecha"].max()).days if len(fall) else np.nan
        fila["t_reparacion_media_previa"] = pasado["t_reparacion"].mean() if len(pasado) else np.nan
        filas.append(fila)
    return pd.DataFrame(filas)


def agregar_derivadas(datos):
    """Variables que dependen de otras; se reutiliza en el simulador."""
    per_dias = datos["periodicidad_meses"] * 30
    datos["ratio_preventivo_vencido"] = datos["dias_desde_ult_preventivo"] / per_dias
    datos["sin_preventivo_previo"] = datos["dias_desde_ult_preventivo"].isna().astype(int)
    datos["sin_falla_previa"] = datos["dias_desde_ult_falla"].isna().astype(int)
    datos["es_primer_evento"] = datos["dias_desde_evento_previo"].isna().astype(int)
    return datos


def crear_variables(df):
    df = df.copy()
    df["tipo_instrumento"] = df["tag"].str.extract(r"^(?:\d{3}-)?([A-Z]+)-")[0].fillna("OTRO")
    df["unidad"] = df["tag"].str.extract(r"^(\d{3})-")[0].fillna("SIN_UNIDAD")
    df["periodicidad_meses"] = df["desc_orden"].apply(periodicidad_meses)

    df["idx_original"] = df.index
    ventanas = pd.concat([construir_fila(g) for _, g in df.groupby("tag")], ignore_index=True)
    datos = df.merge(ventanas, left_on="idx_original", right_on="idx")
    datos = agregar_derivadas(datos)
    datos["mes"] = datos["fecha"].dt.month
    return datos[datos["es_instrumento"] == 1].reset_index(drop=True)


def nivel_riesgo(p, umbral):
    if p >= umbral:
        return "Alto"
    if p >= umbral / 2:
        return "Medio"
    return "Bajo"


def construir_ranking(datos):
    """Último registro de cada instrumento = su estado actual."""
    ultimos = (datos.sort_values("fecha").groupby("tag", as_index=False).tail(1))
    return (ultimos.sort_values("proba", ascending=False).reset_index(drop=True))


def importancia_variables(modelo):
    try:
        nombres = modelo.named_steps["pre"].get_feature_names_out()
        valores = modelo.named_steps["modelo"].feature_importances_
    except Exception:
        return None
    imp = pd.DataFrame({"variable": nombres, "importancia": valores})
    imp["variable"] = (imp["variable"].str.replace("num__", "", regex=False)
                                        .str.replace("cat__", "", regex=False))
    imp["variable"] = imp["variable"].map(lambda v: NOMBRES_VARIABLES.get(v, v))
    return imp.sort_values("importancia", ascending=False).reset_index(drop=True)


def exportar_excel(ranking_vista, datos):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        ranking_vista.to_excel(writer, sheet_name="Ranking de riesgo", index=False)
        resumen = (ranking_vista.groupby("Nivel de riesgo")
                   .size().reindex(NIVELES, fill_value=0).rename("Instrumentos").reset_index())
        resumen.to_excel(writer, sheet_name="Resumen", index=False)
        historial = datos[["fecha", "tag", "tipo_mtto", "falla", "desc_falla", "proba"]].rename(columns={
            "fecha": "Fecha", "tag": "Tag", "tipo_mtto": "Tipo de mantenimiento",
            "falla": "Falla", "desc_falla": "Descripción de la falla",
            "proba": "Probabilidad estimada en ese evento"})
        historial.to_excel(writer, sheet_name="Historial evaluado", index=False)
    return buffer.getvalue()


def plantilla_csv():
    ejemplo = pd.DataFrame([{
        "Fecha": "15/01/2025", "Tag del instrumento": "044-PT-101",
        "Tipo de mantenimiento": "Preventivo", "Falla (Sí/No)": "No",
        "Descripción de la falla": "", "Tiempo de reparación (h)": 2,
        "Tiempo LLVM (h)": 0, "Estado inicial del equipo": "Operativo",
        "Descripción de la orden": "PREV 6M CALIBRACION TRANSMISOR",
        "ID reporte": "R-0001", "Orden": "OT-0001",
    }])
    return ejemplo.to_csv(index=False).encode("utf-8-sig")


# =====================================================================
# Funciones con caché de Streamlit
# =====================================================================
@st.cache_resource
def cargar_modelo(ruta):
    return joblib.load(ruta)


@st.cache_data(show_spinner=False)
def procesar_archivo(contenido, nombre):
    df_crudo = leer_archivo(contenido, nombre)
    df_limpio = limpiar_datos(df_crudo)
    datos = crear_variables(df_limpio)
    return df_crudo, datos


# =====================================================================
# Interfaz
# =====================================================================
st.set_page_config(page_title="Predicción de Fallas - Alquilación",
                   page_icon="🛢️", layout="wide")

st.title("Sistema de Predicción de Fallas - Planta de Alquilación")
st.caption(
    f"Estima la probabilidad de que cada instrumento presente una falla en los "
    f"próximos {VENTANA_DIAS} días a partir de su último registro de mantenimiento."
)

# ---------------- Barra lateral ----------------
with st.sidebar:
    st.header("Ajustes")
    umbral = st.slider(
        "Umbral de riesgo alto", 0.05, 0.95, 0.50, 0.05,
        help="Desde esta probabilidad el instrumento se marca en riesgo alto. "
             "Riesgo medio empieza en la mitad del umbral.")
    st.divider()
    st.subheader("Formato del archivo")
    st.write("Usa los mismos encabezados del archivo de entrenamiento.")
    st.download_button("Descargar plantilla (CSV)", plantilla_csv(),
                       file_name="plantilla_mantenimientos.csv", mime="text/csv")

archivo = st.file_uploader("Archivo de mantenimientos (CSV o Excel)",
                           type=["csv", "xlsx", "xls"])

if archivo is None:
    st.session_state.pop("resultado", None)
    st.info("Sube un archivo para habilitar el cálculo. Si no recuerdas el formato, "
            "descarga la plantilla en la barra lateral.")
    st.stop()

contenido = archivo.getvalue()
firma = (archivo.name, len(contenido))

try:
    with st.spinner("Leyendo y preparando el historial..."):
        df_crudo, datos = procesar_archivo(contenido, archivo.name)
except ValueError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"No se pudo leer el archivo: {e}")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Registros leídos", f"{len(df_crudo):,}")
c2.metric("Instrumentos válidos", f"{datos['tag'].nunique():,}" if not datos.empty else "0")
if not datos.empty:
    c3.metric("Periodo del historial",
              f"{datos['fecha'].min():%d/%m/%Y} – {datos['fecha'].max():%d/%m/%Y}")
with st.expander("Ver primeras filas del archivo"):
    st.dataframe(df_crudo.head(20), width="stretch")

if datos.empty:
    st.warning("No se encontraron tags válidos de instrumentos en el archivo.")
    st.stop()

if st.button("Calcular Riesgo de Falla", type="primary"):
    if not os.path.exists(RUTA_MODELO):
        st.error(f"No se encontró '{RUTA_MODELO}'. Colócalo en la misma carpeta que app.py.")
        st.stop()
    try:
        with st.spinner("Calculando el riesgo de cada instrumento..."):
            modelo = cargar_modelo(RUTA_MODELO)
            evaluados = datos.copy()
            evaluados["proba"] = modelo.predict_proba(evaluados[FEATURES])[:, 1]
    except Exception as e:
        st.error(f"Ocurrió un error al calcular el riesgo: {e}")
        st.stop()
    # Se guarda en la sesión para que los filtros y pestañas no borren el resultado
    st.session_state["resultado"] = {"firma": firma, "datos": evaluados}

res = st.session_state.get("resultado")
if res is None or res["firma"] != firma:
    st.info("Pulsa «Calcular Riesgo de Falla» para ver los resultados.")
    st.stop()

modelo = cargar_modelo(RUTA_MODELO)
evaluados = res["datos"]
ranking = construir_ranking(evaluados)
ranking["nivel"] = ranking["proba"].apply(lambda p: nivel_riesgo(p, umbral))

st.divider()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Instrumentos evaluados", f"{len(ranking):,}")
k2.metric("🔴 Riesgo alto", int((ranking["nivel"] == "Alto").sum()))
k3.metric("🟠 Riesgo medio", int((ranking["nivel"] == "Medio").sum()))
k4.metric("Probabilidad máxima", f"{ranking['proba'].max():.1%}")

tab_rank, tab_analisis, tab_ficha, tab_sim, tab_modelo = st.tabs(
    ["Ranking", "Análisis", "Ficha del instrumento", "Simulador", "Sobre el modelo"])

# ---------------- Pestaña 1: Ranking con filtros ----------------
with tab_rank:
    f1, f2, f3, f4 = st.columns([1.2, 1.2, 1, 1.4])
    unidades = sorted(ranking["unidad"].unique())
    tipos = sorted(ranking["tipo_instrumento"].unique())
    sel_unidad = f1.multiselect("Unidad", unidades, placeholder="Todas")
    sel_tipo = f2.multiselect("Tipo de instrumento", tipos, placeholder="Todos")
    sel_nivel = f3.multiselect("Nivel de riesgo", NIVELES, placeholder="Todos")
    buscar = f4.text_input("Buscar tag", placeholder="Ej. PT-101")

    vista = ranking.copy()
    if sel_unidad:
        vista = vista[vista["unidad"].isin(sel_unidad)]
    if sel_tipo:
        vista = vista[vista["tipo_instrumento"].isin(sel_tipo)]
    if sel_nivel:
        vista = vista[vista["nivel"].isin(sel_nivel)]
    if buscar:
        vista = vista[vista["tag"].str.contains(buscar.strip().upper(), regex=False)]

    tabla = pd.DataFrame({
        "Tag": vista["tag"],
        "Nivel de riesgo": vista["nivel"].map(ICONO_NIVEL),
        "Probabilidad (%)": (vista["proba"] * 100).round(1),
        "Tipo": vista["tipo_instrumento"],
        "Unidad": vista["unidad"],
        "Último evento": vista["fecha"].dt.date,
        "Mantenimiento": vista["tipo_mtto"],
        "Fallas previas": vista["n_fallas_previas"],
        "Fallas último año": vista["fallas_365d"],
        "Días desde última falla": vista["dias_desde_ult_falla"],
        "Días desde último preventivo": vista["dias_desde_ult_preventivo"],
    })

    st.write(f"Mostrando **{len(tabla):,}** de {len(ranking):,} instrumentos, "
             "ordenados de mayor a menor probabilidad. Haz clic en un encabezado para reordenar.")
    st.dataframe(
        tabla, width="stretch", hide_index=True, height=480,
        column_config={
            "Probabilidad (%)": st.column_config.ProgressColumn(
                "Probabilidad (%)", min_value=0, max_value=100, format="%.1f%%"),
            "Último evento": st.column_config.DateColumn("Último evento", format="DD/MM/YYYY"),
            "Días desde última falla": st.column_config.NumberColumn(format="%d"),
            "Días desde último preventivo": st.column_config.NumberColumn(format="%d"),
        },
    )

    d1, d2 = st.columns(2)
    d1.download_button("Descargar vista en Excel", exportar_excel(tabla, evaluados),
                       file_name="riesgo_falla_instrumentos.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    d2.download_button("Descargar vista en CSV", tabla.to_csv(index=False).encode("utf-8-sig"),
                       file_name="riesgo_falla_instrumentos.csv", mime="text/csv")

# ---------------- Pestaña 2: Análisis gráfico ----------------
with tab_analisis:
    escala_nivel = alt.Scale(domain=NIVELES, range=[COLOR_NIVEL[n] for n in NIVELES])

    st.subheader("Distribución de probabilidades")
    base = alt.Chart(ranking.assign(pct=ranking["proba"] * 100))
    hist = base.mark_bar().encode(
        x=alt.X("pct:Q", bin=alt.Bin(step=5), title="Probabilidad de falla (%)"),
        y=alt.Y("count():Q", title="Instrumentos"),
        color=alt.Color("nivel:N", scale=escala_nivel, title="Nivel"),
        tooltip=[alt.Tooltip("count():Q", title="Instrumentos")],
    )
    linea = alt.Chart(pd.DataFrame({"x": [umbral * 100]})).mark_rule(
        strokeDash=[6, 4], color="#555").encode(x="x:Q")
    st.altair_chart((hist + linea).properties(height=300), width="stretch")
    st.caption("La línea discontinua marca el umbral de riesgo alto configurado en la barra lateral.")

    g1, g2 = st.columns(2)
    for col_ui, campo, titulo in [(g1, "tipo_instrumento", "Por tipo de instrumento"),
                                  (g2, "unidad", "Por unidad")]:
        with col_ui:
            st.subheader(titulo)
            agrupado = (ranking.groupby([campo, "nivel"]).size()
                        .rename("n").reset_index())
            orden = (ranking.groupby(campo)["proba"].mean()
                     .sort_values(ascending=False).index.tolist())
            graf = alt.Chart(agrupado).mark_bar().encode(
                y=alt.Y(f"{campo}:N", sort=orden, title=None),
                x=alt.X("n:Q", title="Instrumentos"),
                color=alt.Color("nivel:N", scale=escala_nivel, title="Nivel"),
                order=alt.Order("nivel:N", sort="ascending"),
                tooltip=[alt.Tooltip(f"{campo}:N", title="Grupo"),
                         alt.Tooltip("nivel:N", title="Nivel"),
                         alt.Tooltip("n:Q", title="Instrumentos")],
            ).properties(height=max(220, 26 * len(orden)))
            st.altair_chart(graf, width="stretch")
            st.caption("Ordenado por probabilidad media del grupo, de mayor a menor.")

    st.subheader("Historial de fallas por mes")
    por_mes = (evaluados.assign(periodo=evaluados["fecha"].dt.to_period("M").dt.to_timestamp())
               .groupby("periodo")["falla"].sum().rename("fallas").reset_index())
    st.altair_chart(
        alt.Chart(por_mes).mark_line(point=True).encode(
            x=alt.X("periodo:T", title="Mes"),
            y=alt.Y("fallas:Q", title="Fallas registradas"),
            tooltip=[alt.Tooltip("periodo:T", title="Mes", format="%b %Y"),
                     alt.Tooltip("fallas:Q", title="Fallas")],
        ).properties(height=260),
        width="stretch",
    )

# ---------------- Pestaña 3: Ficha del instrumento ----------------
def etiqueta_tag(t):
    fila = ranking.loc[ranking["tag"] == t].iloc[0]
    return f"{t}  ·  {fila['proba']:.1%}"


with tab_ficha:
    tag_sel = st.selectbox("Instrumento", ranking["tag"].tolist(), format_func=etiqueta_tag,
                           help="La lista está ordenada de mayor a menor riesgo.", key="tag_ficha")
    actual = ranking.loc[ranking["tag"] == tag_sel].iloc[0]
    hist_tag = evaluados[evaluados["tag"] == tag_sel].sort_values("fecha")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Probabilidad actual", f"{actual['proba']:.1%}", help=ICONO_NIVEL[actual["nivel"]])
    m2.metric("Eventos registrados", len(hist_tag))
    m3.metric("Fallas totales", int(hist_tag["falla"].sum()))
    ult_falla = hist_tag.loc[hist_tag["falla"] == 1, "fecha"].max()
    m4.metric("Última falla", "Sin fallas" if pd.isna(ult_falla) else f"{ult_falla:%d/%m/%Y}")

    st.markdown(f"**Nivel de riesgo:** {ICONO_NIVEL[actual['nivel']]} · "
                f"**Tipo:** {actual['tipo_instrumento']} · **Unidad:** {actual['unidad']}")

    st.subheader("Evolución del riesgo estimado")
    hist_graf = hist_tag.assign(
        pct=hist_tag["proba"] * 100,
        evento=np.where(hist_tag["falla"] == 1, "Falla", "Sin falla"))
    linea_riesgo = alt.Chart(hist_graf).mark_line(color="#7f8c8d").encode(
        x=alt.X("fecha:T", title="Fecha"),
        y=alt.Y("pct:Q", title="Probabilidad (%)", scale=alt.Scale(domain=[0, 100])))
    puntos = alt.Chart(hist_graf).mark_point(filled=True, size=90).encode(
        x="fecha:T", y="pct:Q",
        color=alt.Color("evento:N", scale=alt.Scale(domain=["Falla", "Sin falla"],
                                                    range=["#c0392b", "#2c7fb8"]), title="Evento"),
        shape=alt.Shape("tipo_mtto:N", title="Mantenimiento"),
        tooltip=[alt.Tooltip("fecha:T", title="Fecha", format="%d/%m/%Y"),
                 alt.Tooltip("tipo_mtto:N", title="Mantenimiento"),
                 alt.Tooltip("evento:N", title="Resultado"),
                 alt.Tooltip("desc_falla:N", title="Descripción"),
                 alt.Tooltip("pct:Q", title="Probabilidad (%)", format=".1f")])
    regla = alt.Chart(pd.DataFrame({"y": [umbral * 100]})).mark_rule(
        strokeDash=[6, 4], color="#555").encode(y="y:Q")
    st.altair_chart((linea_riesgo + puntos + regla).properties(height=320), width="stretch")
    st.caption("Cada punto es un mantenimiento; su altura es el riesgo que el modelo "
               "habría estimado en esa fecha. Pasa el cursor para ver el detalle.")

    st.subheader("Historial de mantenimientos")
    st.dataframe(
        hist_tag.sort_values("fecha", ascending=False)[
            ["fecha", "tipo_mtto", "falla", "desc_falla", "t_reparacion", "estado_inicial", "proba"]
        ].rename(columns={"fecha": "Fecha", "tipo_mtto": "Mantenimiento", "falla": "Falla",
                          "desc_falla": "Descripción de la falla", "t_reparacion": "Reparación (h)",
                          "estado_inicial": "Estado inicial", "proba": "Probabilidad"}),
        width="stretch", hide_index=True,
        column_config={
            "Fecha": st.column_config.DateColumn(format="DD/MM/YYYY"),
            "Falla": st.column_config.CheckboxColumn(),
            "Probabilidad": st.column_config.NumberColumn(format="percent"),
        },
    )

# ---------------- Pestaña 4: Simulador "¿qué pasaría si...?" ----------------
with tab_sim:
    st.write("Modifica el estado del instrumento para ver cómo cambiaría su riesgo. "
             "Por ejemplo: ¿cuánto baja si se le hace hoy un preventivo?")
    tag_sim = st.selectbox("Instrumento a simular", ranking["tag"].tolist(),
                           format_func=etiqueta_tag, key="tag_sim")
    base_sim = ranking.loc[ranking["tag"] == tag_sim].iloc[[0]].copy()
    b = base_sim.iloc[0]

    def valor_dias(v):
        return 0 if pd.isna(v) else int(v)

    s1, s2 = st.columns(2)
    with s1:
        tipo_opciones = sorted(evaluados["tipo_mtto"].dropna().unique().tolist())
        tipo_sim = st.selectbox("Tipo de mantenimiento del último evento", tipo_opciones,
                                index=tipo_opciones.index(b["tipo_mtto"]) if b["tipo_mtto"] in tipo_opciones else 0)
        estado_opciones = sorted(evaluados["estado_inicial"].dropna().unique().tolist())
        estado_sim = st.selectbox("Estado inicial del equipo", estado_opciones,
                                  index=estado_opciones.index(b["estado_inicial"]) if b["estado_inicial"] in estado_opciones else 0)
        falla_sim = st.toggle("El último evento fue una falla", value=bool(b["falla"]))
    with s2:
        tiene_prev = st.toggle("Tiene preventivos previos", value=not pd.isna(b["dias_desde_ult_preventivo"]))
        dias_prev = st.slider("Días desde el último preventivo", 0, 1500,
                              valor_dias(b["dias_desde_ult_preventivo"]), disabled=not tiene_prev)
        tiene_falla = st.toggle("Tiene fallas previas", value=not pd.isna(b["dias_desde_ult_falla"]))
        dias_falla = st.slider("Días desde la última falla", 0, 1500,
                               valor_dias(b["dias_desde_ult_falla"]), disabled=not tiene_falla)
        fallas_ano = st.slider("Fallas en el último año", 0, 20, int(b["fallas_365d"]))

    sim = base_sim.copy()
    sim["tipo_mtto"] = tipo_sim
    sim["estado_inicial"] = estado_sim
    sim["falla"] = int(falla_sim)
    sim["dias_desde_ult_preventivo"] = dias_prev if tiene_prev else np.nan
    sim["dias_desde_ult_falla"] = dias_falla if tiene_falla else np.nan
    sim["fallas_365d"] = fallas_ano
    sim["n_fallas_previas"] = max(int(b["n_fallas_previas"]), fallas_ano) if tiene_falla else 0
    sim = agregar_derivadas(sim)

    p_sim = float(modelo.predict_proba(sim[FEATURES])[:, 1][0])
    delta = (p_sim - b["proba"]) * 100
    r1, r2, r3 = st.columns(3)
    r1.metric("Riesgo actual", f"{b['proba']:.1%}")
    r2.metric("Riesgo simulado", f"{p_sim:.1%}", delta=f"{delta:+.1f} pts", delta_color="inverse")
    r3.metric("Nivel simulado", ICONO_NIVEL[nivel_riesgo(p_sim, umbral)])
    st.caption("El simulador es exploratorio: muestra cómo responde el modelo a los cambios, "
               "no garantiza el comportamiento real del equipo.")

# ---------------- Pestaña 5: Sobre el modelo ----------------
with tab_modelo:
    imp = importancia_variables(modelo)
    if imp is not None:
        st.subheader("Variables que más pesan en la predicción")
        top = imp.head(15)
        st.altair_chart(
            alt.Chart(top).mark_bar(color="#2c7fb8").encode(
                x=alt.X("importancia:Q", title="Importancia relativa"),
                y=alt.Y("variable:N", sort="-x", title=None),
                tooltip=[alt.Tooltip("variable:N", title="Variable"),
                         alt.Tooltip("importancia:Q", title="Importancia", format=".3f")],
            ).properties(height=26 * len(top)),
            width="stretch",
        )
    st.subheader("Cómo se calcula el riesgo")
    st.markdown(
        f"""
- El modelo es un **Random Forest** entrenado con el historial de mantenimientos.
- Para cada instrumento se toma su **registro más reciente** y se calculan variables con
  información del pasado: fallas previas, días desde el último preventivo, frecuencia de eventos, etc.
- El resultado es la probabilidad de que el instrumento **falle en los próximos {VENTANA_DIAS} días**.
- **Riesgo alto:** probabilidad ≥ {umbral:.0%}. **Riesgo medio:** entre {umbral/2:.0%} y {umbral:.0%}.
"""
    )
