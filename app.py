import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Visualizador CSV Inteligente", layout="wide")

st.title("📊 Visualizador CSV (com limpeza automática)")
st.caption("Escolha o CSV, selecione eixos e o tipo de gráfico. A app corrige vírgula decimal, datas e ordenação para evitar 'linha reta'.")

# --------- utilitários ---------
def smart_to_numeric(s: pd.Series) -> pd.Series:
    """Converte Series para numérico lidando com vírgula decimal e sujeira."""
    if pd.api.types.is_numeric_dtype(s):
        return s
    # vira string e remove espaços
    t = s.astype(str).str.strip()

    # remove qualquer char que não seja dígito, . , - ou expoente
    t = t.str.replace(r"[^0-9,\.\-eE]", "", regex=True)

    # heurística: se há vírgula e (não há ponto) OU há mais vírgulas que pontos -> vírgula é decimal
    comma_freq = t.str.count(",").fillna(0).mean()
    dot_freq = t.str.count(r"\.").fillna(0).mean()
    use_comma_decimal = (comma_freq > 0) and (comma_freq >= dot_freq)

    if use_comma_decimal:
        # remove separador de milhar (.) e troca vírgula por ponto
        t = t.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)

    # tenta converter
    return pd.to_numeric(t, errors="coerce")

def smart_to_datetime(s: pd.Series) -> pd.Series:
    """Converte Series para datetime tentando formatos comuns (dia-primeiro incluso)."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return dt

def coerce_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Limpa cabeçalhos e tenta converter colunas 'obviamente' numéricas."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    # tenta converter colunas que parecem numéricas (muitos dígitos)
    for c in df.columns:
        if df[c].dtype == object:
            # amostra para decidir se parece numérico
            sample = df[c].dropna().astype(str).head(50)
            looks_numeric = (sample.str.contains(r"\d", regex=True).mean() > 0.6)
            if looks_numeric:
                converted = smart_to_numeric(df[c])
                # só mantém se ganhou variância ou perdeu muitos NaN
                if converted.notna().sum() >= df[c].notna().sum() * 0.5:
                    df[c] = converted
    return df

# --------- UI: upload ---------
uploaded = st.file_uploader("Selecione um arquivo CSV", type=["csv"])
sep = st.selectbox("Separador (quando em dúvida, deixe 'auto')", ["auto", ",", ";", "|", "\t"], index=0)
encoding = st.selectbox("Encoding", ["auto", "utf-8", "latin1", "utf-8-sig"], index=0)

if uploaded:
    # leitura resiliente
    read_kwargs = {}
    if sep != "auto":
        read_kwargs["sep"] = sep
    else:
        read_kwargs["sep"] = None
        read_kwargs["engine"] = "python"
    if encoding != "auto":
        read_kwargs["encoding"] = encoding

    try:
        df_raw = pd.read_csv(uploaded, **read_kwargs)
    except Exception as e:
        st.error(f"Erro ao ler o CSV: {e}")
        st.stop()

    df = coerce_dataframe(df_raw)

    st.subheader("Prévia dos dados")
    st.dataframe(df.head(20), use_container_width=True)
    with st.expander("Tipos de dados (diagnóstico)"):
        st.write(df.dtypes)

    # --------- seleção de colunas ---------
    cols = df.columns.tolist()
    grafico = st.radio(
        "Tipo de gráfico",
        ["Linha", "Coluna (barras)", "Dispersão", "Pizza", "Box plot"],
        horizontal=True
    )

    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("Eixo X / Categoria", cols)
        tratar_x_como_tempo = st.checkbox("Tratar X como tempo (datetime)", value=False)
    with col2:
        # múltiplas Y para linha/coluna/dispersão
        if grafico in ["Linha", "Coluna (barras)", "Dispersão"]:
            y_cols = st.multiselect("Eixo(s) Y", [c for c in cols if c != x_col], max_selections=5)
        elif grafico == "Pizza":
            y_cols = [st.selectbox("Valores (Y)", [c for c in cols if c != x_col])]
        else:  # Box plot
            y_cols = st.multiselect("Variáveis para box plot (Y)", [c for c in cols if c != x_col], max_selections=5)

    # --------- tratamento do eixo X ---------
    df_plot = df.copy()
    if tratar_x_como_tempo:
        df_plot[x_col] = smart_to_datetime(df_plot[x_col])
        df_plot = df_plot.dropna(subset=[x_col])
        df_plot = df_plot.sort_values(x_col)

        # filtro de período
        if not df_plot.empty:
            xmin, xmax = df_plot[x_col].min(), df_plot[x_col].max()
            st.markdown("**Período:**")
            start, end = st.slider(
                "Selecione o intervalo de datas",
                min_value=xmin.to_pydatetime(),
                max_value=xmax.to_pydatetime(),
                value=(xmin.to_pydatetime(), xmax.to_pydatetime()),
                format="YYYY-MM-DD HH:mm"
            )
            mask = (df_plot[x_col] >= pd.to_datetime(start)) & (df_plot[x_col] <= pd.to_datetime(end))
            df_plot = df_plot.loc[mask]

            # opções de reamostragem e média móvel
            col_rs1, col_rs2, col_rs3 = st.columns(3)
            with col_rs1:
                freq = st.selectbox("Reamostrar (opcional)", ["Nenhum", "15S", "30S", "1Min", "5Min", "15Min", "1H"], index=0)
            with col_rs2:
                ma_win = st.number_input("Média móvel (janela)", min_value=1, max_value=500, value=1, step=1)
            with col_rs3:
                drop_na = st.checkbox("Remover NaN após processamento", value=True)

            # reamostragem (média)
            if freq != "Nenhum":
                df_plot = df_plot.set_index(x_col)
                # só reamostrar colunas numéricas
                num_cols = df_plot.select_dtypes(include=[np.number]).columns.tolist()
                if num_cols:
                    df_plot = df_plot[num_cols].resample(freq).mean().reset_index()
                else:
                    st.warning("Não há colunas numéricas para reamostrar.")
                    df_plot = df_plot.reset_index()

            # média móvel nas colunas Y (se aplicável)
            if ma_win > 1 and grafico in ["Linha", "Coluna (barras)", "Dispersão"]:
                for yc in y_cols:
                    if yc in df_plot.columns and pd.api.types.is_numeric_dtype(df_plot[yc]):
                        df_plot[yc] = df_plot[yc].rolling(ma_win, min_periods=max(1, ma_win//3)).mean()

            if drop_na:
                subset = [x_col] + y_cols if y_cols else [x_col]
                df_plot = df_plot.dropna(subset=subset)

    # --------- avisos úteis ---------
    if grafico != "Pizza" and y_cols:
        for yc in y_cols:
            if yc in df_plot.columns:
                nun = df_plot[yc].nunique(dropna=True)
                if nun <= 1:
                    st.warning(f"'{yc}' tem pouca variação ({nun} valor único). Isso pode causar uma 'linha reta'.")

    # --------- geração do gráfico ---------
    st.subheader("Gráfico")

    if grafico in ["Linha", "Coluna (barras)", "Dispersão"]:
        if not y_cols:
            st.info("Selecione pelo menos uma coluna para Y.")
        else:
            # garante X ordenado se for numérico/categórico
            if not tratar_x_como_tempo:
                # se X numérico, tente converter
                if not pd.api.types.is_numeric_dtype(df_plot[x_col]):
                    df_plot[x_col] = smart_to_numeric(df_plot[x_col])
                df_plot = df_plot.dropna(subset=[x_col]).sort_values(x_col)

            if grafico == "Linha":
                fig = px.line(df_plot, x=x_col, y=y_cols)
            elif grafico == "Coluna (barras)":
                # barras agrupadas quando várias Y
                df_m = df_plot[[x_col] + y_cols].melt(id_vars=x_col, var_name="Variável", value_name="Valor")
                fig = px.bar(df_m, x=x_col, y="Valor", color="Variável", barmode="group")
            else:  # Dispersão
                if len(y_cols) == 1:
                    fig = px.scatter(df_plot, x=x_col, y=y_cols[0])
                else:
                    # múltiplas Y -> melt para color por variável
                    df_m = df_plot[[x_col] + y_cols].melt(id_vars=x_col, var_name="Variável", value_name="Valor")
                    fig = px.scatter(df_m, x=x_col, y="Valor", color="Variável")

            st.plotly_chart(fig, use_container_width=True)

    elif grafico == "Pizza":
        y = y_cols[0] if y_cols else None
        if not y:
            st.info("Selecione uma coluna de valores (Y) para a Pizza.")
        else:
            # agrega por categoria X
            g = df_plot.groupby(x_col, dropna=False)[y].sum().reset_index()
            fig = px.pie(g, names=x_col, values=y)
            st.plotly_chart(fig, use_container_width=True)

    else:  # Box plot
        if not y_cols:
            st.info("Selecione ao menos uma coluna para o Box plot.")
        else:
            # se x_col for categoria, usamos como hue; caso contrário, só plotamos Y
            if x_col in y_cols:
                ys = y_cols
                fig = px.box(df_plot, y=ys)
            else:
                # empilha para ter várias Y em um só box plot
                m = df_plot[y_cols].melt(var_name="Variável", value_name="Valor")
                if x_col in df_plot.columns and not pd.api.types.is_numeric_dtype(df_plot[x_col]):
                    # usar x como agrupador se for categórico
                    m[x_col] = df_plot[x_col]
                    fig = px.box(m, x=x_col, y="Valor", color="Variável")
                else:
                    fig = px.box(m, x="Variável", y="Valor")
            st.plotly_chart(fig, use_container_width=True)

    # --------- rodapé: dicas ---------
    with st.expander("Por que às vezes o gráfico vira 'linha reta'?"):
        st.markdown(
            "- **Números como texto** (ex.: `-98,5` em vez de `-98.5`). A app corrige vírgula decimal automaticamente.\n"
            "- **X não é tempo**: marque *Tratar X como tempo* para converter e ordenar.\n"
            "- **Baixa variância**: se Y tem 1 valor único, o traço fica plano (veja o aviso acima).\n"
            "- **Dados não ordenados**: a app ordena X; isso evita linhas zigue-zague artificiais."
        )
