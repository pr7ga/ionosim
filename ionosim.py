import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection

# Configuração da página e layout
st.set_page_config(
    page_title="Simulador Skywave HF/VHF Físico - QTC da ECRA",
    page_icon="📻",
    layout="wide"
)

# Estilização do rodapé fixo
st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #111111;
        color: #777777;
        text-align: center;
        padding: 10px;
        font-size: 12px;
        border-top: 1px solid #333333;
        z-index: 100;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Simulador Físico de Propagação Ionosférica")
st.markdown("Modelo contínuo com atenuação na Camada D, clima espacial e ajuste de densidade de plasma.")
st.warning("🚧 **ATENÇÃO (Versão ALFA):** Esta ferramenta é bastante simplificada para facilitar o entendimento, está em fase experimental e pode apresentar inconsistências. Utilize os resultados com cautela.")

# --- BARRA LATERAL ---
st.sidebar.header("📚 Modelo Físico")
st.sidebar.markdown("""
**VHF e a Banda de 2 Metros (150 MHz)**
Ao utilizar frequências muito elevadas, o sinal possui energia cinética suficiente para ignorar o gradiente de elétrons da ionosfera. Sinais de 144 MHz atravessam a atmosfera direto para o espaço, servindo para contatos locais em linha de visada ou via satélite.

**Clima Espacial e Manchas Solares**
O ciclo solar de 11 anos dita a propagação em HF:
* **Alta Atividade:** Muitas manchas solares geram forte radiação EUV, enriquecendo a camada F2 e aumentando a MUF. Bandas como 10m abrem globalmente.
* **Mínimo Solar:** Ionosfera fraca, MUF baixa. Longas distâncias ficam restritas a bandas baixas (40m, 80m).
* **Tempestade Solar (Blackout):** Explosões de Raios-X (Flares) atingem a Terra em minutos, super-ionizando a Camada D no lado diurno e absorvendo completamente os sinais de HF.
""")
st.sidebar.markdown("---")

# --- LAYOUT PRINCIPAL: Parâmetros ---
col1, col2 = st.columns([1, 2.5])

with col1:
    st.markdown("### Parâmetros de Transmissão")
    # Limite superior ampliado para 150 MHz (cobre a banda de 2m)
    freq = st.slider("Frequência de Operação (MHz)", 1.0, 150.0, 7.0, 0.5, help="Abrange de 160m (MF) até 2m (VHF)")
    angle = st.slider("Ângulo de Elevação da Antena (Graus)", 10, 89, 15, 1)
    
    st.markdown("### Ambiente e Clima Espacial")
    cond = st.selectbox("Condição da Ionosfera", ["Dia (Alta Ionização)", "Noite (Baixa Ionização)"])
    
    atividade_solar = st.selectbox(
        "Atividade Solar (Ciclo / Eventos)",
        [
            "Normal (Modelo Atual)", 
            "Alta Atividade (Pico do Ciclo / Muitas Manchas)", 
            "Baixa Atividade (Mínimo Solar / Zero Manchas)", 
            "Tempestade Solar (Blackout de Rádio)"
        ]
    )

    h_E, w_E = 110.0, 20.0
    h_F1, w_F1 = 180.0, 30.0
    h_F2, w_F2 = 300.0, 60.0

    # Valores base da física do modelo padrão (Normal)
    if cond == "Dia (Alta Ionização)":
        fc_E_base = 3.5
        fc_F1_base = 5.5
        fc_F2_base = 9.5
        fator_absorcao_D = 15.0
    else:
        fc_E_base = 0.0
        fc_F1_base = 0.0
        fc_F2_base = 4.5
        fator_absorcao_D = 0.0

    # Aplicação dos modificadores do Clima Espacial
    if atividade_solar == "Alta Atividade (Pico do Ciclo / Muitas Manchas)":
        fc_E_base *= 1.2
        fc_F1_base *= 1.3
        fc_F2_base *= 1.6  # MUF dispara
    elif atividade_solar == "Baixa Atividade (Mínimo Solar / Zero Manchas)":
        fc_E_base *= 0.8
        fc_F1_base *= 0.8
        fc_F2_base *= 0.6  # MUF despenca
    elif atividade_solar == "Tempestade Solar (Blackout de Rádio)":
        if cond == "Dia (Alta Ionização)":
            fator_absorcao_D *= 40.0  # Atenuação cataclísmica por Flare de Raios-X
        else:
            fator_absorcao_D = 15.0   # Alguma absorção noturna residual/auroral
        fc_F2_base *= 0.85 # Leve depressão na camada F típica de tempestades geomagnéticas

    angle_rad = np.radians(angle)
    muf_base = fc_F2_base / np.sin(angle_rad)

    st.markdown("---")
    st.markdown("### Modelo de Ionização (MUF)")
    modo_muf = st.radio("Fonte dos dados de densidade da Ionosfera:", 
                        ["Calculada pelo Modelo Teórico", "Informada pelo Usuário (Dados Reais)"])

    if modo_muf == "Calculada pelo Modelo Teórico":
        fc_E, fc_F1, fc_F2 = fc_E_base, fc_F1_base, fc_F2_base
        muf_estimada = muf_base
        st.metric(label="MUF Resultante (F2)", value=f"{muf_estimada:.2f} MHz")
    else:
        muf_usuario = st.number_input("Informe a MUF atual (MHz):", min_value=1.0, max_value=200.0, value=round(muf_base, 1), step=0.5)
        muf_estimada = muf_usuario
        fator_escala = muf_usuario / max(0.1, muf_base) # Previne divisão por zero
        
        fc_E = fc_E_base * fator_escala
        fc_F1 = fc_F1_base * fator_escala
        fc_F2 = fc_F2_base * fator_escala
        
        st.info(f"O modelo de plasma foi redimensionado internamente para obedecer à sua MUF informada.")

# --- MOTOR DE SIMULAÇÃO (Integração Numérica) ---
x, y = 0.0, 0.0
vx = np.cos(angle_rad)
vy = np.sin(angle_rad)
dt = 0.5

x_vals, y_vals, alpha_vals = [x], [y], [1.0]
escapou = False
absorvido = False
atenuacao_D_dB = 0.0
LIMITE_ABSORCAO_DB = 40.0  # dB de perda na camada D para absorção total

while y >= 0 and x <= 4500:
    # 1. Absorção da Camada D
    if 60 <= y <= 90 and fator_absorcao_D > 0:
        atenuacao_D_dB += (fator_absorcao_D / (freq**2)) * dt

    alpha_atual = max(0.0, 1.0 - (atenuacao_D_dB / LIMITE_ABSORCAO_DB))
    if alpha_atual <= 0.0:
        absorvido = True
        break

    # 2. Somatório do Gradiente de Refração
    dfp2_dy = 0.0
    
    if fc_E > 0.0 and 80 <= y <= 140:
        dfp2_dy += (fc_E**2) * (-2 * (y - h_E) / (w_E**2)) * np.exp(-((y - h_E) / w_E)**2)
        
    if fc_F1 > 0.0 and 140 <= y <= 220:
        dfp2_dy += (fc_F1**2) * (-2 * (y - h_F1) / (w_F1**2)) * np.exp(-((y - h_F1) / w_F1)**2)
        
    if fc_F2 > 0.0 and y >= 220:
        dfp2_dy += (fc_F2**2) * (-2 * (y - h_F2) / (w_F2**2)) * np.exp(-((y - h_F2) / w_F2)**2)
    
    if dfp2_dy != 0.0:
        aceleracao_v = (1.0 / (2.0 * freq**2)) * dfp2_dy
        vy -= aceleracao_v * dt
    
    # 3. Atualização Cinemática
    x += vx * dt
    y += vy * dt

    x_vals.append(x)
    y_vals.append(y)
    alpha_vals.append(alpha_atual)

    if y > 450:
        escapou = True
        break

# --- RENDERIZAÇÃO DO GRÁFICO ---
with col2:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    
    # Renderização Condicional da Camada D (Aumenta opacidade durante tempestades)
    if fator_absorcao_D > 0:
        opacidade_D = 0.5 if atividade_solar == "Tempestade Solar (Blackout de Rádio)" else 0.25
        ax.axhspan(60, 90, color='#7f8c8d', alpha=opacidade_D, label='Camada D (Absorção)')
        
    if fc_E > 0:
        ax.axhspan(90, 130, color='#f1c40f', alpha=0.2, label='Camada E')
    if fc_F1 > 0:
        ax.axhspan(150, 210, color='#e67e22', alpha=0.2, label='Camada F1')
        ax.axhspan(250, 350, color='#d35400', alpha=0.25, label='Camada F2 Principal')
    else:
        ax.axhspan(250, 350, color='#d35400', alpha=0.25, label='Camada F (Noturna)')
    
    ax.axhline(0, color='#2c3e50', linestyle='-', linewidth=2)
    ax.plot(0, 0, marker='^', color='#2c3e50', markersize=10, label="Tx")

    # Linha com opacidade dinâmica
    points = np.array([x_vals, y_vals]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    cor_base_hex = '#e74c3c' if not escapou else '#95a5a6'
    cor_base_rgb = mcolors.to_rgb(cor_base_hex)
    cores_segmentos = [(*cor_base_rgb, a) for a in alpha_vals[:-1]]
    estilo = '-' if not escapou else '-.'

    lc = LineCollection(segments, colors=cores_segmentos, linewidths=2.5, linestyles=estilo)
    ax.add_collection(lc)
    ax.plot([], [], color=cor_base_hex, linestyle=estilo, linewidth=2.5, label=f'Onda ({freq} MHz)')

    if absorvido:
        ax.plot(x_vals[-1], y_vals[-1], marker='X', color='#c0392b', markersize=8)
        ax.text(x_vals[-1]*1.05, y_vals[-1], 'Sinal Absorvido', color='#c0392b', fontsize=10, fontweight='bold')
        
        if atividade_solar == "Tempestade Solar (Blackout de Rádio)":
            st.error(f"☢️ **BLACKOUT DE RÁDIO:** Uma tempestade solar super-ionizou a Camada D. O sinal foi violentamente absorvido em poucos quilômetros de altitude. Nenhuma propagação em HF possível.")
        else:
            st.error(f"⚠️ **Sinal Dissipado:** O sinal perdeu toda a energia na Camada D antes de ser refletido. Atenuação superou {LIMITE_ABSORCAO_DB:.0f} dB.")
    
    elif not escapou:
        distancia_salto = x_vals[-1]
        ax.plot(distancia_salto, 0, marker='v', color='#2980b9', markersize=10, label=f"Rx")
        ax.annotate(
            f'Distância:\n{distancia_salto:.0f} km',
            xy=(distancia_salto, 0), xytext=(distancia_salto, 70),
            arrowprops=dict(facecolor='#2c3e50', shrink=0.05, width=1, headwidth=5),
            ha='center', fontsize=9, fontweight='bold'
        )
        ax.axvspan(0, distancia_salto, ymax=0.03, color='#c0392b', alpha=0.1)
        st.success(f"✅ **Enlace Estabelecido!** Distância de salto: **{distancia_salto:.0f} km**. Perda na Camada D: **{atenuacao_D_dB:.1f} dB**.")
        
    else:
        ax.text(x_vals[-1] - 40, 415, 'Escape Espacial', color='#7f8c8d', fontsize=10, rotation=angle*0.5, ha='right', va='center')
        if freq >= 144.0:
            st.info("🛰️ **Comportamento VHF (Linha de Visada):** A 144+ MHz a onda possui energia excessiva para ser curvada pela ionosfera. O sinal seguiu reto para o espaço.")
        else:
            st.warning("ℹ️ **O sinal escapou.** Frequência acima da MUF para o ângulo e atividade solar atuais.")

    # Escala Dinâmica do Eixo X
    if not escapou and not absorvido:
        limite_x = max(150.0, x_vals[-1] * 1.15)
    else:
        limite_x = max(150.0, x_vals[-1] + 250.0)

    limite_x = min(limite_x, 4500.0)

    ax.set_xlim(0, limite_x)
    ax.set_ylim(0, 450)
    ax.set_xlabel("Distância de Solo (km)")
    ax.set_ylabel("Altitude (km)")
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
    
    st.pyplot(fig)

# --- RODAPÉ OFICIAL ---
st.markdown(
    '<div class="footer">Simulador Físico de Propagação Ionosférica | Desenvolvido por Alisson, PR7GA | QTC da ECRA</div>', 
    unsafe_allow_html=True
)
