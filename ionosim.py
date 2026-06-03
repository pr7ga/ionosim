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

st.title("Simulador da Propagação Ionosférica (em desenvolvimento!!)")
st.markdown("""
Modelo contínuo com atenuação na Camada D e ajuste de densidade de plasma por MUF dinâmica. <br>
<span style='color: #e74c3c; font-size: 0.9em; font-weight: bold;'>⚠️ VERSÃO ALFA: Pode conter inconsistências. Use com precaução.</span>
""", unsafe_allow_html=True)

# --- BARRA LATERAL ---
st.sidebar.header("📚 Modelo Físico")
st.sidebar.markdown("""
**A Camada D e o "Blackout"**
A atenuação diurna da camada D (60-90 km) é inversamente proporcional ao quadrado da frequência. Sinais de baixa frequência (ex: 160m e 80m) perdem enorme quantidade de energia. Quando a atenuação é total, o sinal desaparece antes de ser refratado.

**Camadas F1 e F2**
Durante o dia, a intensa radiação divide a camada F em F1 e F2. À noite, elas se fundem e a camada D se dissipa.

**Modo de MUF Informada**
Ao invés de usar os valores padrão de densidade da ionosfera, você pode inserir a MUF observada em dados reais (ionossondas). O simulador fará a retroengenharia do gradiente de plasma, ajustando as frequências críticas ($f_c$) de todas as camadas para que o modelo físico reflita exatamente as condições do momento.
""")
st.sidebar.markdown("---")

# --- LAYOUT PRINCIPAL: Parâmetros ---
col1, col2 = st.columns([1, 2.5])

with col1:
    st.markdown("### Parâmetros de Transmissão")
    freq = st.slider("Frequência de Operação (MHz)", 1.0, 150.0, 7.0, 0.5, help="Abrange de 160m (MF) até 2m (VHF)")
    angle = st.slider("Ângulo de Elevação da Antena (Graus)", 10, 89, 13, 1)
    cond = st.selectbox("Condição da Ionosfera", ["Dia (Alta Ionização)", "Noite (Baixa Ionização)"])

    h_E, w_E = 110.0, 20.0
    h_F1, w_F1 = 180.0, 30.0
    h_F2, w_F2 = 300.0, 60.0

    # Valores base da física do modelo
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
        # Modo Personalizado
        muf_usuario = st.number_input("Informe a MUF atual (MHz):", min_value=1.0, max_value=100.0, value=round(muf_base, 1), step=0.5)
        muf_estimada = muf_usuario
        
        # Fator de escala: O quanto a ionosfera real está mais forte/fraca que o modelo teórico
        fator_escala = muf_usuario / muf_base
        
        # Redimensionando a ionosfera
        fc_E = fc_E_base * fator_escala
        fc_F1 = fc_F1_base * fator_escala
        fc_F2 = fc_F2_base * fator_escala
        
        st.info(f"O modelo de plasma foi redimensionado. A frequência crítica do pico da Camada F2 ($f_c$) foi ajustada internamente para **{fc_F2:.2f} MHz** para obedecer à sua MUF informada.")

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
    
    # Camadas
    if fator_absorcao_D > 0:
        ax.axhspan(60, 90, color='#7f8c8d', alpha=0.25, label='Camada D (Absorção)')
    if fc_E > 0:
        ax.axhspan(90, 130, color='#f1c40f', alpha=0.2, label='Camada E')
    if fc_F1 > 0:
        ax.axhspan(150, 210, color='#e67e22', alpha=0.2, label='Camada F1')
        ax.axhspan(250, 350, color='#d35400', alpha=0.25, label='Camada F2 Principal')
    else:
        ax.axhspan(250, 350, color='#d35400', alpha=0.25, label='Camada F (Noturna)')
    
    ax.axhline(0, color='#2c3e50', linestyle='-', linewidth=2)
    ax.plot(0, 0, marker='^', color='#2c3e50', markersize=10, label="Tx")

    # Linha com opacidade dinâmica para absorção
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
        ax.text(x_vals[-1]*1.05, y_vals[-1], 'Totalmente Absorvido', color='#c0392b', fontsize=10, fontweight='bold')
        st.error(f"⚠️ **Blackout por Absorção:** O sinal foi dissipado na Camada D antes de ser refletido. Atenuação superou {LIMITE_ABSORCAO_DB:.0f} dB.")
    
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
        st.warning("ℹ️ **O sinal escapou.** Frequência acima da MUF para o ângulo atual.")

    # --- ESCALA DINÂMICA DO EIXO X ---
    if not escapou and not absorvido:
        # Dá uma margem de 15% após o receptor. Garante no mínimo 150 km para não colapsar o gráfico em NVIS.
        limite_x = max(150.0, x_vals[-1] * 1.15)
    else:
        # Se escapou ou sumiu, adiciona 250 km de margem fixa além da última posição.
        limite_x = max(150.0, x_vals[-1] + 250.0)

    # Limite rígido máximo para não processar gráficos infinitos
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
