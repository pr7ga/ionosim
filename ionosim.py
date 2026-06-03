import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Configuração da página e layout
st.set_page_config(
    page_title="Simulador Skywave HF Físico - QTC da ECRA",
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
st.markdown("Modelo contínuo com separação diurna das camadas F1 e F2, e correção de vácuo em HF para o período noturno.")

# --- BARRA LATERAL ---
st.sidebar.header("📚 Modelo Físico")
st.sidebar.markdown("""
**Dinâmica das Camadas F1 e F2**
Durante o dia, a intensa radiação solar ioniza fortemente a alta atmosfera, fazendo com que a camada F se divida em duas: **F1** (aprox. 180 km) e **F2** (aprox. 300 km). Sinais de HF interagem com ambas, dependendo da frequência. À noite, elas se fundem novamente em uma única camada F.

**Propagação Noturna**
Sem o Sol, as camadas D e E se recombinam rapidamente e "desaparecem". O sinal de rádio viaja em linha reta (vácuo virtual para HF) até atingir a camada F, que retém sua ionização por mais tempo.
""")
st.sidebar.markdown("---")

# --- LAYOUT PRINCIPAL: Parâmetros ---
col1, col2 = st.columns([1, 2.5])

with col1:
    st.markdown("### Parâmetros de Transmissão")
    freq = st.slider("Frequência de Operação (MHz)", 1.0, 30.0, 7.0, 0.5)
    angle = st.slider("Ângulo de Elevação da Antena (Graus)", 10, 89, 16, 1)
    cond = st.selectbox("Condição da Ionosfera", ["Dia (Alta Ionização)", "Noite (Baixa Ionização)"])

    # Parâmetros Geométricos das Camadas (Altura h e Espessura w em km)
    h_E, w_E = 110.0, 20.0
    h_F1, w_F1 = 180.0, 30.0
    h_F2, w_F2 = 300.0, 60.0

    # Lógica de Ionização Diurna vs Noturna
    if cond == "Dia (Alta Ionização)":
        fc_E = 3.5
        fc_F1 = 5.5
        fc_F2 = 9.5
        fator_absorcao_D = 15.0
    else:
        fc_E = 0.0    # Camada E desaparece
        fc_F1 = 0.0   # Camada F1 funde-se com a F2
        fc_F2 = 4.5   # Camada F remanescente
        fator_absorcao_D = 0.0

    angle_rad = np.radians(angle)
    muf_estimada = fc_F2 / np.sin(angle_rad)

    st.markdown("---")
    st.metric(label="MUF Estimada (F2)", value=f"{muf_estimada:.2f} MHz")

# --- MOTOR DE SIMULAÇÃO (Integração Numérica) ---
x, y = 0.0, 0.0
vx = np.cos(angle_rad)
vy = np.sin(angle_rad)
dt = 0.5

x_vals, y_vals = [x], [y]
escapou = False
atenuacao_D_dB = 0.0
distancia_total = 0.0

while y >= 0 and x <= 4500:
    # 1. Absorção da Camada D (60km a 90km)
    if 60 <= y <= 90 and fator_absorcao_D > 0:
        atenuacao_D_dB += (fator_absorcao_D / (freq**2)) * dt

    # 2. Somatório do Gradiente de Refração
    dfp2_dy = 0.0
    
    # Contribuição da Camada E (se existir)
    if fc_E > 0.0:
        dfp2_dy += (fc_E**2) * (-2 * (y - h_E) / (w_E**2)) * np.exp(-((y - h_E) / w_E)**2)
        
    # Contribuição da Camada F1 (se existir)
    if fc_F1 > 0.0:
        dfp2_dy += (fc_F1**2) * (-2 * (y - h_F1) / (w_F1**2)) * np.exp(-((y - h_F1) / w_F1)**2)
        
    # Contribuição da Camada F2 / F Noturna
    if fc_F2 > 0.0:
        dfp2_dy += (fc_F2**2) * (-2 * (y - h_F2) / (w_F2**2)) * np.exp(-((y - h_F2) / w_F2)**2)
    
    aceleracao_v = (1.0 / (2.0 * freq**2)) * dfp2_dy
    vy -= aceleracao_v * dt
    
    # 3. Atualização Cinemática
    x += vx * dt
    y += vy * dt
    distancia_total += dt

    x_vals.append(x)
    y_vals.append(y)

    if y > 450:
        escapou = True
        break

# --- RENDERIZAÇÃO DO GRÁFICO ---
with col2:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    
    # Desenho visual rigoroso das camadas ativas
    if fator_absorcao_D > 0:
        ax.axhspan(60, 90, color='#7f8c8d', alpha=0.25, label='Camada D')
        
    if fc_E > 0:
        ax.axhspan(90, 130, color='#f1c40f', alpha=0.2, label='Camada E')
        
    if fc_F1 > 0:
        # Dia: Divisão clara entre F1 e F2
        ax.axhspan(150, 210, color='#e67e22', alpha=0.2, label='Camada F1')
        ax.axhspan(250, 350, color='#d35400', alpha=0.25, label='Camada F2 Principal')
    else:
        # Noite: Apenas a camada F remanescente
        ax.axhspan(250, 350, color='#d35400', alpha=0.25, label='Camada F (Noturna)')
    
    ax.axhline(0, color='#2c3e50', linestyle='-', linewidth=2)

    cor_linha = '#e74c3c' if not escapou else '#95a5a6'
    estilo_linha = '-' if not escapou else '-.'
    ax.plot(x_vals, y_vals, color=cor_linha, linestyle=estilo_linha, linewidth=2.5, label=f'Onda ({freq} MHz)')
    ax.plot(0, 0, marker='^', color='#2c3e50', markersize=10, label="Tx")

    if not escapou:
        distancia_salto = x_vals[-1]
        ax.plot(distancia_salto, 0, marker='v', color='#2980b9', markersize=10, label=f"Rx")
        
        ax.annotate(
            f'Distância:\n{distancia_salto:.0f} km',
            xy=(distancia_salto, 0), xytext=(distancia_salto, 70),
            arrowprops=dict(facecolor='#2c3e50', shrink=0.05, width=1, headwidth=5),
            ha='center', fontsize=9, fontweight='bold'
        )
        ax.axvspan(0, distancia_salto, ymax=0.03, color='#c0392b', alpha=0.1)
    else:
        ax.text(x_vals[-1]*0.8, 410, 'Escape Espacial', color='#7f8c8d', fontsize=10, rotation=angle*0.5)

    limite_x = max(1600, min(x_vals[-1] + 200 if not escapou else 1600, 4500))
    ax.set_xlim(0, limite_x)
    ax.set_ylim(0, 450)
    ax.set_xlabel("Distância de Solo (km)")
    ax.set_ylabel("Altitude (km)")
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # Legend position outside the main propagation path to avoid clutter
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
    
    st.pyplot(fig)

# --- RODAPÉ OFICIAL ---
st.markdown(
    '<div class="footer">Simulador Físico de Propagação Ionosférica | Desenvolvido por Alisson, PR7GA | QTC da ECRA</div>', 
    unsafe_allow_html=True
)
