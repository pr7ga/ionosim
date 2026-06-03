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
st.markdown("Modelo avançado com perfis Gaussianos de densidade de elétrons e absorção da Camada D.")

# --- BARRA LATERAL: Fundamentos Teóricos ---
st.sidebar.header("📚 Modelo Físico")
st.sidebar.markdown("""
**Densidade de Elétrons Contínua**
A refração não ocorre abruptamente. Modelamos as camadas como distribuições Gaussianas, onde o raio é continuamente curvado pela derivada espacial do quadrado da frequência de plasma ($f_p^2$).

**A Camada D e a Absorção**
Presente apenas durante o dia (60-90 km), a camada D possui alta pressão atmosférica. Os elétrons excitados pelas ondas de rádio colidem com moléculas neutras, dissipando a energia do sinal como calor. 
A atenuação obedece à relação fundamental:
$$L_{dB} \\propto \\frac{1}{f^2}$$
*Ondas de frequências mais baixas (ex: 3.5 MHz) perdem muito mais energia na camada D do que ondas de frequências mais altas (ex: 28 MHz).*
""")
st.sidebar.markdown("---")

# --- LAYOUT PRINCIPAL: Parâmetros ---
col1, col2 = st.columns([1, 2.5])

with col1:
    st.markdown("### Parâmetros de Transmissão")
    freq = st.slider("Frequência de Operação (MHz)", 1.0, 30.0, 7.0, 0.5)
    angle = st.slider("Ângulo de Elevação da Antena (Graus)", 10, 89, 30, 1)
    cond = st.selectbox("Condição da Ionosfera", ["Dia (Alta Ionização)", "Noite (Baixa Ionização)"])

    # Parâmetros das Camadas (Altura h e Espessura w em km, Frequência Crítica fc em MHz)
    h_D, w_D = 75.0, 15.0
    h_E, w_E = 110.0, 20.0
    h_F, w_F = 300.0, 70.0

    if cond == "Dia (Alta Ionização)":
        fc_E = 3.5
        fc_F = 9.5
        fator_absorcao_D = 15.0  # Fator multiplicador de perda na Camada D
    else:
        fc_E = 0.5  # Recombinação quase total
        fc_F = 4.5
        fator_absorcao_D = 0.0   # Camada D desaparece à noite

    angle_rad = np.radians(angle)
    muf_estimada = fc_F / np.sin(angle_rad)

    st.markdown("---")
    st.metric(label="MUF Estimada (Aprox.)", value=f"{muf_estimada:.2f} MHz")

# --- MOTOR DE SIMULAÇÃO (Integração Numérica Constante) ---
x, y = 0.0, 0.0
vx = np.cos(angle_rad)
vy = np.sin(angle_rad)
dt = 0.5  # Resolução espacial da integração (km)

x_vals, y_vals = [x], [y]
escapou = False
atenuacao_D_dB = 0.0
distancia_total_percorrida = 0.0

while y >= 0 and x <= 4500:
    # 1. Absorção da Camada D (60km a 90km)
    if 60 <= y <= 90 and fator_absorcao_D > 0:
        # A perda aumenta com a densidade da camada e o tempo gasto nela, caindo com f^2
        atenuacao_D_dB += (fator_absorcao_D / (freq**2)) * dt

    # 2. Refração Contínua (Derivada do perfil de densidade - Lei de Snell diferencial)
    # dfp2_dy representa o gradiente vertical do quadrado da frequência de plasma
    dfp2_dy = (
        (fc_E**2) * (-2 * (y - h_E) / (w_E**2)) * np.exp(-((y - h_E) / w_E)**2) +
        (fc_F**2) * (-2 * (y - h_F) / (w_F**2)) * np.exp(-((y - h_F) / w_F)**2)
    )
    
    # A aceleração de refração empurra a onda baseada no gradiente de plasma
    aceleracao_v = (1.0 / (2.0 * freq**2)) * dfp2_dy
    vy -= aceleracao_v * dt
    
    # 3. Atualização Cinemática
    x += vx * dt
    y += vy * dt
    distancia_total_percorrida += dt

    x_vals.append(x)
    y_vals.append(y)

    # Condição de escape (ultrapassou o pico da camada F significativamente)
    if y > 450:
        escapou = True
        break

# --- CÁLCULO DE PERDAS NO ENLACE (Link Budget Básico) ---
if not escapou:
    # Free Space Path Loss (FSPL) simplificado
    fspl_dB = 32.44 + 20 * np.log10(freq) + 20 * np.log10(distancia_total_percorrida)
    perda_total = fspl_dB + atenuacao_D_dB
else:
    perda_total = float('inf')

# --- RENDERIZAÇÃO DO GRÁFICO ---
with col2:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    
    # Desenho das Camadas Ionosféricas (Gradientes Visuais)
    if fator_absorcao_D > 0:
        ax.axhspan(60, 90, color='#7f8c8d', alpha=0.3, label='Camada D (Absorção)')
    if fc_E > 1.0:
        ax.axhspan(90, 130, color='#f4d03f', alpha=0.2, label='Camada E')
    
    # A camada F é mais larga e difusa
    ax.axhspan(220, 380, color='#eb984e', alpha=0.25, label='Camada F (Refração Principal)')
    
    ax.axhline(0, color='#2c3e50', linestyle='-', linewidth=2)

    cor_linha = '#e74c3c' if not escapou else '#95a5a6'
    estilo_linha = '-' if not escapou else '-.'
    ax.plot(x_vals, y_vals, color=cor_linha, linestyle=estilo_linha, linewidth=2, label=f'Onda ({freq} MHz)')
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
        ax.text(x_vals[-1]*0.8, 410, 'Onda Perfurou a Ionosfera', color='#7f8c8d', fontsize=10, rotation=angle*0.5)

    limite_x = max(1600, min(x_vals[-1] + 200 if not escapou else 1600, 4500))
    ax.set_xlim(0, limite_x)
    ax.set_ylim(0, 450)
    ax.set_xlabel("Distância de Solo (km)")
    ax.set_ylabel("Altitude (km)")
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', framealpha=0.9)
    
    st.pyplot(fig)

    # Painel de Resultados Técnicos
    if not escapou:
        st.success(f"**Enlace Estabelecido!** O sinal atingiu o solo a **{distancia_salto:.0f} km**.")
        st.info(f"**Balanço do Enlace (Estimativa):**\n"
                f"* Atenuação por Absorção (Camada D): **{atenuacao_D_dB:.1f} dB**\n"
                f"* Atenuação de Espaço Livre (FSPL): **{fspl_dB:.1f} dB**\n"
                f"* Perda Total de Percurso: **{perda_total:.1f} dB**")
        
        if atenuacao_D_dB > 30:
            st.warning("⚠️ **Alta Atenuação na Camada D!** Embora a trajetória retorne à Terra, o sinal sofreu absorção extrema. Na prática da estação, este sinal pode estar inaudível abaixo do ruído de fundo (QRM/QRN).")
    else:
        st.error("**Falha no Enlace:** A frequência de operação excede o limite refrativo do gradiente de plasma para este ângulo de irradiação.")

# --- RODAPÉ OFICIAL ---
st.markdown(
    '<div class="footer">Simulador Físico de Propagação Ionosférica | Desenvolvido por Alisson, PR7GA | QTC da ECRA</div>', 
    unsafe_allow_html=True
)
