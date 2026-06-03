import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Configuração da página e layout
st.set_page_config(
    page_title="Simulador Skywave HF - QTC da ECRA",
    page_icon="📻",
    layout="wide"
)

# Estilização personalizada para o rodapé fixo
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

# Título Principal
st.title("Simulador de Propagação Ionosférica (Skywave)")
st.markdown("Explore o comportamento das ondas de rádio em HF através da refração nas camadas ionosféricas.")

# --- BARRA LATERAL: Explicações Técnicas e Fundamentos ---
st.sidebar.header("📚 Fundamentos Teóricos")
st.sidebar.markdown("""
**Mecânica da Onda Celeste (*Skywave*)**
As ondas de rádio na faixa de HF (3 a 30 MHz) não se propagam em linha reta além do horizonte devido à curvatura da Terra. Em vez disso, dependem da **ionosfera** para cobrir longas distâncias (DX).

**Lei da Secante e a MUF**
A Frequência Máxima Utilizável (MUF) para um determinado enlace depende da frequência crítica da camada ($f_c$) e do ângulo de elevação da antena ($\alpha$), seguindo uma aproximação baseada na Lei da Secante:
$$MUF = \\frac{f_c}{\\sin(\\alpha)}$$

* **Frequências abaixo da MUF:** Sofrem refração suficiente e retornam à Terra.
* **Frequências acima da MUF:** Superam a densidade eletrônica da camada, "furam" a ionosfera e escapam para o espaço.
* **Ângulos Baixos:** Aumentam o caminho da onda dentro da camada, facilitando a refração e aumentando a distância do salto.
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Painel de Controlo")

# --- LAYOUT PRINCIPAL: Parâmetros e Gráfico ---
col1, col2 = st.columns([1, 2.5])

with col1:
    st.markdown("### Parâmetros de Transmissão")
    freq = st.slider(
        "Frequência de Operação (MHz)", 
        1.0, 30.0, 7.0, 0.5,
        help="Ajuste a frequência do transceptor. Bandas mais baixas refratam com mais facilidade."
    )
    angle = st.slider(
        "Ângulo de Elevação da Antena (Graus)", 
        10, 89, 30, 1,
        help="Ângulos altos favorecem comunicações locais (NVIS). Ângulos baixos são usados para DX."
    )
    cond = st.selectbox(
        "Condição da Ionosfera", 
        ["Dia (Alta Ionização)", "Noite (Baixa Ionização)"]
    )

    # Definição das frequências críticas com base na física do plasma ionosférico
    if cond == "Dia (Alta Ionização)":
        fc_F = 9.5
        fc_E = 3.5
    else:
        fc_F = 4.5
        fc_E = 0.0  # A camada E recombinou-se e desapareceu quase por completo

    # Cálculo da MUF aproximada para a camada F
    angle_rad = np.radians(angle)
    muf_estimada = fc_F / np.sin(angle_rad)

    st.markdown("---")
    st.metric(label="MUF Estimada (Camada F)", value=f"{muf_estimada:.2f} MHz")
    
    if freq > muf_estimada:
        st.error("⚠️ Alerta: Frequência acima da MUF. O sinal irá escapar para o espaço.")
    else:
        st.success("✅ Sinal refratável: Condição favorável para retorno à Terra.")

# --- MOTOR DE SIMULAÇÃO (Ray Tracing Numérico) ---
x, y = 0.0, 0.0
vx = np.cos(angle_rad)
vy = np.sin(angle_rad)
dt = 0.6  # Passo da resolução espacial

x_vals, y_vals = [x], [y]
escapou = False

# Simulação do trajeto do raio
while y >= 0 and x <= 4500:
    aceleracao_v = 0.0
    
    # Interação com a Camada F (250 a 350 km)
    if 250 <= y <= 350:
        aceleracao_v = (fc_F / freq)**2 / (2 * 100.0)
        
    # Interação com a Camada E (100 a 120 km)
    elif 100 <= y <= 120 and fc_E > 0:
        aceleracao_v = (fc_E / freq)**2 / (2 * 20.0)

    # Aplicação do efeito gradiente do índice de refração (curvando para baixo)
    vy -= aceleracao_v * dt
    
    x += vx * dt
    y += vy * dt

    x_vals.append(x)
    y_vals.append(y)

    if y > 420:
        escapou = True
        break

# --- RENDERIZAÇÃO DO GRÁFICO ---
with col2:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    
    # Desenho das Camadas Ionosféricas
    if fc_E > 0:
        ax.axhspan(100, 120, color='#f4d03f', alpha=0.25, label='Camada E (100-120 km)')
    ax.axhspan(250, 350, color='#eb984e', alpha=0.35, label='Camada F (250-350 km)')
    
    # Solo (Linha da Terra)
    ax.axhline(0, color='grey', linestyle='-', alpha=0.5)

    # Plotagem da Trajetória
    cor_linha = '#e74c3c' if not escapou else '#7f8c8d'
    estilo_linha = '-' if not escapou else '--'
    ax.plot(x_vals, y_vals, color=cor_linha, linestyle=estilo_linha, linewidth=2.5, label=f'Raio de Onda ({freq} MHz)')

    # Marcador do Transmissor (Estação Base)
    ax.plot(0, 0, marker='^', color='#2c3e50', markersize=11, label="Transmissor (Tx)")

    if not escapou:
        distancia_salto = x_vals[-1]
        # Marcador do Ponto de Queda/Recepção
        ax.plot(distancia_salto, 0, marker='v', color='#2980b9', markersize=11, label=f"Receptor (Rx)")
        
        # Seta indicativa da Distância do Salto
        ax.annotate(
            f'Distância do Salto:\n{distancia_salto:.0f} km',
            xy=(distancia_salto, 0), xytext=(distancia_salto, 70),
            arrowprops=dict(facecolor='#2c3e50', shrink=0.08, width=1, headwidth=6),
            ha='center', fontsize=10, fontweight='bold'
        )
        
        # Delimitação visual da Skip Zone
        ax.axvspan(0, distancia_salto, ymax=0.04, color='#c0392b', alpha=0.1)
        ax.text(distancia_salto/2, 12, 'Zona de Silêncio (Skip Zone)', ha='center', color='#962d22', fontsize=9.5, style='italic')
    else:
        # Indicação visual de escape espacial
        ax.text(x_vals[-1]*0.8, 390, 'Escape Espacial', color='#7f8c8d', fontsize=10, rotation=angle*0.6)

    # Configurações de eixos e moldura
    limite_x = max(1600, min(x_vals[-1] + 200 if not escapou else 1600, 4500))
    ax.set_xlim(0, limite_x)
    ax.set_ylim(0, 430)
    ax.set_xlabel("Distância de Solo (km)", fontsize=11)
    ax.set_ylabel("Altitude (km)", fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#e2e2e2')
    
    # Envia o gráfico pronto para o ecrã do Streamlit
    st.pyplot(fig)

    # Resumo dinâmico de texto abaixo do gráfico
    if not escapou:
        st.info(f"ℹ️ **Análise de Enlace:** O sinal emitido a {angle}° refratou com sucesso na ionosfera e retornou à superfície. A área iluminada pelo primeiro salto encontra-se a **{distancia_salto:.0f} km** da estação transmissora.")
    else:
        st.warning("ℹ️ **Análise de Enlace:** A densidade eletrônica atual da ionosfera é insuficiente para curvar uma onda de rádio de frequência tão elevada neste ângulo. O sinal perfurou as camadas e seguiu para o espaço exterior.")

# --- RODAPÉ DE CRÉDITOS ---
st.markdown(
    '<div class="footer">Simulador de Propagação Ionosférica HF | Desenvolvido por Alisson, PR7GA | Colaboração: QTC da ECRA</div>', 
    unsafe_allow_html=True
)
