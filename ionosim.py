import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Configuração inicial da página
st.set_page_config(page_title="Simulador Skywave HF", layout="wide")

st.title("Simulador de Propagação Ionosférica (Skywave)")
st.markdown("Visualização da refração de sinais de rádio de ondas curtas (HF) nas camadas da ionosfera.")

# Dividindo a tela em colunas para os controles e o gráfico
col1, col2 = st.columns([1, 3])

with col1:
    st.header("Parâmetros")
    freq = st.slider("Frequência (MHz)", 1.0, 30.0, 7.0, 0.5,
                     help="Frequências mais baixas (ex: banda de 40m) refratam mais facilmente.")
    angle = st.slider("Ângulo de Elevação (Graus)", 10, 89, 30, 1,
                      help="Ângulos altos tendem a propagação NVIS. Ângulos baixos favorecem longas distâncias (DX).")
    cond = st.selectbox("Condição da Ionosfera", ["Dia (Alta Ionização)", "Noite (Baixa Ionização)"])

# Frequências críticas (f_c) simplificadas para as camadas
if cond == "Dia (Alta Ionização)":
    fc_F = 10.0
    fc_E = 3.5
else:
    fc_F = 5.0
    fc_E = 0.0  # Camada E praticamente desaparece à noite

# Cálculo da MUF usando a Lei da Secante
angle_rad = np.radians(angle)
muf_estimada = fc_F / np.sin(angle_rad)

with col1:
    st.info(f"**MUF Estimada (Camada F):** ~{muf_estimada:.1f} MHz\n\n"
            f"*Se a frequência do rádio ultrapassar este valor, o sinal perfurará a ionosfera e se perderá no espaço.*")

# --- Motor de Física (Ray Tracing Numérico) ---
x, y = 0.0, 0.0
vx = np.cos(angle_rad)
vy = np.sin(angle_rad)
dt = 0.5  # Resolução do passo espacial

x_vals, y_vals = [x], [y]
escapou = False

# Loop de simulação de trajetória
while y >= 0 and x <= 4000:
    aceleracao_refracao = 0.0
    
    # Dentro da Camada F (250 a 350 km) - Espessura de 100km
    if 250 <= y <= 350:
        aceleracao_refracao = (fc_F / freq)**2 / (2 * 100.0)
        
    # Dentro da Camada E (100 a 120 km) - Espessura de 20km
    elif 100 <= y <= 120 and fc_E > 0:
        aceleracao_refracao = (fc_E / freq)**2 / (2 * 20.0)

    # A refração atua diminuindo a velocidade vertical (puxando o sinal para baixo)
    vy -= aceleracao_refracao * dt
    
    # Atualiza as posições
    x += vx * dt
    y += vy * dt

    x_vals.append(x)
    y_vals.append(y)

    # Verifica se o sinal passou da ionosfera e foi para o espaço
    if y > 400:
        escapou = True
        break

# --- Renderização Gráfica ---
with col2:
    fig, ax = plt.subplots(figsize=(12, 6))

    # Desenhando as camadas ionosféricas
    if fc_E > 0:
        ax.axhspan(100, 120, color='#f4d03f', alpha=0.3, label='Camada E (~110 km)')
    ax.axhspan(250, 350, color='#eb984e', alpha=0.4, label='Camada F (~300 km)')

    # Plotando a trajetória do sinal
    cor_raio = 'red' if not escapou else 'gray'
    estilo_raio = '-' if not escapou else '--'
    ax.plot(x_vals, y_vals, color=cor_raio, linestyle=estilo_raio, linewidth=2, label=f'Sinal ({freq} MHz)')

    # Marcadores
    ax.plot(0, 0, marker='^', color='black', markersize=10, label="Transmissor (Tx)")

    # Lógica de interface de sucesso vs falha
    if not escapou:
        distancia_salto = x_vals[-1]
        ax.plot(distancia_salto, 0, marker='v', color='blue', markersize=10, label=f"Receptor (Rx)")
        
        # Anotação da distância
        ax.annotate(f'Distância do Salto:\n{distancia_salto:.0f} km',
                    xy=(distancia_salto, 0), xytext=(distancia_salto, 60),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                    ha='center')
                    
        # Demarcação da Zona de Silêncio (Skip Zone)
        ax.axvspan(0, distancia_salto, ymax=0.05, color='red', alpha=0.15)
        ax.text(distancia_salto/2, 10, 'Zona de Silêncio (Skip Zone)', ha='center', color='darkred', fontsize=9)
        
        st.success(f"**Propagação bem-sucedida!** A onda refratou e retornou à Terra, resultando em um salto de aproximadamente **{distancia_salto:.0f} km**.")
    else:
        st.error("**O sinal perfurou a camada!** A frequência configurada é superior à MUF para este ângulo de elevação. O sinal não retornou à Terra.")

    # Ajustes finais do gráfico
    xlim_max = max(1500, min(x_vals[-1] + 200 if not escapou else 1500, 4000))
    ax.set_xlim(0, xlim_max)
    ax.set_ylim(0, 450)
    ax.set_xlabel("Distância no Solo (km)")
    ax.set_ylabel("Altitude (km)")
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='upper right')

    # Envia o gráfico do Matplotlib para o Streamlit
    st.pyplot(fig)
