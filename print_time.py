import matplotlib.pyplot as plt
import numpy as np

# 1. Definir los tamaños del eje Y (de arriba a abajo para mejor lectura)
tsp_sizes = ['TSP20', 'TSP50', 'TSP100', 'TSP200', 'TSP500', 'TSP1000']

# 2. Extraer los datos exactos de la tabla
data = {
    'NCO Model': np.array([0.000289, 0.000822, 0.003606, 0.012690, 0.094282, 0.500099]),
    'Concorde': np.array([0.063413, 0.107062, 0.347416, 1.421078, 14.385164, 175.187158]),
    'LKH3': np.array([0.054116, 0.063603, 0.097866, 0.268183, 1.605756, 7.181330]),
    'Google OR-Tools': np.array([0.051685, 0.051602, 0.102078, 0.503486, 7.012832, 80.056621]),
    'Ant Colony Optimization': np.array([2.694213, 26.653959, 181.564, np.nan, np.nan, np.nan]),
    'Genetic Algorithm': np.array([1.078433, 14.405861, 144.557, np.nan, np.nan, np.nan])
}

# Invertir el orden para que TSP20 quede en la parte superior del gráfico
tsp_sizes = tsp_sizes[::-1]
for key in data.keys():
    data[key] = data[key][::-1]

# 3. Configurar la figura (Barras horizontales agrupadas)
fig, ax = plt.subplots(figsize=(14, 10))

y = np.arange(len(tsp_sizes))
n_bars = len(data)
total_height = 0.8
height = total_height / n_bars

# Colores y estilos 
# (Se han puesto colores categóricos de alto contraste en lugar de la escala de grises)
colors = ['forestgreen', 'tab:blue', 'tab:orange', 'tab:red', 'tab:purple', 'tab:brown']
hatches = ['', '////', '////', '////', '////', '////']
edge_colors = ['white', 'black', 'black', 'black', 'black', 'black']

# 4. Dibujar las barras iterando sobre el diccionario
for i, (name, times) in enumerate(data.items()):
    # Calcular el offset para agrupar las barras correctamente
    offset = (i - n_bars/2) * height + height/2
    
    # Dibujar barras (matplotlib ignora los np.nan automáticamente en las formas)
    rects = ax.barh(y - offset, times, height, label=name, 
                    color=colors[i], edgecolor=edge_colors[i], hatch=hatches[i])
    
    # Añadir los textos de los valores numéricos
    for j, rect in enumerate(rects):
        val = times[j]
        if not np.isnan(val):
            # Formatear el texto dependiendo de su magnitud
            text = f"{val:.5f} s" if val < 0.1 else f"{val:.2f} s"
            
            ax.annotate(text,
                        xy=(val, rect.get_y() + rect.get_height() / 2),
                        xytext=(5, 0),  # 5 puntos de separación a la derecha
                        textcoords="offset points",
                        ha='left', va='center', fontsize=8,
                        color='darkgreen' if i == 0 else 'black',
                        fontweight='bold' if i == 0 else 'normal')

# 5. Aplicar la ESCALA LOGARÍTMICA al eje X (tiempo)
ax.set_xscale('log')

# 6. Detalles estéticos y etiquetas
ax.set_xlabel('Inference Time (Seconds, Log Scale)', fontsize=12, fontweight='bold')
# El título ha sido eliminado a petición del tutor
ax.set_yticks(y)
ax.set_yticklabels(tsp_sizes, fontsize=12, fontweight='bold')

# Cuadrícula vertical para facilitar la lectura de los saltos logarítmicos
ax.grid(axis='x', linestyle='--', alpha=0.6, which='both')

# Leyenda (colocada arriba a la derecha para no tapar las barras)
ax.legend(title='Algorithms', title_fontsize='12', loc='upper right', fontsize=10, ncol=2)

# Ajustar márgenes para que no se corten los textos largos en la escala logarítmica
plt.xlim(right=3000) # Espacio extra para que el texto de 181.56s no se salga del canvas
plt.tight_layout()

# Guardar la imagen en formato vectorial SVG
plt.savefig('comparativa_tiempos_log.svg', format='svg', bbox_inches='tight')
print("[*] Gráfica de tiempos generada y guardada como comparativa_tiempos_log.svg")