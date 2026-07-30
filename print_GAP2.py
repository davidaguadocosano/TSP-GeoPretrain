import matplotlib.pyplot as plt
import numpy as np

# 1. Definir los tamaños del eje X
tsp_sizes = ['TSP20', 'TSP50', 'TSP100', 'TSP200', 'TSP500', 'TSP1000']
x = np.arange(len(tsp_sizes))  
width = 0.065 # Reducido para acomodar 12 barras

# 2. Valores de coste del solucionador exacto (Baseline Concorde actualizado según tabla)
concorde = np.array([3.812, 5.695, 7.727, 10.649, 16.539, 23.291])

# 3. Diccionarios con los costes medios y sus intervalos de confianza (+/-)
# Se incluyen los valores actualizados de la tabla
model_costs = {
    # Redes Neuronales (Pre-trained)
    'None (Base)': np.array([3.989, 6.087, 8.796, 13.321, 24.139, 38.940]),
    'Rotation': np.array([3.954, 6.039, 8.666, 13.080, 23.278, 36.835]),
    'Reflection': np.array([3.953, 6.030, 8.651, 12.927, 23.201, 36.694]),
    'Translation': np.array([4.008, 6.047, 8.729, 13.453, 24.970, 41.042]),
    'Rot + Refl': np.array([3.953, 6.030, 8.623, 12.936, 22.887, 36.123]),
    'Rot + Trans': np.array([3.973, 6.005, 8.630, 13.344, 26.782, 48.199]),
    'Trans + Ref': np.array([3.973, 6.070, 8.743, 13.183, 24.219, 38.855]),
    'Rot+Ref+Trans': np.array([3.949, 6.028, 8.626, 12.959, 23.350, 37.405]),
    
    # Algoritmos y Heurísticas
    'LKH3': np.array([3.812, 5.695, 7.727, 10.649, 16.539, 23.291]),
    'Google OR-Tools': np.array([3.822, 5.850, 8.077, 11.217, 17.274, 24.480]),
    'Ant Colony Optimization': np.array([3.868, 5.833, 8.032, np.nan, np.nan, np.nan]),
    'Genetic Algorithm': np.array([3.875, 5.740, 7.986, np.nan, np.nan, np.nan])
}

model_cis = {
    # Redes Neuronales (Pre-trained)
    'None (Base)': np.array([0.020, 0.019, 0.031, 0.064, 0.092, 0.137]),
    'Rotation': np.array([0.020, 0.018, 0.033, 0.066, 0.094, 0.134]),
    'Reflection': np.array([0.020, 0.019, 0.028, 0.064, 0.089, 0.132]),
    'Translation': np.array([0.022, 0.019, 0.031, 0.062, 0.094, 0.148]),
    'Rot + Refl': np.array([0.020, 0.018, 0.028, 0.066, 0.091, 0.139]),
    'Rot + Trans': np.array([0.021, 0.018, 0.028, 0.058, 0.255, 0.176]),
    'Trans + Ref': np.array([0.021, 0.019, 0.028, 0.055, 0.276, 0.129]),
    'Rot+Ref+Trans': np.array([0.020, 0.019, 0.033, 0.065, 0.089, 0.137]),
    
    # Algoritmos y Heurísticas
    'LKH3': np.array([0.020, 0.015, 0.032, 0.043, 0.090, 0.132]),
    'Google OR-Tools': np.array([0.020, 0.016, 0.033, 0.048, 0.092, 0.133]),
    'Ant Colony Optimization': np.array([0.020, 0.016, 0.031, np.nan, np.nan, np.nan]),
    'Genetic Algorithm': np.array([0.000, 0.016, 0.031, np.nan, np.nan, np.nan])
}

# 4. Configurar el gráfico
fig, ax = plt.subplots(figsize=(18, 9))
multiplier = 0
colors = plt.cm.tab10.colors

# Definir qué elementos son heurísticas para diferenciarlos visualmente
heuristics = ['LKH3', 'Google OR-Tools', 'Ant Colony Optimization', 'Genetic Algorithm']
heuristic_colors = ['#4d4d4d', 'tab:blue', 'tab:purple', 'tab:orange'] 

# 5. Calcular Gaps, transformar CIs y dibujar
max_gap = 0
for idx, (name, costs) in enumerate(model_costs.items()):
    # Ignorar advertencias de operaciones con NaN
    with np.errstate(invalid='ignore', divide='ignore'):
        # Fórmula del Optimality Gap (%)
        gap = ((costs - concorde) / concorde) * 100
        ci_gap = (model_cis[name] / concorde) * 100
    
    # Actualizar max_gap omitiendo NaNs
    valid_max = np.nanmax(gap + ci_gap)
    if valid_max > max_gap:
        max_gap = valid_max
        
    offset = width * multiplier - (width * len(model_costs) / 2) + width / 2
    
    # Lógica de colores y tramas (hatching)
    if name in heuristics:
        c = heuristic_colors[heuristics.index(name)]
        hatch_pattern = '////'
        edge_c = 'black'
    else:
        hatch_pattern = ''
        edge_c = 'white'
        if name == 'None (Base)':
            c = 'black'
        else:
            c = colors[idx % 10]

    # Dibujar barras con sus respectivos intervalos de confianza (yerr)
    rects = ax.bar(x + offset, gap, width, label=name, color=c, edgecolor=edge_c, hatch=hatch_pattern,
                   yerr=ci_gap, capsize=2, error_kw={'elinewidth': 1.0, 'alpha': 0.8})
    
    # Añadir el número exacto manualmente con annotate (100% compatible con versiones antiguas)
    for i, rect in enumerate(rects):
        if not np.isnan(gap[i]):
            # Obtener la altura de la barra + el valor de la barra de error
            y_pos = rect.get_height() + ci_gap[i]
            x_pos = rect.get_x() + rect.get_width() / 2
            
            ax.annotate(f'{gap[i]:.1f}',
                        xy=(x_pos, y_pos),
                        xytext=(0, 5),  # 5 puntos de padding vertical por encima del error
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=90, fontsize=8)
            
    multiplier += 1

# 6. Añadir detalles, títulos y ejes
ax.set_ylabel('Optimality Gap (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Problem Scale (Nodes)', fontsize=12, fontweight='bold')
ax.set_title('Optimality Gap Evolution', fontsize=16, fontweight='bold', pad=20)

ax.set_xticks(x)
ax.set_xticklabels(tsp_sizes, fontsize=12, fontweight='bold')

# Aumentar el límite Y para que los números girados y las barras de error no se corten
ax.set_ylim(0, max_gap * 1.25)

ax.grid(axis='y', linestyle='--', alpha=0.7)

# Separar leyenda en dos columnas (Redes vs Heurísticas)
ax.legend(title='Strategies & Algorithms', title_fontsize='12', fontsize='10', 
          loc='upper left', ncol=2)

plt.tight_layout()
# Guardar directamente en formato SVG
plt.savefig('optimality_gap_barras_ci.svg', format='svg', bbox_inches='tight')
print("[*] Gráfica de barras con CI generada y guardada como optimality_gap_barras_ci.svg")