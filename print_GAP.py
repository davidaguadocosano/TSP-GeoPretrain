import matplotlib.pyplot as plt
import numpy as np

# 1. Definir los tamaños del eje X
tsp_sizes = ['TSP20', 'TSP50', 'TSP100', 'TSP200', 'TSP500', 'TSP1000']

# 2. Valores de coste del solucionador exacto (Baseline = 0% Gap)
concorde = np.array([3.829, 5.704, 7.790, 10.726, 16.489, 23.058])

# 3. Diccionario con los costes medios de cada modelo
model_costs = {
    'None (Base)': np.array([3.989, 6.087, 8.796, 13.321, 24.139, 38.940]),
    'Rotation': np.array([3.954, 6.039, 8.666, 13.080, 23.278, 36.835]),
    'Reflection': np.array([3.953, 6.030, 8.651, 12.927, 23.201, 36.694]),
    'Translation': np.array([4.008, 6.047, 8.729, 13.453, 24.970, 41.042]),
    'Rot + Refl': np.array([3.953, 6.030, 8.623, 12.936, 22.887, 36.123]),
    'Rot + Trans': np.array([3.973, 6.005, 8.630, 13.344, 26.782, 48.199]),
    'Trans + Ref': np.array([3.973, 6.070, 8.743, 13.183, 24.219, 38.855]),
    'Rot+Ref+Trans': np.array([3.949, 6.028, 8.626, 12.959, 23.350, 37.405])
}

# 4. Configurar el estilo del gráfico
plt.figure(figsize=(12, 7))

# Colores y estilos para diferenciar bien las líneas
colors = plt.cm.tab10.colors

# 5. Calcular Gaps y dibujar
for idx, (name, costs) in enumerate(model_costs.items()):
    # Fórmula del Optimality Gap (%)
    gap = ((costs - concorde) / concorde) * 100
    
    # Destacar visualmente el modelo Base y el mejor modelo (Rot + Refl)
    if name == 'None (Base)':
        plt.plot(tsp_sizes, gap, marker='s', label=name, color='black', linestyle='--', linewidth=3, markersize=8)
    elif name == 'Rot + Refl':
        plt.plot(tsp_sizes, gap, marker='D', label=name, color=colors[idx%10], linestyle='-', linewidth=3, markersize=8)
    elif 'Trans' in name:
        # Poner los modelos con traslación en líneas punteadas para ver cómo degeneran
        plt.plot(tsp_sizes, gap, marker='o', label=name, color=colors[idx%10], linestyle=':', linewidth=1.5)
    else:
        plt.plot(tsp_sizes, gap, marker='o', label=name, color=colors[idx%10], linestyle='-', linewidth=1.5)

# 6. Añadir detalles, títulos y leyendas
plt.ylabel('Optimality Gap (%)', fontsize=12, fontweight='bold')
plt.xlabel('Escala del Problema (Nodos)', fontsize=12, fontweight='bold')
plt.title('Evolución de la Brecha de Optimalidad', fontsize=14, fontweight='bold')

# Configurar la cuadrícula y la leyenda
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Estrategias de Pre-entrenamiento', title_fontsize='11', fontsize='10', loc='upper left')

# Ajustar márgenes y guardar imagen en alta calidad
plt.tight_layout()
plt.savefig('optimality_gap_extrapolation.png', dpi=300)
print("[*] Gráfica generada y guardada como optimality_gap_extrapolation.png")