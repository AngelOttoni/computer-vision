import matplotlib.pyplot as plt
import numpy as np

# Criar figura
fig, ax = plt.subplots(figsize=(12, 8))

# Dados do tetraedro
points = {
    'P1': (102, 201),
    'P2': (300, 201),
    'P3': (102, 270), 
    'P4': (120, 221)
}

# Arestas com cores diferentes para melhor visualização
edges = [
    ('P1', 'P2', 'red'),    # Base - Vermelho
    ('P1', 'P3', 'blue'),   # Vertical - Azul  
    ('P1', 'P4', 'green'),  # Frontal - Verde
    ('P2', 'P3', 'red'),    # Base - Vermelho
    ('P2', 'P4', 'green'),  # Frontal - Verde
    ('P3', 'P4', 'blue')    # Traseira - Azul
]

# Desenhar arestas
for p1, p2, color in edges:
    x1, y1 = points[p1]
    x2, y2 = points[p2]
    ax.plot([x1, x2], [y1, y2], color=color, linewidth=3, alpha=0.8)

# Desenhar vértices
for name, (x, y) in points.items():
    ax.plot(x, y, 'o', markersize=10, color='black', markeredgecolor='black', markerfacecolor='yellow')
    ax.text(x + 8, y + 8, name, fontsize=14, fontweight='bold', color='darkblue')

# Configurações do gráfico
ax.set_xlabel('Coordenada X (pixels)', fontsize=14, fontweight='bold')
ax.set_ylabel('Coordenada Y (pixels)', fontsize=14, fontweight='bold')
ax.set_title('ESBOÇO DO TETRAEDRO NO PLANO DE IMAGEM', fontsize=16, fontweight='bold', pad=20)

# Eixo Y invertido (convenção de imagem)
ax.set_ylim(280, 180)
ax.set_xlim(80, 320)

# Grade
ax.grid(True, linestyle='--', alpha=0.7)

# Legenda
legend_elements = [
    plt.Line2D([0], [0], color='red', lw=3, label='Arestas da Base'),
    plt.Line2D([0], [0], color='blue', lw=3, label='Arestas Verticais'),
    plt.Line2D([0], [0], color='green', lw=3, label='Arestas Frontais'),
    plt.Line2D([0], [0], marker='o', color='yellow', markersize=8, markerfacecolor='yellow', markeredgecolor='black', label='Vértices')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

# Anotação com coordenadas
coord_text = "COORDENADAS:\n" + "\n".join([f"{p}: ({x}, {y})" for p, (x, y) in points.items()])
ax.text(250, 220, coord_text, fontsize=11, bbox=dict(boxstyle="round,pad=1", facecolor="lightcyan", edgecolor="blue"), 
        verticalalignment='top')

plt.tight_layout()
plt.show()

# Salvar em alta qualidade
fig.savefig('tetraedro_esboco_detalhado.png', dpi=300, bbox_inches='tight', facecolor='white')