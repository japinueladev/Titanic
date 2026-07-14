import pandas as pd
import matplotlib.pyplot as plt

# Cargar el CSV en un DataFrame
df = pd.read_csv('train_and_test2.csv')

# Mostrar las primeras 5 filas e información general
print(df.head())
print(df.info())

# Calcular porcentaje de supervivencia por sexo
# Asume que las columnas se llaman 'Sex' y 'Survived' (0/1)
survival_by_sex = df.groupby('Sex')['2urvived'].mean() * 100
survival_by_sex = survival_by_sex.round(2)

print('\nPorcentaje de supervivientes por sexo (%):')
print(survival_by_sex)

# También mostrar conteos y porcentajes para más detalle
counts = df.groupby('Sex')['2urvived'].agg(total_survived='sum', total='count')
counts['percent'] = (counts['total_survived'] / counts['total'] * 100).round(2)

print('\nConteos y porcentajes por sexo:')
print(counts)

# Gráfico de barras del porcentaje de supervivencia por sexo
ax = survival_by_sex.plot(kind='bar', color=['#4C72B0', '#DD8452'])
ax.set_ylabel('Porcentaje de supervivencia (%)')
ax.set_title('Supervivencia por sexo (Titanic)')
ax.set_ylim(0, 100)

# Anotar los valores encima de las barras
for p in ax.patches:
	height = p.get_height()
	ax.annotate(f"{height:.1f}%", (p.get_x() + p.get_width() / 2, height),
				ha='center', va='bottom')

plt.tight_layout()
plt.show()