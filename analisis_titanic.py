import pandas as pd
import matplotlib.pyplot as plt

# Cargar el CSV en un DataFrame
df = pd.read_csv('train_and_test2.csv')

# Renombrar columna errónea '2urvived' a 'Survived' si existe
if '2urvived' in df.columns:
	df.rename(columns={'2urvived': 'Survived'}, inplace=True)
	print("Columna '2urvived' renombrada a 'Survived'.")

# Mostrar las primeras 5 filas e información general
print(df.head())
print(df.info())

# Calcular porcentaje de supervivencia por sexo
# Asume que las columnas se llaman 'Sex' y 'Survived' (0/1)
survival_by_sex = df.groupby('Sex')['Survived'].mean() * 100
survival_by_sex = survival_by_sex.round(2)

print('\nPorcentaje de supervivientes por sexo (%):')
print(survival_by_sex)

# También mostrar conteos y porcentajes para más detalle
counts = df.groupby('Sex')['Survived'].agg(total_survived='sum', total='count')
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

# --- Gráfico de supervivencia por clase (Pclass) ---
if 'Pclass' in df.columns:
	survival_by_pclass = df.groupby('Pclass')['Survived'].mean() * 100
	survival_by_pclass = survival_by_pclass.round(2).sort_index()

	print('\nPorcentaje de supervivientes por clase (Pclass) (%):')
	print(survival_by_pclass)

	ax2 = survival_by_pclass.plot(kind='bar', color=['#2ca02c', '#ff7f0e', '#1f77b4'])
	ax2.set_xlabel('Pclass')
	ax2.set_ylabel('Porcentaje de supervivencia (%)')
	ax2.set_title('Supervivencia por clase (Titanic)')
	ax2.set_ylim(0, 100)

	for p in ax2.patches:
		h = p.get_height()
		ax2.annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2, h), ha='center', va='bottom')

	plt.tight_layout()
	plt.show()