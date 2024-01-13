import pandas as pd

# Carregue o DataFrame a partir do arquivo CSV
df = pd.read_csv('/Users/luryan/Documents/persona_project/titanic_dataset/train.csv')

# Mapeamento de letras para números
mapeamento_letras = {'A': '1', 'B': '2', 'C': '3', 'D': '4', 'E': '5', 'F': '6', 'G': '7', 'T': '8'}

# Função para substituir o valor da coluna 'Cabin' com base na letra inicial
def substituir_cabin_por_letra(valor):
    for letra, numero in mapeamento_letras.items():
        if str(valor).startswith(letra):
            return numero
    return valor

# Aplica a função à coluna 'Cabin'
df['Cabin'] = df['Cabin'].apply(substituir_cabin_por_letra)

# Converte os valores para números (quando possível)
df['Cabin'] = pd.to_numeric(df['Cabin'], errors='coerce')

# Preenche os valores NaN na coluna 'Cabin' com a mediana
mediana_cabin = df['Cabin'].median()
df['Cabin'].fillna(mediana_cabin, inplace=True)

# Visualiza o DataFrame após as alterações
print(df)

# Salva as alterações no arquivo original
df.to_csv('seuarquivo.csv', index=False)
