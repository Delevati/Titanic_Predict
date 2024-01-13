import pandas as pd

# Carregue o DataFrame a partir do arquivo CSV
df = pd.read_csv('/Users/luryan/Documents/persona_project/titanic_dataset/train.csv')

# Função para extrair apenas os números da string da coluna "Cabin"
def extrair_numeros(cabin_str):
    numeros = ''.join(c for c in cabin_str if c.isdigit())
    return numeros

# Aplica a função à coluna "Cabin"
df['Cabin'] = df['Cabin'].apply(lambda x: extrair_numeros(str(x)))

#se quiser visualizar anteriormente o header do documento com o cabin alterado.
# print(df)

# Salva as alterações no arquivo original
df.to_csv('seuarquivo.csv', index=False)
