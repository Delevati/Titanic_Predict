import csv

def ler_arquivo_csv(caminho):
    with open(caminho, 'r', newline='') as arquivo_csv:
        leitor_csv = csv.reader(arquivo_csv)
        return list(leitor_csv)

def calcular_porcentagem_acerto(arquivo_real, arquivo_resultado):
    dados_real = ler_arquivo_csv(arquivo_real)
    dados_resultado = ler_arquivo_csv(arquivo_resultado)

    total_linhas = len(dados_real)
    linhas_acertadas = sum(linha_real == linha_resultado for linha_real, linha_resultado in zip(dados_real, dados_resultado))

    porcentagem_acerto = (linhas_acertadas / total_linhas) * 100

    return porcentagem_acerto

if __name__ == "__main__":
    arquivo_real = "/Users/luryan/Documents/persona_project/titanic_dataset/gender_submission.csv"
    arquivo_resultado = "/Users/luryan/Documents/persona_project/titanic_dataset/submission_knn.csv"

    porcentagem_acerto = calcular_porcentagem_acerto(arquivo_real, arquivo_resultado)

    print(f"A porcentagem de acerto do resultado em relação aos dados reais é: {porcentagem_acerto}%")
