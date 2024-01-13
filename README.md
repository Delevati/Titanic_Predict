# Titanic Machine Learning Project

Este projeto utiliza dados do Titanic obtidos no Kaggle, que foram levemente manipulados para melhorar os resultados. Algumas das alterações incluem a conversão do gênero (M/F) para binário.

## Modelos Testados

Foram testados três tipos de algoritmos de machine learning:
- SVM (Support Vector Machine) -> Este é o único modelo bem documentado em suas funções. 
- Random Forest
- KNN (k-Nearest Neighbors)

## Arquivos

- `correlation_matrix.txt`: Comparador de valores de correlação das colunas via seaborn.
- `test.csv`: Arquivos de teste contendo exemplos para comparação de acurácia, fornecidos pelo Kaggle.
- `train.csv`: Arquivos com dados reais, fornecidos pelo Kaggle.
- `gender_submission.csv`: Arquivo de comparação real, fornecido pelo Kaggle.
- `train_rforest_MF.csv`: Arquivo alterado, com a substituição de "male" e "female" por valores binários.

## Pasta Utils

A pasta `Utils` contém algumas ferramentas utilizadas para analisar, correlacionar e manipular dados, assim como realizar testes. Embora alguns resultados não tenham sido utilizados, os arquivos foram mantidos para referência. Alguns scripts incluem:

- `evaluate.py`: Compara arquivos de saída (pasta `Outputs`) com o arquivo `gender_submission`.
- `knn_mvletter_cabin.py`: Altera a string na coluna Cabin, mapeando letras para números.
- `knn_rmletter_cabin.py`: Remove letras da coluna Cabin, deixando apenas números.
- `matriz_correlacoes_colunas_plot.py`: Avaliação gráfica dos dados correlacionados usando seaborn.
- `matriz_correlacoes_colunas_text.py`: Exporta avaliação correlacional para um arquivo de texto.
- `Verify_data_head.py`: Permite visualizar os dados da tabela de forma diferente.


# Model SVM - model_svm_titanic79ac.py
 Obtive um resultado de acurácia de aproximadamente 79,42% utilizando as seguintes informações: ["Pclass", "Sex", "SibSp", "Parch", "Fare", "Age"].
    Accuracy: 0.7942643391521197


# Model Random Forest - model_rforest_titanic79ac.py
 Obtive um resultado de acurácia de aproximadamente 79,88% utilizando as seguintes informações: ["Pclass", "Sex", "SibSp", "Parch", "Fare", "Age"].
    Accuracy: 0.7988826815642458

# Model KNN (k-Nearest Neighbors) - model_knn_titanic80ac.py
 Obtive um resultado de acurácia de aproximadamente 79,88% utilizando as seguintes informações: ["Pclass", "SibSp", "Sex"].
    Accuracy: 0.800498753117207

