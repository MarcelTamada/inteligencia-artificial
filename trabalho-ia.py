import pandas as pd
import requests
import re
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def extrair_dados_filmes(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    filmes = soup.select('td.titleColumn')
    equipe = [a.attrs.get('title') for a in soup.select('td.titleColumn a')]
    classificacoes = [b.attrs.get('data-value') for b in soup.select('td.posterColumn span[name=ir]')]

    lista_filmes = []
    for index in range(len(filmes)):
        atores = re.findall("(.+?)(?:,|$)", equipe[index])
        top_atores = atores[:3]
        diretor = next((ator for ator in top_atores if "(dir.)" in ator), "")
        top_atores = [ator for ator in top_atores if ator != diretor]
        classificacao = float(classificacoes[index])

        dados_filme = {
            "diretor": diretor,
            "elenco": ", ".join(top_atores),
            "classificacao": classificacao
        }

        lista_filmes.append(dados_filme)

    return lista_filmes


url = 'https://www.imdb.com/chart/top/'
dados_filmes = extrair_dados_filmes(url)

df = pd.DataFrame(dados_filmes)

X = df[['diretor', 'elenco']]
y = df['classificacao']

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

vetorizador = CountVectorizer()
X_treino_transformado = vetorizador.fit_transform(X_treino.apply(lambda x: ' '.join(x), axis=1))
X_teste_transformado = vetorizador.transform(X_teste.apply(lambda x: ' '.join(x), axis=1))

modelo = LinearRegression()
modelo.fit(X_treino_transformado, y_treino)

previsoes = modelo.predict(X_teste_transformado)

for i in range(len(previsoes)):
    print(f"Classificação prevista: {previsoes[i]:.2f}, Classificação real: {y_teste.iloc[i]:.2f}")
