# RBNN — Radial Basis Nearest Neighbors

Implementação do zero de um classificador e de um regressor baseados em funções de base radial, aplicados às bases **Iris** e **Abalone** do repositório UCI.

Trabalho da disciplina de Aprendizado Supervisionado — Ciência de Dados e Inteligência Artificial, PUC-Campinas.

---

## Sobre o algoritmo

O RBNN é um método de aprendizado preguiçoso: não há etapa de treinamento, apenas o armazenamento do conjunto de referência. Toda a computação ocorre no momento da predição.

Diferentemente do KNN clássico, não existe seleção explícita de *k* vizinhos. Todos os pontos de treino participam de cada previsão, mas ponderados por uma função gaussiana da distância euclidiana:

```
w(d) = exp( -d² / 2σ² )
```

Pontos próximos recebem peso alto; pontos distantes têm peso que decai exponencialmente até se tornar desprezível. A seleção da vizinhança é, portanto, contínua em vez de discreta — o hiperparâmetro `σ` regula a largura efetiva dessa vizinhança.

**Classificação:** os pesos são somados separadamente por classe, e a classe de maior soma é atribuída à amostra.

**Regressão:** a previsão é a média dos rótulos de treino ponderada pelos pesos, equivalente ao estimador de Nadaraya-Watson:

```
ŷ = Σ(wᵢ · yᵢ) / Σ(wᵢ)
```

---

## Estrutura do projeto

```
.
├── Trabalho_1_RBNN_Iris_Abalone.ipynb   # notebook completo
└── README.md
```

Funções implementadas:

| Função | Descrição |
|---|---|
| `gaussiana(dist, sigma)` | Converte um vetor de distâncias em pesos |
| `classificar(X_train, y_train, X_test, sigma)` | Classificação por soma de pesos agrupada por classe |
| `regredir(X_train, y_train, X_test, sigma)` | Regressão por média ponderada dos rótulos |
| `testar_sigmas(...)` | Varredura de hiperparâmetros com MAE, RMSE, R² e contagem de nulos |

---

## Requisitos

```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

O `scikit-learn` é utilizado apenas para divisão treino/teste, padronização e cálculo de métricas — o algoritmo em si é implementado manualmente.

As bases são carregadas diretamente das URLs do repositório UCI, portanto a execução requer conexão com a internet.

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
jupyter notebook Trabalho_1_RBNN_Iris_Abalone.ipynb
```

---

## Metodologia

Ambas as bases seguem o mesmo pipeline:

1. Carregamento a partir do repositório UCI e verificação de valores ausentes
2. Separação treino/teste em 70/30, com semente fixa (`random_state=42`)
3. Padronização com `StandardScaler`, ajustada no treino e aplicada ao teste

A padronização é obrigatória: como todo o método depende de distância euclidiana, atributos em escalas diferentes distorceriam o cálculo. Na Abalone, `Whole_weight` chega a 2,8 enquanto `Height` permanece abaixo de 0,3.

---

## Resultados — Iris (classificação)

150 amostras, 4 atributos, 3 classes balanceadas. Divisão estratificada (15 amostras de cada classe no teste).

**σ = 0,25 — acurácia de 93,33%**

| Classe | Precisão | Recall | F1 |
|---|---|---|---|
| Iris-setosa | 1,00 | 1,00 | 1,00 |
| Iris-versicolor | 0,83 | 1,00 | 0,91 |
| Iris-virginica | 1,00 | 0,80 | 0,89 |

Os três erros do modelo são amostras de *Virginica* classificadas como *Versicolor*. *Setosa* é linearmente separável pelas medidas de pétala e não apresenta erros, enquanto as outras duas classes se sobrepõem na fronteira — amostras nessa região recebem peso alto de vizinhos da classe oposta.

---

## Resultados — Abalone (regressão)

4.177 amostras, 8 atributos, alvo `Rings` (número de anéis, proxy da idade). A variável categórica `Sex` foi codificada em *one-hot* por ser nominal.

A distribuição do alvo é assimétrica: mediana de 9 anéis, quartis em 8 e 11, mas valores que se estendem até 29.

### Comparação de valores de σ

| σ | MAE | RMSE | R² | Nulos | Nulos (%) |
|---|---|---|---|---|---|
| 0,01 | 1,8428 | 2,6504 | 0,1782 | 249 | 19,86 |
| 0,10 | 1,7722 | 2,6151 | 0,3271 | 1 | 0,08 |
| **0,25** | **1,5532** | **2,2522** | **0,5005** | **0** | **0,00** |
| 1,00 | 1,7153 | 2,4281 | 0,4194 | 0 | 0,00 |
| 100,00 | 2,3592 | 3,1903 | −0,0023 | 0 | 0,00 |

O critério de seleção foi o **RMSE**, escolhido por penalizar de forma desproporcional os erros grandes — que na Abalone se concentram na cauda superior de `Rings`, onde há poucas amostras de treino disponíveis como vizinhos. O MAE aponta para o mesmo valor de σ, indicando que a conclusão não depende da métrica de erro adotada.

Sigmas que produziram previsões nulas foram excluídos da seleção: como as métricas são calculadas apenas sobre previsões válidas, valores obtidos em subconjuntos de tamanhos diferentes não são comparáveis entre si.

### Comportamento nos extremos

A curva de RMSE em função de σ assume forma de U, com degradação assimétrica nas duas pontas.

**σ muito alto (100):** a gaussiana torna-se tão larga que todos os pesos ficam praticamente idênticos. A média ponderada colapsa na média global de `Rings` e o modelo passa a ignorar os atributos — o R² ligeiramente negativo reflete um desempenho equivalente ao de prever sempre a média.

**σ muito baixo (0,01):** o expoente da gaussiana atinge magnitude suficiente para que o resultado seja arredondado a zero em ponto flutuante. Com todos os pesos nulos, a média ponderada torna-se indefinida e 19,86% das amostras ficam sem previsão. Trata-se de uma limitação de representação numérica, não de desempenho do algoritmo.

**σ = 0,25:** define uma vizinhança local — ampla o bastante para que a previsão não dependa de um único ponto ruidoso, e restrita o bastante para que ainda reflita a região do espaço em que a amostra se encontra. É o equilíbrio entre viés e variância que o hiperparâmetro controla.

### Modelo final

| Métrica | Valor |
|---|---|
| MAE | 1,5532 |
| RMSE | 2,2522 |
| R² | 0,5005 |

A média dos erros é −0,048, indicando ausência de viés sistemático. O gráfico de valores previstos contra reais evidencia o achatamento característico do método: como a média ponderada nunca extrapola o intervalo dos rótulos de treino, o modelo subestima os valores altos e superestima os baixos, regredindo à região densa da distribuição.

O R² em torno de 0,50 é compatível com a literatura para esta base — a idade do molusco possui componentes biológicos que as medidas externas não capturam integralmente.

---

## Limitações e melhorias futuras

**Seleção de hiperparâmetro no conjunto de teste.** O valor de σ foi escolhido observando o desempenho no mesmo conjunto usado para reportar o resultado final, o que torna a métrica otimista. A abordagem correta seria validação cruzada dentro do treino, reservando o teste para uma única medição ao final.

**Grade de sigmas esparsa.** Não há valores testados entre 0,25 e 1, um intervalo de 4×. O mínimo real da curva pode estar nessa faixa.

**Ausência de estabilização numérica.** Subtrair a menor distância ao quadrado antes de exponenciar deslocaria os expoentes sem alterar as razões entre os pesos, tornando avaliáveis os sigmas abaixo de 0,25 — atualmente inacessíveis por underflow.

**Custo computacional.** A complexidade é O(n · m · d), com o laço de predição implementado em Python puro. Uma versão vetorizada com `scipy.spatial.distance.cdist` eliminaria o laço ao custo de materializar a matriz completa de distâncias.

**Ausência de baseline comparativo.** Executar um `KNeighborsRegressor` e uma regressão linear sobre a mesma divisão daria contexto ao R² obtido.

---

## Referências

- Dua, D. & Graff, C. *UCI Machine Learning Repository*. Universidade da Califórnia, Irvine.
  - [Iris](https://archive.ics.uci.edu/dataset/53/iris)
  - [Abalone](https://archive.ics.uci.edu/dataset/1/abalone)
- Nadaraya, E. A. (1964). On Estimating Regression. *Theory of Probability and Its Applications*.
- Watson, G. S. (1964). Smooth Regression Analysis. *Sankhyā: The Indian Journal of Statistics*.

---

## Autores

Gabbriel Nagano · Lucas Cordeiro Raw

Ciência de Dados e Inteligência Artificial — PUC-Campinas
