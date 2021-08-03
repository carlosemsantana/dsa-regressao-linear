# Regressão


Em diversos problemas das áreas médica, biológica, industrial, química, finanças, engenharia entre outras, é de grande interesse verificar se duas ou mais variáveis estão relacionadas de alguma forma. Para expressar esta relação é muito importante estabelecer um modelo matemático. Esse tipo de modelagem ajuda a entender como determinadas variáveis influenciam outra variável, ou seja, verifica como o comportamento de uma(s) variável(is) pode mudar o comportamento de outra. 


A regressão linear ajuda a prever o valor de uma variável desconhecida (uma variável contínua) com base em um valor conhecido. Uma aplicação poderia ser: “Qual é preço de uma casa com base em seu tamanho?”

O preço é o valor que queremos prever, com base no tamanho da casa.

Para resolver esse problema, teríamos que buscar dados históricos de tamanho e preço de casa, treinar um modelo, aprender a relação matemática entre os dados e então fazer a previsão de preços com base em outros tamanhos de casa. Dado que estamos analisando o histórico para estimar um novo preço, ele se torna um problema de regressão. O fato de preço e tamanho estarem linearmente relacionados (quanto maior o tamanho da casa) o torna um problema de regressão linear.



Para exemplificar, geraremos um datasset com informações a partir de uma função do primeiro grau.


O Modelo Preditivo de Machine Learning é uma função matemática, aproximada, que foi encontrada através do treinamento com dados coletados e que permite fazer as previsões.

```python
from IPython.display import Image
Image('img/grafico.png')
```

### Variáveis Dependente e Independente

Uma variável dependente é o valor que estamos prevendo e uma variável independente é a variável que estamos usando para prever uma variável dependente. Por exemplo, a temperatura é diretamente proporcional ao número de sorvetes comprados. À medida que a temperatura aumenta, o número de sorvetes comprados também aumenta. Aqui a temperatura é a variável independente e, com base nela, o número de sorvetes comprados (a variável dependente) é previsto.

Uma variável independente x, explica a variação em outra variável, que é chamada variável dependente y. Este relacionamento existe em apenas uma direção: variável independente (x) --> variável dependente (y)

    


```python
from IPython.display import Image
Image('img/previsoes.png')
```

### O Que Representa a Correlação?

Do exemplo anterior, podemos notar que as compras de sorvete estão diretamente correlacionadas (ou seja, elas se movem na mesma direção) com a temperatura.

Neste exemplo, a correlação é positiva: à medida que a temperatura aumenta, as vendas de sorvete aumentam. Em outros casos, a correlação pode ser negativa: por exemplo, as vendas de um item podem aumentar à medida que o preço do item diminui.



### Correlação positiva

```python
import numpy as np
dados = [0.5*y+3 for y in np.arange(15)]
dados
```

```python
# Grafico
import matplotlib.pyplot as plt
plt.plot(dados, '-', dados, 'ro')
plt.grid(True)
plt.show()
```

### Correlação negativa

```python
import numpy as np
dados2 = [-1*0.5*y+3 for y in np.arange(14)]
dados2
```

```python
# Grafico
import matplotlib.pyplot as plt
plt.plot(dados2, '-', dados2, 'ro')
plt.grid(True)
plt.show()
```

### Sem correlação

```python
rng = np.random.default_rng()
data3 = [9,1,5,9.5,0,10,-9,5,3,9,8,-7,-3,3,2,5,3,4,2,9,0,10,-9,5,3,9,8,-7,-3,6,3,4,5,6]
rng.shuffle(data3)  # shuffle the list in-place

```

```python
# Grafico
import matplotlib.pyplot as plt
plt.plot(data3, 'ro')
plt.grid(True)
plt.show()
```

### Correlação Não Implica Causalidade

No entanto, intuitivamente, podemos dizer com confiança que a temperatura não é controlada pela venda de sorvetes, embora o inverso seja verdadeiro. Isso traz à tona o conceito de causalidade, qual evento influencia outro evento. A temperatura influencia as vendas de sorvete - mas não vice-versa.

Análise de regressão é uma metodologia estatística que utiliza a relação entre duas ou mais variáveis quantitativas de tal forma que uma variável possa ser predita a partir de outra, mas isso não implica causalidade. Ou seja, porque existe correlação entre duas variáveis não significa que uma é a causa da outra!



### Tipo dos Modelos de Regressão

    - Simples (uma variável dependente Y e uma variável independente X);
    - Multiplo; (uma variável dependente Y e duas ou mais variáveis independente X1, X2, ..., Xn );


### A análise de regressão compreende quatro tipos básicos de modelos:


    - Linear simples;
    - Não linear simples;
    - Linear múltiplo;
    - Não linear múltiplo;


### Formalizando a Regressão Linear Simples

Agora que temos os termos básicos definidos, vamos nos aprofundar nos detalhes da regressão linear.

Uma regressão linear simples é representada pela equação abaixo:


```python
from IPython.display import Image
Image('img/variaveis.png')
```

```python
from IPython.display import Image
Image('img/funcao.png')
```

***Onde:***

    y é a variável dependente que estamos prevendo.
    x é a variável independente.
    a é o termo de viés.
    b é a inclinação da variável (o peso atribuído à variável independente).

Y e X são as variáveis dependente e independente respectivamente. Vamos focar nos coeficientes (a e b na equação anterior).

Começamos com o coeficiente a, também chamado de viés ou bias. Considere o exemplo:

Queremos estimar o peso de um bebê pela idade do bebê em meses. Assumiremos que o peso de um bebê depende exclusivamente de quantos meses ele tem. O bebê tem 3 kg ao nascer e seu peso aumenta a uma taxa constante de 0,75 kg por mês. No final de um ano (12 meses), o gráfico do peso do bebê seria assim:


```python
# Exemplo
bebe = [3 + (0.75 * i) for i in range(13)]
bebe
```

```python
# Grafico
import matplotlib.pyplot as plt
plt.plot( bebe, '-', bebe, 'ro')
plt.grid(True)
plt.show()
```

No gráfico, o peso do bebê começa em 3 (a, o viés) e aumenta linearmente em 0,75 (b, a inclinação) a cada mês. Observe que, um termo de viés é o valor da variável dependente quando todas as variáveis independentes são 0.

A inclinação (ou slope) de uma linha é a diferença entre as coordenadas x e y nos dois extremos da linha e no comprimento da linha. No exemplo anterior, o valor da inclinação (b) é o seguinte:

(Diferença entre as coordenadas y nos dois extremos) / (Diferença entre as coordenadas x nos dois extremos)



```python
b = (12 -3) / (12 - 0) 
print(b)
```

***Solução de uma Regressão Linear Simples***

Vimos um exemplo simples de como a saída de uma regressão linear simples pode parecer (resolvendo viés e inclinação). Vamos agora encontrar uma maneira mais generalizada de gerar uma linha de regressão. O conjunto de dados fornecido é o seguinte:


```python
idade = [i for i in range(13)]
idade
```

```python
peso = [3 + (0.75 * i) for i in range(13)]
peso
```

Dado que estamos estimando o peso do bebê com base em sua idade, a regressão linear pode ser construída da seguinte maneira:

y = a + bx

Vamos resolver o problema a partir dos primeiros registros. Vamos supor que o conjunto de dados tenha apenas 2 pontos de dados. A modelagem ficaria assim:

Cálculo para o primeiro mês:


Cálculo para o primeiro mês:

    y = a + b*x
    3.00 = a + b*(0)
    a = 3.00

Cálculo para o segundo mês:

    y = a + b*x
    3.75 = a + b*(1)
    3.75 = 3.00 + b*(1)
    b = 3.75 - 3.00
    b = 0.75



As linhas anteriores já representam o treinamento do modelo e fomos capazes de prever os valores de a e b, nesse caso, a = 3.00 e b = 0.75. Isso é o que o modelo aprende durante o treinamento!


Se aplicarmos os valores de a e b nos demais pontos de dados restantes acima, conseguimos ter como resultado exatamente o valor de y. No entanto, isso provavelmente não seria o caso na prática, porque a maioria dos dados reais não tem uma relação assim tão perfeita. Por isso precisamos treinar o modelo com mais dados e encontrar uma formação mais genérica.


### Método dos Mínimos Quadrados

No cenário anterior, vimos que os coeficientes são obtidos usando apenas dois pontos de dados do conjunto total de dados - ou seja, não consideramos a maioria das observações na elaboração de valores ótimos de a e b. Para evitar deixar de fora a maioria dos pontos de dados durante a construção da equação, podemos modificar o objetivo para minimizar o erro quadrático geral (mínimos quadrados comuns) em todos os pontos de dados.


```python
from IPython.display import Image
Image('img/minimos-quadrados.png')
```

### Minimizando a Soma Geral do Erro ao Quadrado

O Método dos Mínimos Quadrados é o método de computação matemática pelo qual se define a reta de regressão. Esse método definirá uma reta que minimizará a soma das distâncias ao quadrado entre os pontos plotados (X, Y) e a reta (que são os valores previstos de Y’).

O erro ao quadrado geral é definido como a soma da diferença ao quadrado entre os valores reais e previstos de todas as observações. A razão pela qual consideramos o valor do erro ao quadrado e não o valor real do erro é que não queremos erro positivo em alguns pontos de dados compensando erros negativos em outros pontos de dados.

Por exemplo, um erro de +5 em três pontos de dados compensa um erro de –5 em três outros pontos de dados, resultando em um erro geral de 0 entre os seis pontos de dados combinados. O erro quadrático converte o erro –5 dos três últimos pontos de dados em um número positivo, para que o erro quadrático geral se torne 6 × 5^2 = 150. Isso levanta uma questão: por que devemos minimizar o erro quadrático geral? O princípio é o seguinte:

1. O erro geral é minimizado se cada ponto de dados individual for previsto corretamente.

2. Em geral, a superpredição em 5% é tão ruim quanto a subpredição em 5%, portanto, consideramos o erro ao quadrado.

```python
from IPython.display import Image
Image('img/mmq.png')
```

<!-- #region -->
Pelo método dos mínimos quadrados calculam-se os parâmetros “a“ e “b” da reta que minimiza estas distâncias ou as diferenças (ou o erro) entre Y e Y’. Esta reta é chamada de reta de regressão. Para que a soma dos quadrados dos erros tenha um valor mínimo, devem-se aplicar os conceitos de cálculo diferencial com derivadas parciais. Como as incógnitas do problema são os coeficientes "a" e "b" estrutura-se um sistema de duas equações. Assim aplicando os conceitos acima referidos monta-se o sistema de equações normais que permitirá extrair os valores de a e b.

A reta de regressão que se obtém através do método dos mínimos quadrados é apenas uma aproximação da realidade, ela é um modo útil para indicar a tendência dos dados. Mas até que ponto a reta de regressão obtida é útil para avaliar a realidade? Duas medidas podem indicar o quanto útil ou aproximado da realidade é a reta:
    
    - Erro padrão da estimativa
    - Coeficiente de determinação

### Erro Padrão da Estimativa

O erro padrão da regressão (S), também conhecido como erro padrão da estimativa, representa a distância média em que os valores observados "caem" da linha de regressão. Convenientemente, ele mostra o quão errado o modelo de regressão está, em média, usando as unidades da variável de resposta. Valores menores são melhores porque indica que as observações estão mais próximas da linha ajustada.

S mede a precisão das previsões do modelo. O erro padrão da estimativa é usado junto com o R Squared (Coeficiente de Determinação) na seção de ajuste da maioria dos resultados estatísticos. Ambas as medidas fornecem uma avaliação numérica de quão bem um modelo se ajusta aos dados da amostra. No entanto, existem diferenças entre as duas estatísticas.

O erro padrão da regressão fornece a medida absoluta da distância típica em que os pontos de dados "caem" da linha de regressão. S está nas unidades da variável dependente.

O R Squared fornece a medida relativa da porcentagem da variação da variável dependente explicada pelo modelo. O R Squared pode variar de 0 a 100%.


### Coeficiente de Determinação

O coeficiente de determinação (R Squared) deve ser interpretado como a proporção de variação total da variável dependente Y que é explicada pela variação da variável independente X.

O coeficiente de determinação é igual ao quadrado do coeficiente de correlação. Assim a partir do valor do coeficiente de determinação podemos obter o valor do coeficiente de correlação. O coeficiente de determinação é sempre positivo, enquanto que o coeficiente de correlação pode admitir valores entre -1 e +1. Valor igual a 1 indica perfeito relacionamento positivo, enquanto valor igual a -1 indica perfeito relacionamento negativo. Valores próximos de zero indicam que não há correlação.

O coeficiente de determinação indica o quanto a reta de regressão explica o ajuste da reta, enquanto que o coeficiente de correlação deve ser usado como uma medida de força da relação entre as variáveis.

Para mensurarmos o poder explicativo de um determinado modelo de regressão, ou o percentual de variabilidade da variável Y que é explicado pelo comportamento de variação das variáveis preditoras, precisamos entender alguns importantes conceitos.

Soma Total dos Quadrados (STQ ou SST) – Mostra a variação em Y em torno da própria média.

Soma dos Quadrados de Regressão (SQR) – Oferece a variação de Y considerando as variáveis X utilizadas no modelo.

Soma dos Quadrados dos Resíduos (SQU ou SSE) – Variação de Y que não é explicada pelo modelo elaborado.

STQ = SQR + SQU

<!-- #endregion -->

```python
from IPython.display import Image
Image('img/r2.png')
```

R2 é a fração da variância da amostra de Yi explicada (ou prevista) pelas variáveis preditoras. Para um modelo de regressão simples, esta medida mostra quanto do comportamento da variável Y é explicado pelo comportamento de variação da variável X, sempre lembrando que não existe, necessariamente, uma relação de causa e efeito entre as variáveis X e Y. Para um modelo de regressão múltipla, esta medida mostra quanto do comportamento da variável Y é explicado pela variação conjunta das variáveis X consideradas no modelo.

O coeficiente de ajuste R2 não diz aos analistas se uma determinada variável explicativa é estatisticamente significante e se esta variável é a causa verdadeira da alteração de comportamento da variável dependente.


```python
import pandas as pd
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings(action='once')
```

```python
# Carregando o dataset
df = pd.read_csv('dados/pesos.csv')
```

```python
# Criando o Modelo de Regressão
estimativa = smf.ols(formula = 'Peso ~ Idade', data = df)
```

```python
# Treinando o Modelo de Regressão
modelo = estimativa.fit()
```

```python
# Imprimindo o resumo do modelo
print(modelo.summary())
```

```python
# Variável independente
x = df['Idade']
```

```python
# Gerando os valores previstos
valores_previstos = modelo.predict(x)
valores_previstos
```

```python
# Variável dependente
y = df['Peso']
```

```python
y
```

----


[Carlos Eugênio](https://carlosemsantana.github.io/)

Graduando Engenharia Mecatrônica


### Referências


- Data Science Academy - <a href="https://www.datascienceacademy.com.br">https://www.datascienceacademy.com.br</a>
