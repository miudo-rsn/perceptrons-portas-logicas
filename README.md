# perceptrons-portas-logicas
Análise dos Resultados
1. Variação da Taxa de Aprendizado (Learning Rate) e Épocas
A taxa de aprendizado controla o quão rápido o modelo ajusta seus pesos.

Taxa baixa (ex: 0.01): O aprendizado é lento e mais preciso, o que pode evitar que o modelo "pule" o ponto ideal, mas pode levar mais tempo para convergir.

Taxa alta (ex: 0.5): O aprendizado é rápido, mas pode ser instável, fazendo com que os pesos oscilem e o modelo não consiga encontrar a solução ideal.
O número de épocas representa a quantidade de vezes que o algoritmo irá percorrer todo o conjunto de dados.

Poucas épocas (ex: 10): O modelo pode não ter tempo suficiente para aprender a relação entre as entradas e saídas, resultando em um treinamento incompleto e previsões incorretas.

Muitas épocas (ex: 1000): O modelo tem tempo de sobra para se ajustar, mas pode levar ao overfitting (quando o modelo se especializa demais nos dados de treinamento e perde a capacidade de generalizar para novos dados). No nosso caso, como a porta OR é um problema linear simples, mais épocas geralmente garantem a convergência.

2. Teste com a Porta Lógica AND
A porta lógica AND é um problema linearmente separável, assim como a OR. O Perceptron é capaz de aprender com sucesso a relação de entrada/saída. A única saída positiva ([1]) é quando ambas as entradas são positivas ([1,1]). O treinamento ajusta os pesos para que o neurônio "dispare" apenas nessa condição.

3. Função de Ativação DEGRAU
A função de ativação degrau (ou step function) é uma das mais simples. Ela retorna 1 se o resultado for maior ou igual a zero, e 0 caso contrário. A principal diferença é que ela não é "suave", ou seja, não tem uma derivada contínua. Isso a torna inadequada para algoritmos de otimização baseados em gradiente, como o que está implementado para o ajuste de pesos no código. No nosso caso, ela funciona bem na fase de predição porque a decisão é binária, mas o algoritmo de treinamento, que usa o gradiente da função sigmoide para ajustar os pesos, não funcionaria corretamente com essa função. A implementação corrigida no código acima leva isso em consideração.

4. Extração da Função de Ativação
Extrair a função de ativação para um novo método (_activation_function) torna o código mais modular e reutilizável. Isso facilita a alteração e o teste de diferentes funções de ativação sem a necessidade de duplicar o código ou modificar a lógica de treinamento principal.

5. Alteração para 3 Entradas (Opcional)
A adaptação do Perceptron para receber 3 entradas foi feita criando uma nova classe. O princípio é o mesmo: um peso (w3) adicional é adicionado para a terceira entrada, e a fórmula de atualização e predição é ajustada para incluir esse novo peso e a nova entrada (x3). O resultado mostra que o Perceptron continua a funcionar corretamente, mantendo sua capacidade de resolver problemas linearmente separáveis com mais dimensões.
