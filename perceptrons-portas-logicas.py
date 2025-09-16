# import -> incorpora uma biblioteca
# (código já escrito para resolver algum problema específico)
import numpy as np

class Perceptron:
    # Declaração do construtor da classe
    def __init__(self):
        self.w1, self.w2, self.bias = 0, 0, 0
    
    def _activation_function(self, value, func_type='sigmoid'):
        if func_type == 'sigmoid':
            return 1 / (1 + np.exp(-value))
        elif func_type == 'step':
            return 1 if value >= 0 else 0

    def train(self, inputs, outputs, learning_rate=0.1, epochs=100, activation_func_type='sigmoid'):
        self.inputs = inputs
        self.outputs = outputs
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Inicialização de pesos iniciais de forma aleatória
        self.w1, self.w2, self.bias = np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)

        for i in range(epochs):
            for j in range(len(inputs)):
                # Entrada para a função de ativação
                net_input = self.w1 * inputs[j][0] + self.w2 * inputs[j][1] + self.bias
                
                # função de ativação
                activated_output = self._activation_function(net_input, activation_func_type)

                # atualização dos pesos por iteração
                error = outputs[j][0] - activated_output
                self.w1 = self.w1 + learning_rate * error * inputs[j][0]
                self.w2 = self.w2 + learning_rate * error * inputs[j][1]
                self.bias = self.bias + (learning_rate * error)

        return self.w1, self.w2, self.bias

    def predict(self, x1, x2, activation_func_type='sigmoid'):
        net_input = (x1 * self.w1) + (x2 * self.w2) + self.bias
        
        # A lógica de decisão para a função degrau é diferente, pois não há um limiar (threshold)
        if activation_func_type == 'step':
            return self._activation_function(net_input, 'step')
        else:
            return 1 if self._activation_function(net_input, 'sigmoid') > 0.5 else 0

# Teste para 3 entradas (opcional)
class Perceptron3Inputs:
    def __init__(self):
        self.w1, self.w2, self.w3, self.bias = 0, 0, 0, 0
    
    def _activation_function(self, value):
        return 1 / (1 + np.exp(-value))

    def train(self, inputs, outputs, learning_rate=0.1, epochs=100):
        self.w1, self.w2, self.w3, self.bias = np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)

        for i in range(epochs):
            for j in range(len(inputs)):
                net_input = self.w1 * inputs[j][0] + self.w2 * inputs[j][1] + self.w3 * inputs[j][2] + self.bias
                activated_output = self._activation_function(net_input)
                error = outputs[j][0] - activated_output
                
                self.w1 = self.w1 + learning_rate * error * inputs[j][0]
                self.w2 = self.w2 + learning_rate * error * inputs[j][1]
                self.w3 = self.w3 + learning_rate * error * inputs[j][2]
                self.bias = self.bias + (learning_rate * error)

        return self.w1, self.w2, self.w3, self.bias

    def predict(self, x1, x2, x3):
        net_input = (x1 * self.w1) + (x2 * self.w2) + (x3 * self.w3) + self.bias
        return 1 if self._activation_function(net_input) > 0.5 else 0

if __name__ == '__main__':
    # 1. Porta Lógica OR (Exemplo da aula)
    print("--- 1. Análise da Porta Lógica OR ---")
    inputs_or = [[0, 0], [0, 1], [1, 0], [1, 1]]
    outputs_or = [[0], [1], [1], [1]]
    
    perceptron_or = Perceptron()
    
    print("\n--- a) Variando a taxa de aprendizado (learning_rate) ---")
    # Taxa de aprendizado baixa: 0.01
    weights_low_lr = perceptron_or.train(inputs=inputs_or, outputs=outputs_or, learning_rate=0.01, epochs=100)
    print("Com learning_rate=0.01:", "Previsão para [1,1] ->", perceptron_or.predict(1, 1))

    # Taxa de aprendizado alta: 0.5
    weights_high_lr = perceptron_or.train(inputs=inputs_or, outputs=outputs_or, learning_rate=0.5, epochs=100)
    print("Com learning_rate=0.5:", "Previsão para [1,1] ->", perceptron_or.predict(1, 1))

    print("\n--- b) Variando a quantidade de épocas (epochs) ---")
    # Poucas épocas: 10
    weights_low_epochs = perceptron_or.train(inputs=inputs_or, outputs=outputs_or, learning_rate=0.1, epochs=10)
    print("Com epochs=10:", "Previsão para [1,1] ->", perceptron_or.predict(1, 1))
    
    # Muitas épocas: 1000
    weights_high_epochs = perceptron_or.train(inputs=inputs_or, outputs=outputs_or, learning_rate=0.1, epochs=1000)
    print("Com epochs=1000:", "Previsão para [1,1] ->", perceptron_or.predict(1, 1))

    # 2. Porta Lógica AND
    print("\n--- 2. Análise da Porta Lógica AND ---")
    inputs_and = [[0, 0], [0, 1], [1, 0], [1, 1]]
    outputs_and = [[0], [0], [0], [1]]
    
    perceptron_and = Perceptron()
    perceptron_and.train(inputs=inputs_and, outputs=outputs_and, learning_rate=0.1, epochs=100)
    
    print("Treinamento para a porta AND:")
    print("Previsão para [0,0] ->", perceptron_and.predict(0, 0))
    print("Previsão para [0,1] ->", perceptron_and.predict(0, 1))
    print("Previsão para [1,0] ->", perceptron_and.predict(1, 0))
    print("Previsão para [1,1] ->", perceptron_and.predict(1, 1))

    # 3. Alteração da função de ativação para DEGRAU
    print("\n--- 3. Análise da Função de Ativação DEGRAU ---")
    perceptron_step = Perceptron()
    perceptron_step.train(inputs=inputs_or, outputs=outputs_or, learning_rate=0.1, epochs=100, activation_func_type='step')
    
    print("Treinamento com função DEGRAU:")
    print("Previsão para [0,0] ->", perceptron_step.predict(0, 0, 'step'))
    print("Previsão para [0,1] ->", perceptron_step.predict(0, 1, 'step'))
    print("Previsão para [1,0] ->", perceptron_step.predict(1, 0, 'step'))
    print("Previsão para [1,1] ->", perceptron_step.predict(1, 1, 'step'))

    # 4. Opcional: Perceptron com 3 entradas
    print("\n--- 4. Perceptron com 3 Entradas ---")
    # Exemplo de porta lógica OR com 3 entradas
    inputs_3d = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
    outputs_3d = [[0], [1], [1], [1], [1], [1], [1], [1]]

    perceptron_3d = Perceptron3Inputs()
    perceptron_3d.train(inputs=inputs_3d, outputs=outputs_3d, learning_rate=0.1, epochs=100)

    print("Treinamento para 3 entradas (Porta OR):")
    print("Previsão para [1,1,1] ->", perceptron_3d.predict(1, 1, 1))
    print("Previsão para [0,0,0] ->", perceptron_3d.predict(0, 0, 0))
