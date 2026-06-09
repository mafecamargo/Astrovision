# Classificação de Imagens de Desastres usando Redes Neurais Convolucionais

## Integrantes: 
- Fernanda Kaory Saito - RM551104
- João Pedro Borsato da Cruz - RM550294
- Maria Fernanda Vieira de Camargo - RM97956
- Pedro Lucas de Andrades Nunes - RM550366
- Vinicius Almeida Bernardino de Souza - RM97888

---

# Visão Geral

Este projeto investiga o uso de Redes Neurais Convolucionais Profundas, também conhecidas como CNNs, para a classificação automática de imagens relacionadas a desastres. O objetivo é treinar e avaliar diferentes arquiteturas modernas de classificação de imagens e comparar seus desempenhos em um dataset composto por imagens de desastres.

O projeto segue um fluxo completo de aprendizado de máquina:

1. Preparação e pré-processamento do dataset
2. Treinamento dos modelos utilizando transfer learning
3. Validação e teste dos modelos
4. Comparação quantitativa dos resultados
5. Implantação interativa utilizando Streamlit

O sistema final permite que o usuário faça o upload de uma imagem e receba como resultado a categoria de desastre prevista pelo modelo, juntamente com a pontuação de confiança da classificação.

---
# Video no Youtube

````text
link 
````

---

# Motivação

Desastres naturais, como enchentes, terremotos, incêndios florestais e deslizamentos de terra, podem causar grandes danos à infraestrutura, aos ecossistemas e às populações humanas.

Durante operações de resposta a emergências, grandes quantidades de imagens são coletadas a partir de diferentes fontes, como:

- Redes sociais
- Câmeras de vigilância
- Drones
- Satélites
- Dispositivos móveis

A análise manual dessas imagens é demorada e difícil de escalar. Sistemas automáticos de classificação de imagens podem auxiliar equipes de gerenciamento de emergências ao categorizar rapidamente informações visuais recebidas e identificar regiões afetadas.

Este projeto explora se arquiteturas modernas de CNN conseguem distinguir de forma eficiente diferentes categorias de desastres utilizando transfer learning.

---

# Dataset

O dataset é composto por imagens agrupadas em categorias relacionadas a desastres.

```text
data/
├── Damaged_Infrastructure
│   ├── Earthquake
│   └── Infrastructure
│
├── Fire_Disaster
│   ├── Urban_Fire
│   └── Wild_Fire
│
├── Land_Disaster
│   ├── Drought
│   └── Land_Slide
│
├── Non_Damage
│   ├── Non_Damage_Buildings_Street
│   ├── Non_Damage_Wildlife_Forest
│   └── sea
│
└── Water_Disaster
```

A implementação atual considera as pastas de primeiro nível como classes:

| ID da Classe | Nome da Classe          |
| ------------ | ----------------------- |
| 0            | Damaged_Infrastructure  |
| 1            | Fire_Disaster           |
| 2            | Land_Disaster           |
| 3            | Non_Damage              |
| 4            | Water_Disaster          |

Dessa forma, o problema tratado neste projeto é uma tarefa de classificação de imagens com 5 classes.

---

# Arquiteturas dos Modelos

Duas arquiteturas de Redes Neurais Convolucionais foram avaliadas neste projeto.

## 1. VGG16

A VGG16 é uma rede neural convolucional profunda introduzida pelo Visual Geometry Group, da Universidade de Oxford.

Diagrama da arquitetura do modelo VGG16: 
![VGG16 Model Architecture](https://raw.githubusercontent.com/kennethleungty/Neural-Network-Architecture-Diagrams/main/vgg16_image.png)

Características:

- 16 camadas treináveis
- Arquitetura sequencial
- Utiliza pequenos filtros convolucionais de tamanho 3×3
- Possui aproximadamente 138 milhões de parâmetros

Vantagens:

- Arquitetura simples
- Bom modelo de referência
- Fácil de entender e interpretar

Limitações:

- Alto consumo de memória
- Treinamento e inferência mais lentos
- Maior risco de overfitting

---

## 2. ResNet50

A ResNet50 foi introduzida pela Microsoft Research e apresentou o conceito de aprendizado residual.

Características:

- 50 camadas
- Conexões residuais, também chamadas de skip connections
- Possui aproximadamente 25 milhões de parâmetros

Vantagens:

- Rede mais profunda
- Melhor propagação do gradiente
- Convergência mais rápida durante o treinamento
- Geralmente apresenta melhor acurácia

Limitações:

- Arquitetura mais complexa
- Menos interpretável quando comparada à VGG16

---

# Métricas de Avaliação

Os modelos são comparados utilizando as seguintes métricas:

- Acurácia
- Precisão
- Recall
- F1-score
- Matriz de Confusão

Essas métricas permitem avaliar não apenas a taxa geral de acertos do modelo, mas também seu desempenho em cada classe individualmente.

---
# Pré requisitos para fazer o treinamento dos modelos:

Entrar no link: 
```` text 
https://fiapcom-my.sharepoint.com/:f:/g/personal/rm97956_fiap_com_br/IgAUzPAnCv2xSaLtsWg52Sd-AcJ9WomXpsnglrqVktGauCs?e=dfUjcv 
````

Neste link há as pastas checkpoint e data. Para executar o treinamento de forma correta é necessário baixar os arquivos e adicioná-los em suas respectivas pastas.


Além disso, é necessário verificar em requirements.txt se as bibliotecas necessárias estão nas versões corretas/ atualizadas.

---
# Treinamento de um Modelo

Exemplo de treinamento utilizando a ResNet50:

```bash
python train.py
```

Dentro do arquivo `train.py`, o modelo é definido por meio da variável:

```python
MODEL_NAME = "resnet50"
```

Para utilizar a VGG16, basta alterar o valor da variável para:

```python
MODEL_NAME = "vgg16"
```

O melhor checkpoint do modelo será salvo automaticamente durante o treinamento.

---

# Aplicação com Streamlit

O projeto inclui uma interface interativa de implantação utilizando Streamlit.

Para iniciar a aplicação, execute:

```bash
streamlit run app.py
```

A aplicação web permite que o usuário:

1. Faça o upload de uma imagem
2. Execute a inferência utilizando o modelo treinado
3. Visualize a classe prevista
4. Visualize a pontuação de confiança
5. Observe as probabilidades atribuídas a cada classe

---

# Fluxo Esperado de Execução

Treinar o modelo ResNet50:

```bash
python train.py
```

Iniciar a aplicação:

```bash
streamlit run app.py
```

Fazer o upload de uma imagem, por exemplo:

```text
flood_scene.jpg
```

Exemplo de saída esperada:

```text
Classe Prevista: Water_Disaster

Confiança: 94.7%
```

---

# Melhorias Futuras

Possíveis melhorias e extensões para este projeto incluem:

- Uso da arquitetura EfficientNet
- Uso da arquitetura DenseNet
- Uso de Vision Transformers, também conhecidos como ViT
- Otimização de hiperparâmetros
- Validação cruzada com K-fold
- Métodos de explicabilidade, como Grad-CAM
- Classificação multi-label de desastres
- Implantação utilizando Docker
- Integração com uma API REST

---

# Referências

- Simonyan, K., & Zisserman, A. (2015). *Very Deep Convolutional Networks for Large-Scale Image Recognition*.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*.
- Documentação do PyTorch: https://pytorch.org

