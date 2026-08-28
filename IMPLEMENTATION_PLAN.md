# Plano de implementação

Status: concluído em 2026-08-27 — segunda versão com treinamento real e revisão corretiva validada.

## Objetivo

Preservar a animação original e criar uma segunda versão vertical, com no máximo dois minutos, na qual dados, pesos, previsões, fronteiras e perda sejam provenientes da mesma rede neural treinada de forma determinística com NumPy.

## Etapas

1. [Concluído] Gerar duas luas entrelaçadas e treinar uma rede 2–8–1 com NumPy e semente fixa.
2. [Concluído] Registrar pesos, previsões, perdas e superfícies de decisão reais nas épocas 0, 1, 10, 50 e 100.
3. [Concluído] Mostrar corretamente que um único neurônio produz uma fronteira reta.
4. [Concluído] Mostrar as retas completas dos neurônios ocultos e a fronteira não linear obtida pela combinação.
5. [Concluído] Conectar numericamente ativação, erro, gradiente, atualização de peso e deslocamento da fronteira.
6. [Concluído] Renderizar uma prévia e inspecionar os plots e as transições críticas.
7. [Concluído] Renderizar em 1080 × 1350 a 60 FPS, validar a mídia final e atualizar o changelog.

## Revisão corretiva

1. [Concluído] Tornar explícita a combinação das oito ativações ocultas no neurônio de saída.
2. [Concluído] Informar que quatro retas são mostradas, mas que a fronteira final usa os oito neurônios.
3. [Concluído] Separar conceitualmente erro absoluto, BCE individual e perda média de treinamento.
4. [Concluído] Explicar a normalização e identificar a acurácia como métrica do conjunto de treinamento.
5. [Concluído] Validar `W1`, `b1`, `W2` e `b2` por diferenças finitas sem assumir monotonicidade estrita como propriedade geral.
6. [Concluído] Reduzir a duração da versão base, corrigir espaçamentos, verificar quadros representativos e preservar o loop de abertura/fechamento.

## Revisão visual baseada nas capturas

1. [Concluído] Usar azul e vermelho em todas as ocorrências dos gráficos de classes.
2. [Concluído] Remover pontos finais dos textos narrativos e centralizar corretamente os blocos de frases.
3. [Concluído] Reposicionar pesos e viés fora das conexões e ligar a expressão simbólica de `z_2` à substituição numérica real por animações e cores semânticas.
4. [Concluído] Colocar os valores de `h_1,\ldots,h_8` acima dos círculos do neurônio de saída.
5. [Concluído] Explicar visualmente a definição e os dois casos da ReLU.
6. [Concluído] Construir fronteiras parciais reais durante a entrada progressiva dos neurônios ocultos e então mostrar a fronteira final dos oito neurônios.
7. [Concluído] Alinhar as camadas da rede na cena de retropropagação.
8. [Concluído] Inspecionar quadros representativos em resolução integral e gerar as versões normal (73,783 s) e lenta (147,567 s), ambas em 1080 × 1350 a 60 FPS.

## Publicação técnica no GitHub — 2026-08-28

Status: concluído em 2026-08-28 — repositório público, artefatos e documentação verificados.

### Escopo público

Publicar somente os componentes da animação NumPy sobre treinamento de redes neurais, mantendo fora do commit os experimentos Manim não relacionados, ambientes locais, distribuições LaTeX e o repositório de referência ManimML.

### Etapas

1. [Concluído] Auditar o conteúdo para credenciais, caminhos absolutos, dados locais, arquivos gerados desnecessários e menções incompatíveis com a publicação solicitada.
2. [Concluído] Organizar código, scripts de renderização, dependências, imagens representativas e o vídeo final em caminhos portáveis.
3. [Concluído] Criar um README técnico com modelo matemático, arquitetura 2–8–1, resultados numéricos, instalação, reprodução e limitações.
4. [Concluído] Criar um notebook intercalando explicação, fórmulas e código executável baseado no módulo real de treinamento.
5. [Concluído] Validar sintaxe, treinamento, gradientes, notebook, links relativos, mídia e ausência de termos proibidos no conjunto que será versionado.
6. [Concluído] Atualizar o changelog, revisar o índice Git, criar commits focados e publicar em um repositório público.
