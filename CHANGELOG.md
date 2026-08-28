# Changelog

Todas as alterações relevantes deste projeto são registradas neste arquivo.

## [1.0.0] — 2026-08-28

### Adicionado

- README técnico com formulação matemática, resultados, reprodução, validação e limitações.
- Notebook executável com 12 células Markdown e 11 células Python intercaladas.
- Cinco quadros representativos e o vídeo final em 1080 × 1350.
- Licença MIT, avisos de terceiros, política de segurança e configuração do Dependabot.
- Fluxo de integração contínua para sintaxe, treinamento, gradientes, notebook e documentação.
- Validador de publicação para ativos, links, notebook, caminhos locais e conteúdo incompatível com a distribuição pública.

### Alterado

- Script de renderização do Windows tornado portável, com suporte a Manim no ambiente virtual ou no PATH.
- Script de redução de velocidade passou a validar origem, destino e fator, além de importar explicitamente o tipo de taxa de quadros usado como fallback.
- Escopo do repositório restringido à animação NumPy sobre treinamento de redes neurais.

### Validado

- Treinamento determinístico nas épocas 0, 1, 10, 50 e 100.
- Acurácia de treinamento final de 100% sobre 28 observações sintéticas.
- Maior erro relativo de gradiente igual a 1,533 × 10⁻⁹.
- Notebook executado integralmente sem erros.
- Vídeo final com 1080 × 1350, 60 FPS, 8.854 quadros e 147,567 segundos.
- Publicação pública conferida em `Everlinmart/neural-network-learning-manim`, incluindo código, notebook, imagens, vídeo e automação de validação.

## [0.2.0] — 2026-08-27

### Alterado

- Cores das classes ajustadas para azul e vermelho em todos os gráficos.
- Textos narrativos centralizados e apresentados sem ponto final.
- Rótulos de pesos e viés reposicionados para evitar sobreposição com conexões.
- Substituição numérica de $z_2$ conectada aos valores exibidos na rede.
- Definição e interpretação da ReLU adicionadas à animação.
- Valores de $h_1,\ldots,h_8$ movidos para cima dos neurônios.
- Fronteiras parciais reais animadas durante a incorporação progressiva dos neurônios ocultos.
- Camadas da rede alinhadas na cena de retropropagação.

### Adicionado

- Versão temporal duplicada do vídeo, com 147,567 segundos e sem alterações de conteúdo.

## [0.1.0] — 2026-08-27

### Adicionado

- Treinamento NumPy determinístico de uma rede 2–8–1.
- Conjunto sintético de duas meias-luas com 28 observações.
- Snapshots numéricos das épocas 0, 1, 10, 11, 50 e 100.
- Cena Manim em formato vertical com equações LaTeX.
- Validação independente dos gradientes por diferenças finitas centrais.
- Renderização-base em 1080 × 1350 a 60 FPS.
