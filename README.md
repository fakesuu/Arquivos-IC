# Avaliação de Algoritmos Evolucionistas Gerados por LLMs

Este repositório contém a base de código e os testes de desempenho para a avaliação de algoritmos evolucionistas projetados iterativamente através de cadeias de prompts (Prompt Chaining) com Grandes Modelos de Linguagem (LLMs).

O objetivo atual é analisar o desempenho de três baselines robustos em problemas multimodais de alta dimensionalidade (suíte GNBG):
- **BIPOP-CMA-ES**
- **LM-CMA-ES**
- **L-SHADE**

Todos os algoritmos foram implementados com integração de **estratégias de niching**, reflexão de limites (reflective bounds) e redução linear do tamanho da população.

## Estrutura do Repositório
A geração de código foi dividida entre diferentes LLMs para fins de comparação da eficácia do *prompt engineering*.
- `/GPT` - Implementações completas geradas via OpenAI GPT.
- `/DeepSeek` - Implementações completas geradas via DeepSeek AI.
- `/Claude` - *(Em desenvolvimento - Aguardando liberação de cota de tokens)*.
- `Cadeia de Prompts.txt` - Documentação dos 8 passos de prompt utilizados para forçar o baseline, adaptar operadores e integrar ao benchmark.
