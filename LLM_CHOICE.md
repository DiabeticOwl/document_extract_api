# LLM Choice Rationale: `phi3:mini`

## Introduction
This document outlines the decision-making process for selecting the Large Language Model (LLM) for the Intelligent Document Understanding API. The chosen model for the entity extraction task is microsoft/phi-3-mini-4k-instruct, accessed locally via Ollama as phi3:mini.

The primary goal was to find a model that balances high accuracy in JSON-based entity extraction with the practical constraints of local deployment. This means the model needed to be small enough to run efficiently on standard developer hardware without sacrificing the quality of its output.

## The Chosen Model: phi3:mini
phi3:mini is a 3.8 billion parameter, lightweight, state-of-the-art open model from Microsoft. It is designed to deliver performance comparable to much larger models in a significantly smaller package.

### Key Reasons for Selection:
- **Excellent Performance-to-Size Ratio**: phi3:mini has demonstrated remarkable capabilities in language understanding, reasoning, and instruction-following. It excels at generating structured output like JSON, which is the core requirement for our entity extraction task.
- **Optimized for Local Deployment**: With a quantized size of just over 2GB, it can run efficiently on consumer-grade CPUs and GPUs. This aligns perfectly with our goal of providing a self-contained, local-first application that does not depend on expensive cloud-based APIs.
- **State-of-the-Art Training Methodology**: It was trained on a highly curated dataset of "textbook-quality" synthetic data, heavily filtered web data, and code [^1]. This focus on high-quality, educational content is believed to be the reason for its strong reasoning and logic capabilities, which are crucial for accurately interpreting document layouts and extracting information.
- **Strong Instruction Following**: The model has been specifically trained to adhere to complex instructions, such as "Return your response as a valid JSON object" and "Provide no additional text...". This reduces the need for extensive output parsing and error correction.
- **Fast Inference Speed**: Its small size allows for rapid processing, which is crucial for a responsive API.

## Comparison with Alternatives
Several other leading open-source models were considered. The primary trade-off is between model size (resource requirements) and performance on reasoning and instruction-following tasks.

| **Model** 	| **Parameters** 	| **Approx. Size** 	| **Key Strengths** 	| **Considerations for this Project** 	|
|:---:	|:---:	|:---:	|:---:	|:---:	|
| phi3:mini (Chosen) 	| 3.8B 	| ~2.2 GB 	| Best-in-class reasoning for its size, excellent instruction following [^1].	| Ideal balance of performance and low resource usage. 	|
| Mistral-7B-Instruct 	| 7B 	| ~4.1 GB 	| Excellent all-around performance, strong community support [^2]. 	| Strongest competitor, but nearly 2x the size for potentially marginal gains on this specific task. 	|
| Llama-3-8B-Instruct 	| 8B 	| ~4.5 GB 	| Top-tier general performance, great at conversational tasks [^3]. 	| Overkill for this specific use case; higher resource cost is not justified. 	|
| DeepSeek-Coder-v2-Lite 	| 1.6B 	| ~1.0 GB 	| Heavily optimized for code and logic [^4]. 	| Very small and fast, but its specialization in code may make it less robust on general document text. 	|
| Gemma-2B-Instruct 	| 2B 	| ~1.5 GB 	| Very lightweight and fast [^5]. 	| Less capable than phi3:mini on complex reasoning and instruction following benchmarks [^1]. 	|

## Conclusion
After comparing phi3:mini with other top-tier small language models, it remains the optimal choice for the Intelligent Document Understanding API.

The decision is not based on phi3:mini being the most powerful model overall, but on it having the best-fit characteristics for the project's specific goals:

1. Local-First Deployment: Its small footprint is the most critical factor, making the API accessible to the widest range of users.
2. Structured Data Extraction: Its "textbook-quality" training gives it a distinct advantage in the logical reasoning required to parse documents and follow the strict JSON output format.

While Mistral 7B is a close second in terms of capability, phi3:mini's superior performance-to-size ratio makes it the clear winner for building a lightweight, efficient, and powerful local-first document extraction tool.

## References
[^1]: Abdin, M., et al. (2024). Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone. arXiv:2404.14219. Available: https://arxiv.org/abs/2404.14219

[^2]: Jiang, A. Q., et al. (2023). Mistral 7B. arXiv:2310.06825. Available: https://arxiv.org/abs/2310.06825

[^3]: Meta AI. (2024). The Llama 3 Herd of Models. Available: https://ai.meta.com/blog/meta-llama-3/

[^4]: Guo, D., et al. (2024). DeepSeek-Coder: When the Large Language Model Meets Programming - The Rise of Code Intelligence. arXiv:2401.14196. Available: https://arxiv.org/abs/2401.14196

[^5]: Google. (2024). Gemma: Open Models Based on Gemini Research and Technology. Available: https://blog.google/technology/developers/gemma-open-models/
