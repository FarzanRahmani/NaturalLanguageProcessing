# Natural Language Processing

Graduate NLP coursework, workshops, and a multimodal retrieval-augmented generation project from the second semester of my M.Sc. in Computer Engineering at Sharif University of Technology. Instructor: Dr. Ehsaneddin Asgari. Author: Farzan Rahmani.

## Repository Guide

| Path | Contents |
| --- | --- |
| `HWs/HW1/` | Crawling, preprocessing, structured JSON extraction with language models, and annotation/analysis workflows |
| `HWs/HW2/` | Multi-phase NLP experiments, question preparation, similarity analysis, and evaluation outputs |
| `HWs/HW3/` | Multimodal exercises, including CLIP/LoRA experiments and text-image questions |
| `workshops/` | Web crawling, NLP frameworks, PyTorch, and Hugging Face tutorials |
| `project/Main_pipeline/` | Text/image retrieval components, configuration, initialization, RAG logic, and Gradio interface |
| `project/GenerativeModel/` | Generative-model experiments |
| `project/EvaluationCodes/` | Text-only and multimodal evaluation notebooks, with/without retrieval and different retrieval depths |
| `project/Questions/` | Question preprocessing material |

## Multimodal RAG Project

The main pipeline combines text retrieval, person-name extraction, image-based person search, and generative answering. `config.py` selects BGE-M3 text embeddings and a Persian NER model, and defines locations for JSON sources and cached embeddings. `setup.py` initializes the retrieval components. `rag.py` assembles retrieved context and calls the generative-model helper, with Gemini 2.5 Flash selected in the current function.

## Setup Considerations

Use Python with Jupyter for notebook work. The application additionally uses Gradio and the libraries imported by its text/image pipeline modules. There is no repository-wide dependency lockfile; the `Main_pipeline/setup.py` file is an initialization script, not Python packaging metadata.

Before attempting the project:

1. Review the notebooks, imports, model requirements, and any API authentication used by the generative helper.
2. Set `Config.PROJECT_PATH` for your machine; the checked-in value points to the original Linux development directory.
3. Provide the expected JSON data and image sources, and configure embedding caches and model downloads.
4. Initialize the components and pass them into the `RAG` function when experimenting with queries.

**The Gradio entry point is an incomplete integration snapshot:** `app.py` calls `RAG(question, image)`, while `rag.py` also requires initialized embedders, vector databases, a face recognizer, a name extractor, and JSON data. Those dependencies must be connected in a local adaptation before `python app.py` can serve working answers. The notebooks provide additional pipeline and evaluation examples.

## Reproducing Experiments

Run each notebook's setup and data preparation before evaluation. Check remote data availability, model access, GPU memory, API costs, and file paths for the chosen workflow. Recorded outputs compare specific settings and do not establish identical results across environments or model-service versions.

## License

See [LICENSE](LICENSE) for repository license terms. External models, datasets, APIs, and workshop materials retain their own terms and attribution.
