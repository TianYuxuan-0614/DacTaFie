# DacTaFie📑
DacTaFie：Dual-Layer Agent Collaboration with Test-Time Adaptation for Form Information Extraction

## Introduction🚀
Form Information Extraction (FIE) is a critical task in document understanding but faces challenges such as poor domain adaptability, weak tool-awareness of agents, and low scalability.  
To address these issues, we propose **DacTaFie**, a dual-layer agent collaboration framework with test-time adaptation.  
It dynamically optimizes plugin execution chains, enabling efficient parsing of complex forms and robust cross-domain generalization **without model fine-tuning**.


## Datasets📂
We evaluate DacTaFie on three datasets:
Ad-buy form: public dataset for visually rich form understanding.
CORD: public receipt dataset.
BANK: private dataset (commercial bank statements, not publicly available due to confidentiality).

## Usage⚙️
1. We provide the required environment dependencies in requirements.txt.
2. Configure your OpenAI client in main.py:
```Python
client = OpenAI(
    api_key='YOUR_API_KEY',
    base_url='YOUR_BASE_URL'
)
```
3. Run the pipeline:
```Python
python main.py
```
## Project Structure✨
```Bash
├─ datasets/                       # Pre-processed Ad-buy example data (full dataset can be downloaded from the official source)
├─ core/
│  ├─ main.py                      # Simple processing pipeline integrating OCR tools, Structural Parsing, and Multi-page Integration plugins
│  ├─ plugins.py                   # Additional plugin implementations, including Missing Field Localization & Completion and Hallucinated Field Filtering
│  ├─ Ad-buy_result_evaluation.py  # Evaluation script for the Ad-buy form dataset
├─ metrics.py                      # Evaluation metrics
├─ README.md
├─ requirements.txt
├─ ch_PP-OCRv4_det_infer.onnx      # OCR model file
```
