# Offline Translation

Monorepo for two related translation services plus tooling:

- **Product API**: lightweight runtime API for application use (initialize once, then translate).
- **Eval API**: model benchmarking/evaluation API with quality + resource metrics.
- **Frontend**: Streamlit UI for translation and evaluation workflows.
- **Finetune pipeline**: scripts for dataset prep and OPUS model training.

## Repository overview

- Product runtime service: [product/](product/)  
  - Detailed docs: [product/README.md](product/README.md)
  - API endpoints: [`app.main`](product/app/main.py)
- Evaluation backend: [evaluation/eval_api/](evaluation/eval_api/)  
  - Detailed docs: [evaluation/eval_api/README.md](evaluation/eval_api/README.md)
  - API endpoints: [`app.main`](evaluation/eval_api/app/main.py)
- Frontend UI: [evaluation/frontend/](evaluation/frontend/)  
  - Detailed docs: [evaluation/frontend/README.md](evaluation/frontend/README.md)
- Fine-tuning: [finetune/](finetune/)  
  - Detailed docs: [finetune/README.md](finetune/README.md)
- Integration tests: [tests/test_integration.sh](tests/test_integration.sh)
