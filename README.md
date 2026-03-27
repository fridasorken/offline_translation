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
- Evaluation backend: [src/eval_api/](src/eval_api/)  
  - Detailed docs: [src/eval_api/README.md](src/eval_api/README.md)
  - API endpoints: [`app.main`](src/eval_api/app/main.py)
- Frontend UI: [src/frontend/](src/frontend/)  
  - Detailed docs: [src/frontend/README.md](src/frontend/README.md)
- Fine-tuning: [finetune/](finetune/)  
  - Detailed docs: [finetune/README.md](finetune/README.md)
- Integration tests: [tests/test_integration.sh](tests/test_integration.sh)
