# Test Model Download Guide

This project is wired for the small ONNX sentiment classifier hosted at:

- Repository: `Xenova/distilbert-base-uncased-finetuned-sst-2-english`
- Tokenizer file: `tokenizer.json`
- Model file: `onnx/model_quantized.onnx`

Create the target directory first:

```bash
mkdir -p models/distilbert-sst2
```

Download the tokenizer:

```bash
curl -L "https://huggingface.co/Xenova/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/tokenizer.json?download=true" \
  -o models/distilbert-sst2/tokenizer.json
```

Download the quantized ONNX model:

```bash
curl -L "https://huggingface.co/Xenova/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/onnx/model_quantized.onnx?download=true" \
  -o models/distilbert-sst2/model.onnx
```

The `model_quantized.onnx` file is smaller than the full-precision export, which makes it a better starter model for a tutorial server.
