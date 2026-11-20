# Use the installed local pretrained model

The explicit Ollama route lets Task 4 use the installed Qwen3.5 model through the
same `interpret(request)` and `last_receipt` interface as the grounding adapter.
It is a separate pretrained capacity baseline. It does not load a trained LoRA,
complete Task 5's fine-tuning requirement, or provide the matched Qwen3-VL-8B
pretrained condition for an eventual trained-8B comparison.

No Torch installation, model download, remote service or API key is used by this
adapter. Ollama must already be running locally with the explicitly pinned model
installed. The model tag, installed digest, vision capability and loaded digest
are checked. An unknown tag or changed digest fails without selecting a fallback.

Run from the checkout in the normal reviewer environment:

```text
python -m tools.dropoff_predict predict --backend ollama --requests <REQUESTS.jsonl> --config configs/dropoff/model-ollama-pretrained-9b.json --out <NEW_OUTPUT_DIRECTORY>
```

The unchanged default backend remains the HF Qwen3-VL route. The local route
rejects `--adapter-dir`, `--cache-dir` and `--allow-download`. Its configuration
accepts only the two explicitly supported tags, `qwen3.5:9b` and `qwen3.5:4b`;
switching to the latter requires an explicit configuration with its actual digest
and chosen GPU-layer count. Do not mix these conditions in a quality comparison.

Application code can construct `OllamaGroundingConfig.from_mapping(settings)` and
`OllamaGroundingModel(config)` from `trackrag.dropoff.ollama`. It then calls
`outcome = model.interpret(request)` just as for the original model. The scene,
utterance, dialogue, original-coordinate result and operational-error contracts
remain the same. The application still owns clarification interaction and explicit
confirmation; this model never controls a vehicle or certifies stopping legality.

Every request snapshots and hashes the current original image before provider
calls. The image is resized proportionally within 672 by 384 pixels and encoded as
JPEG. Receipts preserve original and sent-image dimensions/hashes, exact prompt
text, model/runtime identity, token counts, GPU-layer settings and provider output.
Normalized outputs are converted to original-image pixels inside the adapter.
No original annotation, target, source-review metadata or future observation is
put in the prompt. Typed and corrected recognized text use this same route.

The local configuration uses 8,192 context tokens, 192 generated tokens, batch
16 and 14 GPU layers. It reserves 4,096 context tokens beyond a conservative UTF8
text-byte bound for image/template/output processing. This is not an exact
pretokenization check or the same image/token budget as the HF training model.
Ollama performs its own image tokenization. Over-budget text is rejected without
truncation, and a response at the reserved context boundary or with an incomplete
finish is an operational error. Actual token counts remain in the receipt.

Connections are restricted to literal loopback HTTP endpoints. Ambient proxies
are ignored, redirects are refused, response sizes are bounded and calls share a
request deadline, including slowly arriving headers and successful/error bodies.
The loaded context length must be an integer at least as large as requested;
its actual value is retained separately from the requested settings.
Checkpoint/model identity errors, timeouts, memory failures,
provider failures and malformed outputs return `ok=false`, never a semantic
abstention. Prediction artifacts retain the entire input cohort and every failure.

The adapter requests a two-minute keep-alive and does not unload an existing model
on another application's behalf. After an isolated trial, its operator can unload
the selected model with `ollama stop qwen3.5:9b`. Windows may need available system
RAM and paging space in addition to VRAM. Successful local generation does not
establish that the planned 8B training job fits this laptop.

Software tests use generated images and mocked provider replies. Real-scene
execution results must be saved separately; JSON validity and a plausible output
are not reference-scored grounding accuracy.
