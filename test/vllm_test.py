import os
os.environ["VLLM_USE_FLASHINFER"] = "0"   # turn off FlashInfer sampling
os.environ["VLLM_TORCH_COMPILE"] = "0"    # skip torch.compile fastpath
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # get a useful stack if it still crashes

from vllm import LLM, SamplingParams

# llm = LLM(model="meta-llama/Llama-3.2-1B")
# llm = LLM(model="/root/.cache/huggingface/hub/models--meta-lamma--Llama-3.2-1B/")
# llm = LLM(model="/root/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08/")
llm = LLM(model="/root/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B/snapshots/13afe5124825b4f3751f836b40dafda64c1ed062/")

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


llm.shutdown()
