import re
import runpy

try:
    from lighteval.metrics.metrics import ExactMatches

    def extract_mcq_answer(text):
        if isinstance(text, dict):
            text = text.get("generated_text", "")

        if isinstance(text, list) and len(text) == 1:
            text = text[0]

        if not isinstance(text, str):
            return text

        text = text.strip().upper()

        match = re.search(r"\b([ABCD])\b", text)
        if match:
            return match.group(1)

        matches = re.findall(r"[ABCD]", text)
        if matches:
            return matches[-1]

        return text

    _original_compute = ExactMatches.compute

    def patched_compute(self, doc, model_response, **kwargs):

        # Extract prediction depending on structure
        pred = None

        # Case 1: common case
        if hasattr(model_response, "text"):
            pred = model_response.text

        # Case 2: HF-like
        elif hasattr(model_response, "generated_text"):
            pred = model_response.generated_text

        # Case 3: fallback
        else:
            pred = str(model_response)

        # Normalize answer
        pred = extract_mcq_answer(pred)


        if hasattr(model_response, "text_post_processed"):
            model_response.text_post_processed = pred
        else:
            model_response.text = pred  # fallback

        # print("PATCH NEW:", pred)  # debug
        return _original_compute(self, doc, model_response, **kwargs)

    print("Patch applied successfully.")
    ExactMatches.compute = patched_compute

except Exception as e:
    print("Patch failed:", e)

# Run lighteval after patch
runpy.run_module('lighteval', run_name='__main__')


# RUN with:
# python misc/patch_mmlu_metric.py accelerate "model_name=meta-llama/Llama-3.2-3B-instruct" "mmlu:high_school_us_history" --max-samples 10