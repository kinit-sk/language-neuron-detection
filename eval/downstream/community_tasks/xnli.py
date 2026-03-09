from lighteval.tasks.requests import Doc
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig

## Helper wheels
_warned = False
def truncate(text, max_chars):
    global _warned
    if not _warned:
        print(f"\033[91m [USER WARNING]\033[93m Using truncated query ({max_chars} chars) to avoid OOM errors. If you want to change this, modify the `truncate()` function in {__file__}.")
        print("Set as high as possible\033[0m")
        _warned = True
    return text[:max_chars]

## --- Multilingual Prompts --- ##
XNLI_PROMPTS = {
    "ar": {
        "premise_label": "المقدمة",
        "hypothesis_label": "الفرضية",
        "instruction": "بالاعتماد فقط على المقدمة، هل الفرضية صحيحة (استلزام)، أم خاطئة (تناقض)، أم غير محددة (حياد)؟",
        "answer_label": "أجب بالخيار الصحيح.",
    },
    "bg": {
        "premise_label": "Предпоставка",
        "hypothesis_label": "Хипотеза",
        "instruction": "Само въз основа на предпоставката, хипотезата вярна ли е (следствие), невярна (противоречие) или неопределена (неутрално)?",
        "answer_label": "Отговорете с правилната опция.",
    },
    "de": {
        "premise_label": "Prämisse",
        "hypothesis_label": "Hypothese",
        "instruction": "Nur auf Grundlage der Prämisse: Ist die Hypothese wahr (Entailment), falsch (Widerspruch) oder unbestimmt (Neutral)?",
        "answer_label": "Antworte mit der richtigen Option.",
    },
    "el": {
        "premise_label": "Προκείμενη",
        "hypothesis_label": "Υπόθεση",
        "instruction": "Με βάση μόνο την προκείμενη, είναι η υπόθεση αληθής (συνεπαγωγή), ψευδής (αντίφαση) ή απροσδιόριστη (ουδέτερη);",
        "answer_label": "Απάντησε με τη σωστή επιλογή.",
    },
    "en": {
        "premise_label": "Premise",
        "hypothesis_label": "Hypothesis",
        "instruction": "Based only on the premise, is the hypothesis true (entailment), false (contradiction), or undetermined (neutral)?",
        "answer_label": "Answer with the correct option.",
    },
    "es": {
        "premise_label": "Premisa",
        "hypothesis_label": "Hipótesis",
        "instruction": "Basándote solo en la premisa, ¿la hipótesis es verdadera (implicación), falsa (contradicción) o indeterminada (neutral)?",
        "answer_label": "Responde con la opción correcta.",
    },
    "fr": {
        "premise_label": "Prémisse",
        "hypothesis_label": "Hypothèse",
        "instruction": "En te basant uniquement sur la prémisse, l’hypothèse est-elle vraie (implication), fausse (contradiction) ou indéterminée (neutre) ?",
        "answer_label": "Réponds avec l’option correcte.",
    },
    "hi": {
        "premise_label": "आधार",
        "hypothesis_label": "परिकल्पना",
        "instruction": "केवल दिए गए आधार के आधार पर, क्या परिकल्पना सही (entailment), गलत (contradiction) या अनिर्धारित (neutral) है?",
        "answer_label": "सही विकल्प के साथ उत्तर दें।",
    },
    "ru": {
        "premise_label": "Посылка",
        "hypothesis_label": "Гипотеза",
        "instruction": "Основываясь только на посылке, является ли гипотеза истинной (следование), ложной (противоречие) или неопределённой (нейтрально)?",
        "answer_label": "Ответьте правильным вариантом.",
    },
    "sw": {
        "premise_label": "Dai",
        "hypothesis_label": "Nadharia",
        "instruction": "Kwa kuzingatia dai pekee, je nadharia ni kweli (inahusiana moja kwa moja), si kweli (inapingana), au haijabainishwa (ya kati)?",
        "answer_label": "Jibu kwa chaguo sahihi.",
    },
    "th": {
        "premise_label": "ข้อสมมติฐานตั้งต้น",
        "hypothesis_label": "สมมติฐาน",
        "instruction": "พิจารณาจากข้อสมมติฐานตั้งต้นเท่านั้น สมมติฐานนี้เป็นจริง (สรุปตามได้), เป็นเท็จ (ขัดแย้ง) หรือไม่สามารถสรุปได้ (เป็นกลาง)?",
        "answer_label": "ตอบด้วยตัวเลือกที่ถูกต้อง",
    },
    "tr": {
        "premise_label": "Öncül",
        "hypothesis_label": "Hipotez",
        "instruction": "Yalnızca öncüle dayanarak, hipotez doğru mu (çıkarım), yanlış mı (çelişki) yoksa belirsiz mi (nötr)?",
        "answer_label": "Doğru seçeneği yazın.",
    },
    "ur": {
        "premise_label": "مقدمہ",
        "hypothesis_label": "مفروضہ",
        "instruction": "صرف مقدمے کی بنیاد پر، کیا مفروضہ درست ہے (لازم آنا)، غلط ہے (تضاد) یا غیر متعین ہے (غیر جانبدار)؟",
        "answer_label": "درست آپشن کے ساتھ جواب دیں۔",
    },
    "vi": {
        "premise_label": "Tiền đề",
        "hypothesis_label": "Giả thuyết",
        "instruction": "Chỉ dựa trên tiền đề, giả thuyết là đúng (suy ra), sai (mâu thuẫn) hay không xác định (trung lập)?",
        "answer_label": "Trả lời bằng lựa chọn đúng.",
    },
    "zh": {
        "premise_label": "前提",
        "hypothesis_label": "假设",
        "instruction": "仅根据前提，假设是真（蕴含）、假（矛盾）还是无法确定（中立）？",
        "answer_label": "请给出正确选项。",
    },
}

## --- Prompt functions --- ##
def xnli_prompt_fn(line: dict, task_name: str):
    max_chars = 8_000

    premise = truncate(line["premise"], max_chars)
    hypothesis = line["hypothesis"]

    choices = ["entailment", "neutral", "contradiction"]

    t = XNLI_PROMPTS[task_name]


    query = (
        f"{t['premise_label']}:\n{premise}\n\n"
        f"{t['hypothesis_label']}:\n{hypothesis}\n\n"
        f"{t['instruction']}\n"
        f"{t['answer_label']}"
    )

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=int(line["label"]),
    )

# Natural Language Inference
class xnlis(LightevalTaskConfig):
    def __init__(
        self,
        name, 
        hf_subset,
    ):
        super().__init__(
            name                = name,
            prompt_function     = xnli_prompt_fn,
            hf_repo             = "facebook/xnli",
            hf_subset           = hf_subset,
            hf_avail_splits     = ["validation", "test"],
            evaluation_splits   = ["test"],
            few_shots_split     = "validation",
            few_shots_select    = "random",
            metrics             = [Metrics.loglikelihood_acc, Metrics.loglikelihood_f1],
            generation_size     = 3,
        )
XNLIS = [
    xnlis(name=subset_name, hf_subset=subset_name)
    for subset_name in ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 
                        'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']
]

## --- Task table --- ##
TASKS_TABLE  = [
    # Natural Language Inference
    *XNLIS,
]
