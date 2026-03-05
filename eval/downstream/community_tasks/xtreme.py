import random
from datasets import load_dataset
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
def get_lang_from_subset(subset_name):
    return subset_name.split("_")[-1].lower()
def get_prompt_template(prompts_dict: dict, lang: str, task_name: str = "unknown") -> dict:
    if lang in prompts_dict:
        return prompts_dict[lang]
    print(
        f"\033[91m [USER WARNING]\033[93m Language '{lang}' not found in prompt dict "
        f"for task '{task_name}'. Falling back to English ('en').\033[0m"
    )
    return prompts_dict["en"]
_TATOEBA_POOLS: dict[str, list[str]] = {}
def _build_tatoeba_pool(hf_subset: str) -> list[str]:
    """Load all target sentences for a subset once and cache them."""
    if hf_subset not in _TATOEBA_POOLS:
        dataset = load_dataset("google/xtreme", hf_subset, split="validation")
        _TATOEBA_POOLS[hf_subset] = [ex["target_sentence"] for ex in dataset] # type: ignore
    return _TATOEBA_POOLS[hf_subset]

## --- Multilingual Prompts --- ##
MLQA_PROMPTS = {
    "ar": {
        "instruction":    "اقرأ النص التالي وأجب على السؤال. أعط الإجابة كجزء قصير من النص بالضبط.",
        "text_label":     "النص",
        "question_label": "السؤال",
        "answer_label":   "الإجابة",
    },
    "de": {
        "instruction":    "Lies den folgenden Text und beantworte die Frage. Gib die Antwort als kurze Textspanne exakt aus dem Text wieder.",
        "text_label":     "Text",
        "question_label": "Frage",
        "answer_label":   "Antwort",
    },
    "en": {
        "instruction":    "Read the following text and answer the question. Give the answer as a short span taken exactly from the text.",
        "text_label":     "Text",
        "question_label": "Question",
        "answer_label":   "Answer",
    },
    "es": {
        "instruction":    "Lee el siguiente texto y responde la pregunta. Da la respuesta como un fragmento corto extraído exactamente del texto.",
        "text_label":     "Texto",
        "question_label": "Pregunta",
        "answer_label":   "Respuesta",
    },
    "hi": {
        "instruction":    "निम्नलिखित पाठ पढ़ें और प्रश्न का उत्तर दें। उत्तर को पाठ से हूबहू एक छोटे अंश के रूप में दें।",
        "text_label":     "पाठ",
        "question_label": "प्रश्न",
        "answer_label":   "उत्तर",
    },
    "vi": {
        "instruction":    "Đọc đoạn văn sau và trả lời câu hỏi. Hãy trích dẫn câu trả lời ngắn gọn trực tiếp từ văn bản.",
        "text_label":     "Văn bản",
        "question_label": "Câu hỏi",
        "answer_label":   "Trả lời",
    },
    "zh": {
        "instruction":    "阅读以下文本并回答问题。请从文本中直接摘取简短片段作为答案。",
        "text_label":     "文本",
        "question_label": "问题",
        "answer_label":   "答案",
    },
}
PAWSX_PROMPTS = {
    "de": {
        "instruction":  "Lies die folgenden zwei Sätze und entscheide, ob sie die gleiche Bedeutung haben.\nAntworten Sie nur mit: Ja oder Nein",
        "first_label":  "Erster Satz",
        "second_label": "Zweiter Satz",
        "answer_label": "Antwort",
        "yes":          "Ja",
        "no":           "Nein",
    },
    "en": {
        "instruction":  "Read the following two sentences and decide whether they have the same meaning.\nAnswer only with: Yes or No",
        "first_label":  "First sentence",
        "second_label": "Second sentence",
        "answer_label": "Answer",
        "yes":          "Yes",
        "no":           "No",
    },
    "es": {
        "instruction":  "Lee las siguientes dos oraciones y decide si tienen el mismo significado.\nResponde solo con: Sí o No",
        "first_label":  "Primera oración",
        "second_label": "Segunda oración",
        "answer_label": "Respuesta",
        "yes":          "Sí",
        "no":           "No",
    },
    "fr": {
        "instruction":  "Lis les deux phrases suivantes et décide si elles ont le même sens.\nRéponds uniquement par : Oui ou Non",
        "first_label":  "Première phrase",
        "second_label": "Deuxième phrase",
        "answer_label": "Réponse",
        "yes":          "Oui",
        "no":           "Non",
    },
    "ja": {
        "instruction":  "次の2つの文を読んで、同じ意味かどうかを判断してください。\n「はい」または「いいえ」のみで答えてください。",
        "first_label":  "最初の文",
        "second_label": "2番目の文",
        "answer_label": "回答",
        "yes":          "はい",
        "no":           "いいえ",
    },
    "ko": {
        "instruction":  "다음 두 문장을 읽고 같은 의미인지 판단하세요.\n예 또는 아니오로만 답하세요.",
        "first_label":  "첫 번째 문장",
        "second_label": "두 번째 문장",
        "answer_label": "답변",
        "yes":          "예",
        "no":           "아니오",
    },
    "zh": {
        "instruction":  "阅读以下两个句子，判断它们是否具有相同的含义。\n只需回答：是或否",
        "first_label":  "第一句",
        "second_label": "第二句",
        "answer_label": "答案",
        "yes":          "是",
        "no":           "否",
    },
}
XQUAD_PROMPTS = {
    "ar": {
        "instruction":    "اقرأ النص التالي وأجب على السؤال. اختر إجابتك كجزء قصير مقتبس حرفياً من النص.",
        "text_label":     "النص",
        "question_label": "السؤال",
        "answer_label":   "الإجابة",
    },
    "de": {
        "instruction":    "Lies den folgenden Text und beantworte die Frage. Wähle deine Antwort als kurze Textspanne, die wörtlich aus dem Text stammt.",
        "text_label":     "Text",
        "question_label": "Frage",
        "answer_label":   "Antwort",
    },
    "el": {
        "instruction":    "Διάβασε το παρακάτω κείμενο και απάντησε στην ερώτηση. Δώσε την απάντηση ως σύντομο απόσπασμα αυτούσιο από το κείμενο.",
        "text_label":     "Κείμενο",
        "question_label": "Ερώτηση",
        "answer_label":   "Απάντηση",
    },
    "en": {
        "instruction":    "Read the following text and answer the question. Choose your answer as a short span quoted verbatim from the text.",
        "text_label":     "Text",
        "question_label": "Question",
        "answer_label":   "Answer",
    },
    "es": {
        "instruction":    "Lee el siguiente texto y responde la pregunta. Elige tu respuesta como un fragmento corto citado literalmente del texto.",
        "text_label":     "Texto",
        "question_label": "Pregunta",
        "answer_label":   "Respuesta",
    },
    "hi": {
        "instruction":    "निम्नलिखित पाठ पढ़ें और प्रश्न का उत्तर दें। पाठ से हूबहू उद्धृत एक छोटा अंश चुनें।",
        "text_label":     "पाठ",
        "question_label": "प्रश्न",
        "answer_label":   "उत्तर",
    },
    "ru": {
        "instruction":    "Прочитайте следующий текст и ответьте на вопрос. Выберите ответ в виде короткого фрагмента, дословно взятого из текста.",
        "text_label":     "Текст",
        "question_label": "Вопрос",
        "answer_label":   "Ответ",
    },
    "th": {
        "instruction":    "อ่านข้อความต่อไปนี้และตอบคำถาม โดยเลือกคำตอบเป็นข้อความสั้นๆ ที่คัดลอกมาจากข้อความโดยตรง",
        "text_label":     "ข้อความ",
        "question_label": "คำถาม",
        "answer_label":   "คำตอบ",
    },
    "tr": {
        "instruction":    "Aşağıdaki metni okuyun ve soruyu yanıtlayın. Yanıtınızı metinden birebir alıntılanan kısa bir bölüm olarak seçin.",
        "text_label":     "Metin",
        "question_label": "Soru",
        "answer_label":   "Cevap",
    },
    "vi": {
        "instruction":    "Đọc đoạn văn sau và trả lời câu hỏi. Chọn câu trả lời là một đoạn ngắn trích dẫn nguyên văn từ văn bản.",
        "text_label":     "Văn bản",
        "question_label": "Câu hỏi",
        "answer_label":   "Trả lời",
    },
    "zh": {
        "instruction":    "阅读以下文本并回答问题。请从文本中逐字摘取一个简短片段作为答案。",
        "text_label":     "文本",
        "question_label": "问题",
        "answer_label":   "答案",
    },
}

## --- Prompt functions --- ##
def mlqa_prompt_fn(line: dict, task_name: str):
    max_chars = 8_000

    context = (
        truncate(line["context"], max_chars)
        if len(line["context"]) > max_chars
        else line["context"]
    )

    lang = get_lang_from_subset(task_name)
    t    = get_prompt_template(MLQA_PROMPTS, lang, task_name)

    query = (
        f"{t['instruction']}\n\n"
        f"{t['text_label']}:\n{context}\n\n"
        f"{t['question_label']}:\n{line['question']}\n\n"
        f"{t['answer_label']}: "
    )


    correct_answer = line["answers"]["text"][0].strip()

    return Doc(
        task_name=task_name,
        query=query,
        choices=[correct_answer],
        gold_index=0
    )
def pawsx_prompt_fn(line: dict, task_name: str):
    max_chars = 8_000

    sentence1 = (
        truncate(line["sentence1"], max_chars)
        if len(line["sentence1"]) > max_chars
        else line["sentence1"]
    )
    sentence2 = (
        truncate(line["sentence2"], max_chars)
        if len(line["sentence2"]) > max_chars
        else line["sentence2"]
    )

    lang = get_lang_from_subset(task_name)
    t    = get_prompt_template(PAWSX_PROMPTS, lang, task_name)

    query = (
        f"{t['instruction']}\n\n"
        f"{t['first_label']}:\n{sentence1}\n\n"
        f"{t['second_label']}:\n{sentence2}\n\n"
        f"{t['answer_label']}: "
    )

    choices = [t["no"], t["yes"]]
    gold_index = int(line["label"])

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=gold_index
    )
def xquads_prompt_fn(line: dict, task_name: str):
    max_chars = 8_000

    context = (
        truncate(line["context"], max_chars)
        if len(line["context"]) > max_chars
        else line["context"]
    )

    lang = get_lang_from_subset(task_name)
    t    = get_prompt_template(XQUAD_PROMPTS, lang, task_name)

    query = (
        f"{t['instruction']}\n\n"
        f"{t['text_label']}:\n{context}\n\n"
        f"{t['question_label']}:\n{line['question']}\n\n"
        f"{t['answer_label']}: "
    )

    correct_answer = line["answers"]["text"][0].strip()

    return Doc(
        task_name=task_name,
        query=query,
        choices=[correct_answer],
        gold_index=0
    )
def make_tatoeba_prompt_fn(hf_subset: str, n_distractors: int = 9):
    """
    Factory that returns a prompt_fn closed over the target pool for `hf_subset`.
    
    At call time (per example):
      - sample `n_distractors` wrong answers from the pool
      - insert the correct answer at a random position
      - return a Doc where choices = [distractor, ..., correct, ...]
    
    The query is just the source sentence — lighteval scores each choice
    via loglikelihood and picks the highest (loglikelihood_acc).
    """
    pool = _build_tatoeba_pool(hf_subset)

    def prompt_fn(line: dict, task_name: str) -> Doc:
        correct = line["target_sentence"].strip()

        # Sample distractors — exclude the correct answer to avoid duplicates
        distractor_pool = [s for s in pool if s.strip() != correct]
        k = min(n_distractors, len(distractor_pool))
        distractors = random.sample(distractor_pool, k)

        # Insert correct answer at a random position
        gold_index = random.randint(0, len(distractors))
        choices = distractors[:gold_index] + [correct] + distractors[gold_index:]

        query = (
            "Translate the following sentence:\n\n"
            f"{line['source_sentence'].strip()}\n\n"
            "Choose the correct translation:"
        )

        return Doc(
            task_name=task_name,
            query=query,
            choices=choices,
            gold_index=gold_index,
        )

    return prompt_fn

## --- Task definitions --- ##
## extractive QA (Reading Comprehension) (config_names[:49])
class mlqas(LightevalTaskConfig):
    def __init__(
        self,
        name, 
        hf_subset,
    ):
        super().__init__(
            name                = name,
            prompt_function     = mlqa_prompt_fn,
            hf_repo             = "google/xtreme",
            hf_subset           = hf_subset,
            hf_avail_splits     = ["validation", "test"],
            evaluation_splits   = ["test"],
            few_shots_split     = "validation",
            few_shots_select    = "random",
            metrics             = [Metrics.exact_match, Metrics.f1_score],
            generation_size     = 8,
            stop_sequence       = ["\n"],
        )
MLQAS = [
    mlqas(name=f"{subset_name.lower().replace('.', '_')}", hf_subset=subset_name)
    for subset_name in ['MLQA.ar.ar', 'MLQA.ar.de', 'MLQA.ar.en', 'MLQA.ar.es', 'MLQA.ar.hi', 
                        'MLQA.ar.vi', 'MLQA.ar.zh', 'MLQA.de.ar', 'MLQA.de.de', 'MLQA.de.en', 
                        'MLQA.de.es', 'MLQA.de.hi', 'MLQA.de.vi', 'MLQA.de.zh', 'MLQA.en.ar', 
                        'MLQA.en.de', 'MLQA.en.en', 'MLQA.en.es', 'MLQA.en.hi', 'MLQA.en.vi', 
                        'MLQA.en.zh', 'MLQA.es.ar', 'MLQA.es.de', 'MLQA.es.en', 'MLQA.es.es', 
                        'MLQA.es.hi', 'MLQA.es.vi', 'MLQA.es.zh', 'MLQA.hi.ar', 'MLQA.hi.de', 
                        'MLQA.hi.en', 'MLQA.hi.es', 'MLQA.hi.hi', 'MLQA.hi.vi', 'MLQA.hi.zh', 
                        'MLQA.vi.ar', 'MLQA.vi.de', 'MLQA.vi.en', 'MLQA.vi.es', 'MLQA.vi.hi', 
                        'MLQA.vi.vi', 'MLQA.vi.zh', 'MLQA.zh.ar', 'MLQA.zh.de', 'MLQA.zh.en', 
                        'MLQA.zh.es', 'MLQA.zh.hi', 'MLQA.zh.vi', 'MLQA.zh.zh']
]    

## extractive QA (Reading Comprehension) (config_names[89:96])
class pawsxs(LightevalTaskConfig):
    def __init__(
        self,
        name, 
        hf_subset,
    ):
        super().__init__(
            name                = name,
            prompt_function     = pawsx_prompt_fn,
            hf_repo             = "google/xtreme",
            hf_subset           = hf_subset,
            hf_avail_splits     = ["train","validation", "test"],
            evaluation_splits   = ["test"],
            few_shots_split     = "train",
            few_shots_select    = "random_sampling_from_train",
            metrics             = [Metrics.loglikelihood_acc, Metrics.loglikelihood_f1],
            generation_size     = 8,
            stop_sequence       = ["\n"],
        )
PAWSXS = [
    pawsxs(name=f"{subset_name.lower().replace('.', '_')}", hf_subset=subset_name)
    for subset_name in ['PAWS-X.de', 'PAWS-X.en', 'PAWS-X.es', 'PAWS-X.fr', 'PAWS-X.ja', 'PAWS-X.ko', 'PAWS-X.zh']
]

## sentence pair classification (Natural Language Inference) (config_names[89:96])
class xquads(LightevalTaskConfig):
    def __init__(
        self,
        name, 
        hf_subset,
    ):
        super().__init__(
            name                = name,
            prompt_function     = xquads_prompt_fn,
            hf_repo             = "google/xtreme",
            hf_subset           = hf_subset,
            hf_avail_splits     = ["validation"],
            evaluation_splits   = ["validation"],
            metrics             = [Metrics.exact_match, Metrics.f1_score],
            generation_size     = 8,
            stop_sequence       = ["\n"],
        )
XQUADS = [
    xquads(name=f"{subset_name.lower().replace('.', '_')}", hf_subset=subset_name)
    for subset_name in ['XQuAD.ar', 'XQuAD.de', 'XQuAD.el', 'XQuAD.en', 'XQuAD.es', 'XQuAD.hi',
                         'XQuAD.ru', 'XQuAD.th', 'XQuAD.tr', 'XQuAD.vi', 'XQuAD.zh']
]

## Bitext retrieval (Language Understanding) (config_names[113:149])
class tatoebas(LightevalTaskConfig): 
    def __init__(self, name, hf_subset):
        super().__init__(
            name                = name,
            prompt_function     = make_tatoeba_prompt_fn(hf_subset),
            hf_repo             = "google/xtreme",
            hf_subset           = hf_subset,
            hf_avail_splits     = ["validation"],
            evaluation_splits   = ["validation"],
            few_shots_split     = None,
            few_shots_select    = None,
            metrics             = [Metrics.loglikelihood_acc],
            generation_size     = -1,   # loglikelihood task — no generation
            stop_sequence       = [],
        )

TATOEBAS = [
    tatoebas(name=s.lower().replace(".", "_"), hf_subset=s)
    for s in ['tatoeba.afr', 'tatoeba.ara', 'tatoeba.ben', 'tatoeba.bul', 'tatoeba.cmn', 'tatoeba.deu', 
              'tatoeba.ell', 'tatoeba.est', 'tatoeba.eus', 'tatoeba.fin', 'tatoeba.fra', 'tatoeba.heb', 
              'tatoeba.hin', 'tatoeba.hun', 'tatoeba.ind', 'tatoeba.ita', 'tatoeba.jav', 'tatoeba.jpn', 
              'tatoeba.kat', 'tatoeba.kaz', 'tatoeba.kor', 'tatoeba.mal', 'tatoeba.mar', 'tatoeba.nld', 
              'tatoeba.pes', 'tatoeba.por', 'tatoeba.rus', 'tatoeba.spa', 'tatoeba.swh', 'tatoeba.tam', 
              'tatoeba.tel', 'tatoeba.tgl', 'tatoeba.tha', 'tatoeba.tur', 'tatoeba.urd', 'tatoeba.vie']
]

## --- Task table --- ##
TASKS_TABLE  = [
    # Reading Comprehension
    *MLQAS,
    # Reading Comprehension
    *PAWSXS,
    # Natural Language Inference
    *XQUADS,
    # Language Understanding
    *TATOEBAS,
]


"""
How to get all subsets of a hf repo:
`
from datasets import get_dataset_config_names
config_names = get_dataset_config_names("<org/dataset>")
`
"""