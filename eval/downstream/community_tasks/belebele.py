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

## --- Prompt functions --- ##
def belebele_prompt_fn(line: dict, task_name: str):
    max_chars = 8_000

    context = truncate(line["flores_passage"], max_chars)
    question = line["question"]

    choices = [
        line["mc_answer1"],
        line["mc_answer2"],
        line["mc_answer3"],
        line["mc_answer4"],
    ]

    query = (
        f"Passage:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Choose the correct answer based only on the passage.\n"
        "Answer with the correct option."
    )

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=int(line["correct_answer_num"])-1,  # convert to 0-indexed
    )



## --- Task definitions --- ##
class belebeles(LightevalTaskConfig):
    def __init__(
        self,
        name, 
        hf_subset,
    ):
        super().__init__(
            name                = name,
            prompt_function     = belebele_prompt_fn,
            hf_repo             = "facebook/belebele",
            hf_subset           = hf_subset,
            hf_avail_splits     = ["test"],
            evaluation_splits   = ["test"],
            metrics             = [Metrics.loglikelihood_acc, Metrics.loglikelihood_f1],
            generation_size     = 8,
            stop_sequence       = ["\n"],
        )
BELEBELES = [
    belebeles(name=subset_name, hf_subset=subset_name)
    for subset_name in ['acm_Arab', 'arz_Arab', 'ceb_Latn', 'fin_Latn', 'hin_Deva', 'ita_Latn', 'khm_Khmr', 
                        'lvs_Latn', 'npi_Deva', 'pol_Latn', 'slv_Latn', 'swe_Latn', 'tso_Latn', 'xho_Latn', 
                        'afr_Latn', 'asm_Beng', 'ces_Latn', 'fra_Latn', 'hin_Latn', 'jav_Latn', 'kin_Latn', 
                        'mal_Mlym', 'npi_Latn', 'por_Latn', 'sna_Latn', 'swh_Latn', 'tur_Latn', 'yor_Latn', 
                        'als_Latn', 'azj_Latn', 'ckb_Arab', 'fuv_Latn', 'hrv_Latn', 'jpn_Jpan', 'kir_Cyrl', 
                        'mar_Deva', 'nso_Latn', 'snd_Arab', 'tam_Taml', 'ukr_Cyrl', 'zho_Hans', 'amh_Ethi', 
                        'bam_Latn', 'dan_Latn', 'gaz_Latn', 'hun_Latn', 'kac_Latn', 'kor_Hang', 'mkd_Cyrl', 
                        'nya_Latn', 'ron_Latn', 'som_Latn', 'tel_Telu', 'urd_Arab', 'zho_Hant', 'apc_Arab', 
                        'ben_Beng', 'deu_Latn', 'grn_Latn', 'hye_Armn', 'kan_Knda', 'lao_Laoo', 'mlt_Latn', 
                        'ory_Orya', 'rus_Cyrl', 'sot_Latn', 'tgk_Cyrl', 'urd_Latn', 'zsm_Latn', 'arb_Arab', 
                        'ben_Latn', 'ell_Grek', 'guj_Gujr', 'ibo_Latn', 'kat_Geor', 'lin_Latn', 'mri_Latn', 
                        'pan_Guru', 'shn_Mymr', 'spa_Latn', 'tgl_Latn', 'uzn_Latn', 'zul_Latn', 'arb_Latn', 
                        'bod_Tibt', 'eng_Latn', 'hat_Latn', 'ilo_Latn', 'kaz_Cyrl', 'lit_Latn', 'mya_Mymr', 
                        'pbt_Arab', 'sin_Latn', 'srp_Cyrl', 'tha_Thai', 'vie_Latn', 'ars_Arab', 'bul_Cyrl', 
                        'est_Latn', 'hau_Latn', 'ind_Latn', 'kea_Latn', 'lug_Latn', 'nld_Latn', 'pes_Arab', 
                        'sin_Sinh', 'ssw_Latn', 'tir_Ethi', 'war_Latn', 'ary_Arab', 'cat_Latn', 'eus_Latn', 
                        'heb_Hebr', 'isl_Latn', 'khk_Cyrl', 'luo_Latn', 'nob_Latn', 'plt_Latn', 'slk_Latn', 
                        'sun_Latn', 'tsn_Latn', 'wol_Latn']
]

## --- Task table --- ##
TASKS_TABLE  = [
    # Reading Comprehension
    *BELEBELES,
]