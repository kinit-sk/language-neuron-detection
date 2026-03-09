def get_task_lines(tasks): 
    basic_tasks = "" 
    temperature_tasks = "" 

    for task in tasks: 
        if not "*" in task:
            basic_tasks += f"{task}|0," 
        else:
            task = task.replace("*", "")
            temperature_tasks += f"{task}|0," 
    return f'"{basic_tasks.rstrip(",")}"', f'"{temperature_tasks.rstrip(",")}"'

def build_basic_command(cmd, model, task_block, task_dir, max_samples):
    base = (
        f"{cmd} \\\n"
        f'  "model_name={model}" \\\n'
        f"  {task_block} \\\n"
        f"  --custom-tasks {task_dir}"
    )

    if max_samples is not None:
        base += f" \\\n  --max-samples {max_samples}"

    return base
def build_temperature_command(cmd, model, task_block, task_dir, max_samples, params):
    base = (
        f"{cmd} \\\n"
        f'  "model_name={model},generation_parameters={{{params}}}" \\\n'
        f"  {task_block} \\\n"
        f"  --custom-tasks {task_dir}"
    )

    if max_samples is not None:
        base += f" \\\n  --max-samples {max_samples}"

    return base


if __name__=="__main__":
    cmd       = "lighteval accelerate"
    model     = "meta-llama/Llama-3.2-1B,batch_size=16"
    task_dir  = "community_tasks/cze_benczechmark.py"
    params    = '\\"temperature\\":0.7'
    samples   = None
    nohup     = True

    """
    tasks marked with asterisk require temperature parameter
    
    example:
        tasks = [
            "umimeto_qa_biology_clu", <- without temperature
            "propaganda_zamereni_nli*",  <- with temperature
        ]
    """
    if task_dir == "community_tasks/xtreme.py":
        # Full list of xtreme.py tasks
        tasks = [task_name.lower().replace('.', '_') 
                 for task_name in ['MLQA.ar.ar', 'MLQA.ar.de', 'MLQA.ar.en', 'MLQA.ar.es', 'MLQA.ar.hi', 
                        'MLQA.ar.vi', 'MLQA.ar.zh', 'MLQA.de.ar', 'MLQA.de.de', 'MLQA.de.en', 
                        'MLQA.de.es', 'MLQA.de.hi', 'MLQA.de.vi', 'MLQA.de.zh', 'MLQA.en.ar', 
                        'MLQA.en.de', 'MLQA.en.en', 'MLQA.en.es', 'MLQA.en.hi', 'MLQA.en.vi', 
                        'MLQA.en.zh', 'MLQA.es.ar', 'MLQA.es.de', 'MLQA.es.en', 'MLQA.es.es', 
                        'MLQA.es.hi', 'MLQA.es.vi', 'MLQA.es.zh', 'MLQA.hi.ar', 'MLQA.hi.de', 
                        'MLQA.hi.en', 'MLQA.hi.es', 'MLQA.hi.hi', 'MLQA.hi.vi', 'MLQA.hi.zh', 
                        'MLQA.vi.ar', 'MLQA.vi.de', 'MLQA.vi.en', 'MLQA.vi.es', 'MLQA.vi.hi', 
                        'MLQA.vi.vi', 'MLQA.vi.zh', 'MLQA.zh.ar', 'MLQA.zh.de', 'MLQA.zh.en', 
                        'MLQA.zh.es', 'MLQA.zh.hi', 'MLQA.zh.vi', 'MLQA.zh.zh', 'PAWS-X.de', 'PAWS-X.en', 'PAWS-X.es', 'PAWS-X.fr', 'PAWS-X.ja', 'PAWS-X.ko', 'PAWS-X.zh', 'XQuAD.ar', 'XQuAD.de', 'XQuAD.el', 'XQuAD.en', 'XQuAD.es', 'XQuAD.hi',
                         'XQuAD.ru', 'XQuAD.th', 'XQuAD.tr', 'XQuAD.vi', 'XQuAD.zh', 'tatoeba.afr', 'tatoeba.ara', 'tatoeba.ben', 'tatoeba.bul', 'tatoeba.cmn', 'tatoeba.deu', 
              'tatoeba.ell', 'tatoeba.est', 'tatoeba.eus', 'tatoeba.fin', 'tatoeba.fra', 'tatoeba.heb', 
              'tatoeba.hin', 'tatoeba.hun', 'tatoeba.ind', 'tatoeba.ita', 'tatoeba.jav', 'tatoeba.jpn', 
              'tatoeba.kat', 'tatoeba.kaz', 'tatoeba.kor', 'tatoeba.mal', 'tatoeba.mar', 'tatoeba.nld', 
              'tatoeba.pes', 'tatoeba.por', 'tatoeba.rus', 'tatoeba.spa', 'tatoeba.swh', 'tatoeba.tam', 
              'tatoeba.tel', 'tatoeba.tgl', 'tatoeba.tha', 'tatoeba.tur', 'tatoeba.urd', 'tatoeba.vie']]
    elif task_dir == "community_tasks/xnli.py": # force batch size 32 on rtx 3090
        # Full list of xnli.py tasks
        tasks = ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']
    elif task_dir == "community_tasks/belebele.py": # force batch size 16 on rtx 3090
        # Full list of belebele.py tasks
        tasks = ['acm_Arab', 'arz_Arab', 'ceb_Latn', 'fin_Latn', 'hin_Deva', 'ita_Latn', 'khm_Khmr', 'lvs_Latn', 'npi_Deva', 'pol_Latn', 'slv_Latn', 'swe_Latn', 'tso_Latn', 'xho_Latn', 'afr_Latn', 'asm_Beng', 'ces_Latn', 'fra_Latn', 'hin_Latn', 'jav_Latn', 'kin_Latn', 'mal_Mlym', 'npi_Latn', 'por_Latn', 'sna_Latn', 'swh_Latn', 'tur_Latn', 'yor_Latn', 'als_Latn', 'azj_Latn', 'ckb_Arab', 'hrv_Latn', 'jpn_Jpan', 'kir_Cyrl', 'mar_Deva', 'nso_Latn', 'snd_Arab', 'tam_Taml', 'ukr_Cyrl', 'zho_Hans', 'amh_Ethi', 'bam_Latn', 'dan_Latn', 'gaz_Latn', 'hun_Latn', 'kac_Latn', 'kor_Hang', 'mkd_Cyrl', 'nya_Latn', 'ron_Latn', 'som_Latn', 'tel_Telu', 'urd_Arab', 'zho_Hant', 'apc_Arab', 'ben_Beng', 'deu_Latn', 'grn_Latn', 'hye_Armn', 'kan_Knda', 'lao_Laoo', 'mlt_Latn', 'ory_Orya', 'rus_Cyrl', 'sot_Latn', 'tgk_Cyrl', 'urd_Latn', 'zsm_Latn', 'arb_Arab', 'ben_Latn', 'ell_Grek', 'guj_Gujr', 'kat_Geor', 'lin_Latn', 'mri_Latn', 'pan_Guru', 'shn_Mymr', 'spa_Latn', 'tgl_Latn', 'uzn_Latn', 'zul_Latn', 'arb_Latn', 'bod_Tibt', 'eng_Latn', 'hat_Latn', 'ilo_Latn', 'kaz_Cyrl', 'lit_Latn', 'mya_Mymr', 'pbt_Arab', 'sin_Latn', 'srp_Cyrl', 'tha_Thai', 'vie_Latn', 'ars_Arab', 'bul_Cyrl', 'est_Latn', 'hau_Latn', 'ind_Latn', 'kea_Latn', 'lug_Latn', 'nld_Latn', 'pes_Arab', 'sin_Sinh', 'ssw_Latn', 'tir_Ethi', 'war_Latn', 'ary_Arab', 'cat_Latn', 'eus_Latn', 'heb_Hebr', 'isl_Latn', 'khk_Cyrl', 'luo_Latn', 'nob_Latn', 'plt_Latn', 'slk_Latn', 'sun_Latn', 'tsn_Latn', 'wol_Latn']
    elif task_dir == "community_tasks/cze_benczechmark.py":
        # Full list of czechbenchmark.py tasks
        tasks = ["squad_3_2_filtered_rc*", "czechbench_belebele_rc", "propaganda_argumentace_nli", "propaganda_fabulace_nli", "propaganda_nazor_nli", "propaganda_strach_nli", "propaganda_zamereni_nli", "propaganda_demonizace_nli", "propaganda_lokace_nli", "propaganda_relativizace_nli", "propaganda_vina_nli", "propaganda_zanr_nli", "propaganda_emoce_nli", "propaganda_nalepkovani_nli", "propaganda_rusko_nli", "cs_snli_nli", "mall_sentiment_balanced_sent", "fb_sentiment_balanced_sent", "csfd_sentiment_balanced_sent", "czechbench_subjectivity_sent", "cs_gec_clu", "umimeto_qa_biology_clu", "umimeto_qa_chemistry_clu", "umimeto_qa_czech_clu", "umimeto_qa_history_clu", "umimeto_qa_informatics_clu", "umimeto_qa_math_clu", "umimeto_qa_physics_clu", "cermat_czech_tf_clu", "cermat_czech_mc_clu", "czechbench_agree_clu", "cnc_skript12_lm", "cnc_fictree_lm", "cnc_ksk_lm", "cnc_khavlicek_histnews_lm", "cnc_oral_ortofon_lm", "cnc_dialekt_lm"]

    # build command
    basic_task, temperature_task = get_task_lines(tasks)
    basic_cmd = build_basic_command(cmd, model, basic_task, task_dir, samples)
    temperature_cmd = build_temperature_command(cmd, model, temperature_task, task_dir, samples, params) if temperature_task != '""' else ""

    if nohup:
        basic_cmd = f'nohup {basic_cmd} > lighteval.log 2>&1 &'
        temperature_cmd = f'nohup {temperature_cmd} > lighteval.log 2>&1 &'

    print("\nCmd for basic tasks:")
    print("-"*50)
    print(basic_cmd)
    print("\nCmd for tasks that require temperature parameter:")
    print("-"*50)
    print(temperature_cmd)
    print("-"*50)
