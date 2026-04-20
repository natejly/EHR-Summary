from summarize_prompt import Summarizer

LLM_OVERALL = LLM(model='gpt-4o', temperature=0.05, top_p=0.05)
BASE = spark.table(BASE_TABLE+ '_extended').toPandas().sample(frac=1, random_state=39)

def which_rules(index, n) -> (dict, int):
    ''' Returns args for build_prompt, %badness '''
    if index %2 == 1:
        return {'anti_rules': 0, 'omit_rules': 0}, 0
    else:
        i = index // 2
        n_omit = i // 10
        n_anti = i % 10
        return {'anti_rules': n_anti, 'omit_rules': n_omit}, round(100 * (n_anti + n_omit) / n)

def overall_summary(pbar, pair, index):
    enc_id, df = pair
    df = df.sort_values('prev_contact_date').iloc[-MAX_PREVIOUS_ENCOUNTERS:]
    dates = [d.strftime("%Y-%m-%d") for d in df['prev_contact_date']]
    authors =  list(df['prev_author'])
    specialties = list(df['prev_specialty'])
    contents = list(df['prev_summary'])
    pids = list(df['prev_pat_enc_csn_id'])
    summarizer = Summarizer(contents, authors, dates, specialties)
    n_rules = len(summarizer.rules)
    args, badness = which_rules(index, n_rules)
    prompt, applied = summarizer.build_prompt(**args)
    result = LLM_OVERALL.evaluate(prompt)
    if not result.ok:
        LOGGER.error(f'LLM ERROR: {result.message}')
    
    row = df.iloc[0]
    result = {
        'pat_enc_csn_id': enc_id,
        'specialty': row['prev_specialty'],
        'contact_date': row['prev_contact_date'],
        'overall_summary': result.message,
        'overall_badness': badness,
        'overall_modifications': json.dumps(applied),
        'prev_authors' : ' • '.join(authors),
        'prev_specialties' : ' • '.join(specialties),
        'prev_enc_csn_ids' : ' • '.join(pids),
    }
    pbar.update(1)
    return result

groups = BASE.groupby('base_pat_enc_csn_id')
with tqdm(total=len(groups)) as pbar, ThreadPool(8) as pool:
    f  = partial(overall_summary, pbar)
    args = list(zip(groups, range(len(groups))))
    rows = pool.starmap(f, args)

result = pd.DataFrame(rows)

spark.sql(f"DROP TABLE IF EXISTS {RESULTS_TABLE}")
spark.createDataFrame(result).write.mode('overwrite').saveAsTable(RESULTS_TABLE)
display(result)