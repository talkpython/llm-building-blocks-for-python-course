import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import llm
    from dotenv import load_dotenv

    load_dotenv(".env")
    return llm, mo, pl


@app.cell
def _(pl):
    df = pl.read_csv("spam.csv")
    df.head(200).group_by("label").len()
    return (df,)


@app.cell
async def _():
    import asyncio
    from mosync import async_map_with_retry


    async def delayed_double(x):
        await asyncio.sleep(1)
        return x * 2

    results = await async_map_with_retry(
        range(100), 
        delayed_double, 
        max_concurrency=10, 
        description="Showing a simple demo"
    )
    return (async_map_with_retry,)


@app.cell
def _(llm):
    for model in llm.get_async_models():
        print(model.model_id)
    return


@app.cell
def _(llm):
    from diskcache import Cache

    cache = Cache("accuracy-experiment")

    models = {
        "gpt-4": llm.get_async_model("gpt-4"), 
        "gpt-4o": llm.get_async_model("gpt-4o"), 
    }


    prompt = "is this spam or ham? only reply with spam or ham"
    mod = "gpt-4o"

    async def classify(text, prompt=prompt, model=mod):
        tup = (text, prompt, model)
        if tup in cache: 
            return cache[tup]
        resp = await models[model].prompt(prompt + "\n" + text).json()
        cache[tup] = resp
        return resp
    return classify, prompt


@app.cell
async def _(classify):
    await classify("hello there")
    return


@app.cell
async def _(async_map_with_retry, classify, df):
    n_eval = 200

    llm_results = await async_map_with_retry(
        [_["text"] for _ in df.head(n_eval).to_dicts()], 
        classify, 
        max_concurrency=3, 
        description="Running LLM experiments"
    )
    return llm_results, n_eval


@app.cell
def _(df, llm_results, mo, n_eval, pl, prompt):
    n_correct = pl.DataFrame({**d, "pred": p} for d, p in zip(
        df.head(200).to_dicts(),
        [i.result["content"] for i in llm_results]
    )).filter(pl.col("label") == pl.col("pred")).shape[0]

    mo.md(f"""
    ### Prompt: 
    ```
    {prompt}
    ```
    The accuracy is {n_correct}/{n_eval} = {n_correct/n_eval*100:.1f}%
    """)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Let's jot down some summaries. 

    - "is this spam or ham? only reply with spam or ham" / `gpt-4` `67.0%`
    - "is this spam or ham? only reply with spam or ham" / `gpt-4o` `67.5%`
    - "sometimes we need to deal with spammy text messages, that often promise free/cheap good. is this spam or ham? only reply with spam or ham" / `gpt-4` `66.5%`
    - "sometimes we need to deal with spammy text messages, that often promise free/cheap good. is this spam or ham? only reply with spam or ham" / `gpt-4o` `72.5%`
    """
    )
    return


@app.cell
def _(df):
    df
    return


@app.cell
def _(mo):
    mo.md("""Running this experiment cost me about $2. In fairness: I had to rerun a few things a few times. But at the same time: that's pretty darn expensive for 6 variants on just 200 examples! Especially when you consider you could also build a spaCy/scikit-learn pipeline for this task.""")
    return


@app.cell
def _(df):
    import numpy as np
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import CountVectorizer

    df_valid, df_train = df.head(200), df.tail(200)
    text_valid = df_valid["text"].to_list()
    text_train = df_train["text"].to_list()
    y_valid = df_valid["label"].to_list()
    y_train = df_train["label"].to_list()

    pipe = make_pipeline(CountVectorizer(), LogisticRegression())

    # It's pretty dang accurate
    preds = pipe.fit(text_train, y_train).predict(text_valid)
    np.mean(preds == np.array(y_valid))
    return


if __name__ == "__main__":
    app.run()
