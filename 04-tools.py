# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "mirascope==1.23.3",
#     "pydantic==2.11.4",
#     "python-dotenv==1.1.0",
#     "smartfunc==0.2.0",
# ]
# ///

import marimo

__generated_with = "0.13.6"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _(template):
    from smartfunc import backend
    from pydantic import BaseModel
    from dotenv import load_dotenv
    from pprint import pprint

    load_dotenv(".env")

    class Summary(BaseModel):
        summary: str
        pros: list[str]
        cons: list[str]

    llmify = backend("gpt-4o-mini")

    @llmify
    def generate_poke_desc(text: str) -> Summary:
        t = template("""Describe the following pokemon: {{ text }}""")
        return t.render(text=text)
    

    pprint(generate_poke_desc("pikachu"))
    return (Summary,)


@app.cell
def _(Summary):
    from mirascope import llm


    @llm.call(provider="openai", model="gpt-4o-mini", response_model=Summary)
    def get_poke_summary(pokemon: str) -> str:
        return f"Describe the following pokemon: {pokemon}"

    response = get_poke_summary("Pikachu")
    return (response,)


@app.cell
def _(response):
    response
    return


if __name__ == "__main__":
    app.run()
