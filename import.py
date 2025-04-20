import pandas as pd

file_path = "data/babe/train-00000-of-00001.parquet"
df_existing = pd.read_parquet(file_path)

new_articles = [
    {
        "text": "Early voting surges for 2026, but critics warn of looming chaos.",
        "news_link": "https://www.nytimes.com/2025/04/01/us/politics/wisconsin-florida-elections-musk-trump.html",
        "label": 1,
        "outlet": "The New York Times",
        "topic": "elections-2020",
        "type": "center",
        "label_opinion": "Somewhat factual but also opinionated",
        "biased_words": "['surges', 'looming']",
        "uuid": "AbCdEfGhIjKlMnOpQrStUv"
    },
    {
        "text": "Vaccine trials tout success, yet doubts linger over rushed results.",
        "news_link": "https://www.reuters.com/business/healthcare-pharmaceuticals/us-cdc-plans-study-into-vaccines-autism-sources-say-2025-03-07/",
        "label": 1,
        "outlet": "Reuters",
        "topic": "vaccine",
        "type": "center",
        "label_opinion": "Somewhat factual but also opinionated",
        "biased_words": "['tout', 'rushed']",
        "uuid": "BcDeFgHiJkLmNoPqRsTuVw"
    },
    {
        "text": "Tax hikes on crypto spark debate, with investors crying foul.",
        "news_link": "https://www.washingtonpost.com/business/2025/03/20/crypto-tax-hike-debate/",
        "label": 1,
        "outlet": "The Washington Post",
        "topic": "taxes",
        "type": "center",
        "label_opinion": "Expresses writer’s opinion",
        "biased_words": "['spark', 'crying']",
        "uuid": "CdEfGhIjKlMnOpQrStUvWx"
    },
    {
        "text": "Police body cam funds rise, but skeptics question real change.",
        "news_link": "https://www.washingtonpost.com/national-security/2025/02/15/police-body-cam-funding/",
        "label": 1,
        "outlet": "The Washington Post",
        "topic": "black lives matter",
        "type": "center",
        "label_opinion": "Somewhat factual but also opinionated",
        "biased_words": "['rise', 'question']",
        "uuid": "DeFgHiJkLmNoPqRsTuVxWy"
    },
    {
        "text": "Carbon goals tighten, though industries grumble about unfair burdens.",
        "news_link": "https://www.bbc.com/news/science-environment-2025-carbon-goals",
        "label": 1,
        "outlet": "BBC",
        "topic": "environment",
        "type": "center",
        "label_opinion": "Somewhat factual but also opinionated",
        "biased_words": "['tighten', 'grumble']",
        "uuid": "EfGhIjKlMnOpQrStUvWxYz"
    },
    {
        "text": "Gun law push falters as lawmakers dodge bold moves.",
        "news_link": "https://www.nytimes.com/2025/03/25/us/gun-law-stalls-2025",
        "label": 1,
        "outlet": "The New York Times",
        "topic": "gun-control",
        "type": "center",
        "label_opinion": "Expresses writer’s opinion",
        "biased_words": "['falters', 'dodge']",
        "uuid": "FgHiJkLmNoPqRsTuVxWyZa"
    },
    {
        "text": "Border wall talks heat up, with both sides digging in.",
        "news_link": "https://www.reuters.com/world/us/border-wall-debate-2025-03-28/",
        "label": 1,
        "outlet": "Reuters",
        "topic": "immigration",
        "type": "center",
        "label_opinion": "Somewhat factual but also opinionated",
        "biased_words": "['heat', 'digging']",
        "uuid": "GhIjKlMnOpQrStUvWxYzAb"
    },
    {
        "text": "Medicare expansion stalls, fueling calls for drastic action.",
        "news_link": "https://www.washingtonpost.com/health/2025/03/10/medicare-expansion-delay/",
        "label": 1,
        "outlet": "The Washington Post",
        "topic": "universal health care",
        "type": "center",
        "label_opinion": "Expresses writer’s opinion",
        "biased_words": "['stalls', 'drastic']",
        "uuid": "HiJkLmNoPqRsTuVxWyZaBc"
    },
    {
        "text": "Abortion bans stir unease, with support wavering in key states.",
        "news_link": "https://www.bbc.com/news/world-us-canada-abortion-bans-2025",
        "label": 1,
        "outlet": "BBC",
        "topic": "abortion",
        "type": "center",
        "label_opinion": "Somewhat factual but also opinionated",
        "biased_words": "['stir', 'wavering']",
        "uuid": "IjKlMnOpQrStUvWxYzAbCd"
    },
    {
        "text": "Loan forgiveness plan surges, but critics fear a costly mess.",
        "news_link": "https://www.nytimes.com/2025/03/15/education/loan-forgiveness-2025",
        "label": 1,
        "outlet": "The New York Times",
        "topic": "student-debt",
        "type": "center",
        "label_opinion": "Somewhat factual but also opinionated",
        "biased_words": "['surges', 'mess']",
        "uuid": "JkLmNoPqRsTuVxWyZaBcDe"
    },
    {
        "text": "Global trade rules shift, sparking cries of unfairness.",
        "news_link": "https://www.reuters.com/business/global-trade-rules-2025-03-31/",
        "label": 1,
        "outlet": "Reuters",
        "topic": "international-politics-and-world-news",
        "type": "center",
        "label_opinion": "Expresses writer’s opinion",
        "biased_words": "['shift', 'cries']",
        "uuid": "KlMnOpQrStUvWxYzAbCdEf"
    },
    {
        "text": "Tech oversight grows, with users grumbling about overreach.",
        "news_link": "https://www.nytimes.com/2025/03/12/technology/tech-oversight-2025",
        "label": 1,
        "outlet": "The New York Times",
        "topic": "middle-class",
        "type": "center",
        "label_opinion": "Somewhat factual but also opinionated",
        "biased_words": "['grows', 'overreach']",
        "uuid": "LmNoPqRsTuVxWyZaBcDeFg"
    },
    {
        "text": "Pay equity talks stumble, fueling quiet frustration.",
        "news_link": "https://www.bbc.com/news/uk-politics-pay-equity-2025",
        "label": 1,
        "outlet": "BBC",
        "topic": "gender",
        "type": "center",
        "label_opinion": "Somewhat factual but also opinionated",
        "biased_words": "['stumble', 'frustration']",
        "uuid": "MnOpQrStUvWxYzAbCdEfGh"
    },
    {
        "text": "NFL diversity rules spark debate, with fans split on impact.",
        "news_link": "https://www.washingtonpost.com/sports/2025/03/08/nfl-diversity-rules/",
        "label": 1,
        "outlet": "The Washington Post",
        "topic": "sport",
        "type": "center",
        "label_opinion": "Somewhat factual but also opinionated",
        "biased_words": "['spark', 'split']",
        "uuid": "NoPqRsTuVxWyZaBcDeFgHi"
    },
    {
        "text": "Extremist threats loom, though experts question the scale.",
        "news_link": "https://www.reuters.com/world/us/extremist-threats-2025-03-18/",
        "label": 1,
        "outlet": "Reuters",
        "topic": "white-nationalism",
        "type": "center",
        "label_opinion": "Somewhat factual but also opinionated",
        "biased_words": "['loom', 'question']",
        "uuid": "OpQrStUvWxYzAbCdEfGhIj"
    },
    {
        "text": "Same-sex adoption rules shift, stirring cautious hope.",
        "news_link": "https://www.nytimes.com/2025/03/28/us/same-sex-adoption-2025",
        "label": 1,
        "outlet": "The New York Times",
        "topic": "marriage-equality",
        "type": "center",
        "label_opinion": "Somewhat factual but also opinionated",
        "biased_words": "['shift', 'stirring']",
        "uuid": "PqRsTuVxWyZaBcDeFgHiJk"
    },
    {
        "text": "Muslim travel ban talk resurfaces, drawing sharp scrutiny.",
        "news_link": "https://www.bbc.com/news/world-middle-east-travel-ban-2025",
        "label": 1,
        "outlet": "BBC",
        "topic": "islam",
        "type": "center",
        "label_opinion": "Expresses writer’s opinion",
        "biased_words": "['resurfaces', 'sharp']",
        "uuid": "QrStUvWxYzAbCdEfGhIjKl"
    },
    {
        "text": "Virus fears spike, despite rosy vaccine claims.",
        "news_link": "https://www.reuters.com/business/healthcare-pharmaceuticals/measles-cases-european-region-doubled-2024-2025-03-13/",
        "label": 1,
        "outlet": "Reuters",
        "topic": "coronavirus",
        "type": "center",
        "label_opinion": "Somewhat factual but also opinionated",
        "biased_words": "['spike', 'rosy']",
        "uuid": "RsTuVxWyZaBcDeFgHiJkLm"
    },
    {
        "text": "Harassment case reignites, fueling calls for accountability.",
        "news_link": "https://www.washingtonpost.com/nation/2025/03/22/harassment-case-2025/",
        "label": 1,
        "outlet": "The Washington Post",
        "topic": "#metoo",
        "type": "center",
        "label_opinion": "Somewhat factual but also opinionated",
        "biased_words": "['reignites', 'fueling']",
        "uuid": "StUvWxYzAbCdEfGhIjKlMn"
    },
    {
        "text": "Trump policy rollback stalls, with critics scoffing at delays.",
        "news_link": "https://www.nytimes.com/2025/03/31/politics/trump-policy-rollback-2025",
        "label": 1,
        "outlet": "The New York Times",
        "topic": "trump-presidency",
        "type": "center",
        "label_opinion": "Expresses writer’s opinion",
        "biased_words": "['stalls', 'scoffing']",
        "uuid": "TuVxWyZaBcDeFgHiJkLmNo"
    }
]

# Convert to DataFrame
df_new = pd.DataFrame(new_articles)


for col in df_existing.columns:
    if col not in df_new.columns:
        df_new[col] = ""


df_combined = pd.concat([df_existing, df_new], ignore_index=True)


df_combined.to_parquet(file_path, index=False)


print("Original rows:", len(df_existing))
print("New rows:", len(df_new))
print("Combined rows:", len(df_combined))