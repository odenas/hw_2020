
artist_data = "data/input/red_artist_data.csv"
#artist_data = "data/input/formatted_full_artist_data.csv"

years = [1951, 1952, 1953, 1954, 1955]
years = [1953]
relations = ["nominated", "year", "champ", "film", "genre", "prodhouse", "role"]
relations = ["genre", "prodhouse"]
metrics = ["cosine", "euclidean", "correlation"]
metrics = ["cosine"]

rule all:
    input:
        expand("data/output/rp_{y}_{r}_{m}.csv",
               y=years, r=relations, m=metrics),

rule report:
    input:
        "scripts/report.py",
        "data/output/sm_{year}_{relation}.pkl",
        "data/output/bm_{year}_{relation}_{metric}.pkl",
        "data/output/ts_{year}_{relation}.pkl",
        "data/input/blacklist.csv"
    output:
        "data/output/rp_{year}_{relation}_{metric}.csv"
    shell:
        ("python {input} >{output}")

rule block_matrix:
    input:
        "scripts/block_matrix.py",
        "data/output/sm_{year}_{relation}.pkl",
        "data/input/blacklist.csv"
    output:
        "data/output/bm_{year}_{relation}_{metric}.pkl"
    shell:
        ("python {input} {output}")


rule tie_strengths:
    input:
        "scripts/tie_strengths.py",
        artist_data,
        "data/output/sm_{year}_{relation}.pkl"
    output:
        "data/output/ts_{year}_{relation}.pkl"
    shell:
        ("python {input} {output}")

rule socio_matrix:
    input:
        "scripts/socio_matrix.py",
        artist_data
    output:
        "data/output/sm_{year}_{relation}.pkl"
    shell:
        ("python {input} {output}")

rule input_db:
    input:
        adata="data/input/formatted_full_artist_data.csv",
        bdata="data/input/blacklist.csv"
    output:
        "data/input/all_data.db"
    shell:
        ("echo -e '"
        ".mode csv\n"
        ".import {input.adata} adata\n"
        ".import {input.bdata} bdata\n"
        ".quit\n' | sqlite3 {output}")


rule format_adata:
    input:
        "scripts/format_full_artist_data.py", "data/input/full_artist_data.csv"
    output:
        protected("data/input/formatted_full_artist_data.csv")
    shell:
        ("python {input} --outfile {output}")
