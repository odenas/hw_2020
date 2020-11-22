
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
        "data/input/red_artist_data.csv",
        "data/output/sm_{year}_{relation}.pkl"
    output:
        "data/output/ts_{year}_{relation}.pkl"
    shell:
        ("python {input} {output}")

rule socio_matrix:
    input:
        "scripts/socio_matrix.py",
        "data/input/red_artist_data.csv"
    output:
        "data/output/sm_{year}_{relation}.pkl"
    shell:
        ("python {input} {output}")
