
#artist_data = "data/input/red_artist_data.csv"
artist_data = "data/input/adata.db"

years = [1951, 1952, 1953, 1954, 1955]
#years = [1953]
relations = ["nominated", "experience", "champ", "film", "genre", "house", "role",
             "ts1", "ts2", "ts3"]
#relations = ["genre", "prodhouse"]
metrics = ["cosine", "euclidean", "correlation"]
#metrics = ["cosine"]
#

reports = expand("data/output/rp_{y}_{r}_{m}.csv", y=years, r=relations, m=metrics)
tie_strengths = expand("data/output/ts_{y}_{r}.pkl", y=years, r=relations, m=metrics)
socio_m = expand("data/output/sm_{y}_{r}.pkl", y=years, r=relations, m=metrics)

rule all:
    input:
        socio_m

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
        ("python {input} {output}")

rule block_matrix:
    input:
        "scripts/block_matrix.py",
        "data/output/sm_{year}_{relation}.pkl",
        "data/input/blacklist.csv"
    output:
        "data/output/bm_{year}_{relation}_{metric}.pkl"
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
