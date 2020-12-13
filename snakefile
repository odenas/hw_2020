
#artist_data = "data/input/red_artist_data.csv"
artist_data = "data/input/adata.db"

years = [1951, 1952, 1953, 1954, 1955]
years = [1953]
relations = ["nominated", "year", "champ", "film", "genre", "house", "roles"]
ts = ["ts1", "ts2", "ts3"]
relations = ["genre"]
metrics = ["cosine", "euclidean", "correlation"]
#metrics = ["cosine"]


wildcard_constraints:
    relation="(" + ")|(".join(relations) + ")",
    trel="(" + ")|(".join(ts) + ")"

reports = expand("data/output/rp_{y}_{r}_{m}.csv", y=years, r=relations, m=metrics)
bmats = expand("data/output/bm_{y}_{r}_{m}.pkl", y=years, r=relations, m=metrics)
tie_strengths = expand("data/output/sm_{y}_{r}.pkl", y=years, r=ts, m=metrics)
socio_m = expand("data/output/sm_{y}_{r}.pkl", y=years, r=relations, m=metrics)

rule all:
    input:
        "data/output/report.csv"

rule full_report:
    input:
        s="scripts/merge_reports.py",
        r=reports
    output:
        "data/output/report.csv"
    shell:
        ("python {input.s} --skip_missing `dirname {input.r[0]}` {output}")

rule report:
    input:
        "scripts/report.py",
        "data/output/sm_{year}_{relation}.pkl",
        "data/output/bm_{year}_{relation}_{metric}.pkl",
        "data/output/sm_{year}_ts1.pkl", "data/output/sm_{year}_ts2.pkl", "data/output/sm_{year}_ts3.pkl",
        artist_data
    output:
        "data/output/rp_{year}_{relation}_{metric}.csv"
    shell:
        ("python {input} {output}")

rule block_matrix:
    input:
        "scripts/block_matrix.py",
        "data/output/sm_{year}_{relation}.pkl",
        artist_data
    output:
        "data/output/bm_{year}_{relation}_{metric}.pkl"
    shell:
        ("python {input} {output}")

rule ts_socio_matrix:
    input:
        "scripts/socio_matrix.py",
        artist_data
    output:
        "data/output/sm_{year}_{trel}.pkl"
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
