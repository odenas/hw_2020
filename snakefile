from ghw import pklSave, pklLoad

artist_data = "data/input/adata.db"
report_data = "report.db"

years = [1951, 1952, 1953, 1954, 1955]
years = [1953]
relations = ["nominated", "year", "champ", "film", "genre", "house", "roles"]
ts = ["ts1", "ts2", "ts3"]
#relations = ["genre"]
metrics = ["cosine", "euclidean", "correlation"]
metrics = ["euclidean"]


wildcard_constraints:
    relation="(" + ")|(".join(relations) + ")",
    trel="(" + ")|(".join(ts) + ")",
    nm="(bc)|(tr[1-4])|(shp)"

reports = expand("data/output/rp_{y}_{r}_{m}.csv", y=years, r=relations, m=metrics)
bmats = expand("data/output/bm_{y}_{r}_{m}.pkl", y=years, r=relations, m=metrics)
tie_strengths = expand("data/output/sm_{y}_{r}.pkl", y=years, r=ts, m=metrics)
socio_m = expand("data/output/sm_{y}_{r}.pkl", y=years, r=relations, m=metrics)

rule all:
    input:
        sm=socio_m,
        ts=tie_strengths,
        db=report_data
    output:
        [touch(f + ".imported") for f in socio_m + tie_strengths]
    shell:
         ("python scripts/import_outputs.py {input.sm} {input.ts} {input.db}")


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
        "data/output/cm_{year}_ts1.pkl",
        artist_data
    output:
        "data/output/rp_{year}_{relation}_{metric}.csv"
    shell:
        ("python {input} {output}")

rule collect_network_measures:
    input:
        "data/output/bc_{year}_ts1.pkl",
        "data/output/tr1_{year}_ts1.pkl",
        "data/output/tr2_{year}_ts1.pkl",
        "data/output/tr3_{year}_ts1.pkl",
        "data/output/tr4_{year}_ts1.pkl",
        "data/output/shp_{year}_ts1.pkl"
    output:
        "data/output/cm_{year}_ts1.pkl"
    run:
        pklSave(str(output),
                {os.path.basename(f).split("_")[0]: pklLoad(f)
                 for f in input})

rule make_network_measures:
    input:
        "scripts/centrality.py",
        "data/output/sm_{year}_ts1.pkl",
        artist_data
    output:
        "data/output/{nm}_{year}_ts1.pkl"
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

rule out_db:
    input:
        artist_data,
    output:
        report_data
    shell:
        ("cp {input} {output}")
