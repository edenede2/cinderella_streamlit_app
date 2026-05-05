from __future__ import annotations

import json
import math
import re
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


APP_ROOT = Path(__file__).parent
DATA_ROOT = APP_ROOT / "data"

THERAPEUTIC_CATEGORIES = {
    "None": None,
    "Kinase": "is_kinase",
    "GPCR": "is_gPCR",
    "Ion channel": "is_ion_channel",
    "Nuclear receptor": "is_nuclear_receptor",
    "Transporter": "is_transporter",
    "Enzyme": "is_enzyme",
    "Receptor": "is_receptor",
    "Transcription factor": "is_transcription_factor",
    "Known drug target": "is_known_drug_target",
    "Alzheimer evidence": "has_ad_evidence",
}

MODULE_CATEGORY_COUNT = {
    "is_kinase": "n_kinase",
    "is_gPCR": "n_GPCR",
    "is_ion_channel": "n_ion_channel",
    "is_nuclear_receptor": "n_nuclear_receptor",
    "is_transporter": "n_transporter",
    "is_enzyme": "n_enzyme",
    "is_receptor": "n_receptor",
    "is_transcription_factor": "n_transcription_factor",
    "is_known_drug_target": "n_known_drug_target",
    "has_ad_evidence": "n_Alzheimer_evidence",
}

TISSUE_COLORS = {
    "AC": "#1f77b4",
    "MFBA9BA46": "#2ca02c",
    "PCGBA23": "#ff7f0e",
    "phenotype": "#7b3f9b",
    "module": "#607d8b",
    "highlight": "#c44e52",
    "default": "#78909c",
}


st.set_page_config(page_title="CINDERellA Results", layout="wide")


@st.cache_data(show_spinner=False)
def load_index() -> dict:
    return json.loads((DATA_ROOT / "index.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_manifest(path: str) -> dict:
    return json.loads((DATA_ROOT / path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_tsv(path: str) -> pd.DataFrame:
    return pd.read_csv(DATA_ROOT / path, sep="\t", low_memory=False)


@st.cache_data(show_spinner=False)
def load_gene_annotations(path: str) -> pd.DataFrame:
    df = load_tsv(path)
    bool_cols = [col for col in THERAPEUTIC_CATEGORIES.values() if col and col in df.columns]
    for col in bool_cols:
        df[col] = df[col].fillna(False).astype(bool)
    return df


def clean_phenotype(name: str) -> str:
    return str(name).replace("PHENO_", "")


def module_number(name: str) -> int | None:
    match = re.search(r"(\d+)$", str(name).replace("ME_", ""))
    return int(match.group(1)) if match else None


def gene_base(raw_name: str) -> str:
    gene = str(raw_name).rsplit("_", 1)[0]
    return gene.split(".", 1)[0]


def tissue_from_raw(raw_name: str) -> str:
    text = str(raw_name)
    return text.rsplit("_", 1)[1] if "_" in text and not text.startswith("PHENO_") else ""


def display_gene_label(raw_name: str, gene_ann: pd.DataFrame) -> str:
    base = gene_base(raw_name)
    row = gene_ann.loc[gene_ann["ensembl_gene_id_base"].eq(base)]
    symbol = row["hgnc_symbol"].iloc[0] if len(row) and pd.notna(row["hgnc_symbol"].iloc[0]) else str(raw_name).rsplit("_", 1)[0]
    tissue = tissue_from_raw(raw_name)
    return f"{symbol} ({tissue})" if tissue else str(symbol)


def build_network_figure(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    threshold: float,
    category: str | None,
    gene_ann: pd.DataFrame,
    module_ann: pd.DataFrame,
    title: str,
    max_edges: int = 250,
) -> go.Figure:
    nodes = nodes.copy()
    edges = edges[edges["frequency"] >= threshold].copy()
    if edges.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{title}<br><sup>No edges at threshold {threshold:.3f}</sup>", height=620)
        return fig
    edges = edges.sort_values("frequency", ascending=False).head(max_edges)

    id_to_name = dict(zip(nodes["node_id"].astype(int), nodes["pretty_name"].astype(str)))
    id_to_raw = dict(zip(nodes["node_id"].astype(int), nodes["raw_name"].astype(str)))
    keep_ids = set(edges["source_id"].astype(int)) | set(edges["target_id"].astype(int))
    nodes = nodes[nodes["node_id"].astype(int).isin(keep_ids)].copy()

    graph = nx.DiGraph()
    for _, row in nodes.iterrows():
        node_id = int(row["node_id"])
        raw = str(row["raw_name"])
        pretty = str(row["pretty_name"])
        is_pheno = bool(int(row["is_phenotype"]))
        graph.add_node(node_id, raw=raw, pretty=pretty, is_phenotype=is_pheno)
    for _, row in edges.iterrows():
        src = int(row["source_id"])
        dst = int(row["target_id"])
        if src in graph and dst in graph:
            graph.add_edge(src, dst, weight=float(row["frequency"]))

    if not graph.nodes:
        return go.Figure()

    pos = nx.spring_layout(
        graph,
        seed=42,
        k=2.4 / math.sqrt(max(graph.number_of_nodes(), 2)),
        iterations=120,
        weight="weight",
    )

    edge_x, edge_y, edge_text_x, edge_text_y, edge_text = [], [], [], [], []
    arrow_annotations = []
    max_w = max((graph[u][v]["weight"] for u, v in graph.edges), default=1.0)
    for u, v in graph.edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        w = graph[u][v]["weight"]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        if w >= max(threshold, 0.10):
            edge_text_x.append((x0 + x1) / 2)
            edge_text_y.append((y0 + y1) / 2)
            edge_text.append(f"{w:.3f}")
        arrow_annotations.append(
            dict(
                ax=x0,
                ay=y0,
                x=x1,
                y=y1,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1.1,
                arrowwidth=max(0.5, 1.2 + 3.0 * (w / max_w)),
                arrowcolor="rgba(80, 80, 80, 0.45)",
                opacity=0.55,
            )
        )

    node_x, node_y, labels, hovers, colors, sizes, symbols = [], [], [], [], [], [], []
    for node_id, attrs in graph.nodes(data=True):
        x, y = pos[node_id]
        raw = attrs["raw"]
        pretty = attrs["pretty"]
        is_pheno = attrs["is_phenotype"]
        node_x.append(x)
        node_y.append(y)

        if is_pheno:
            label = clean_phenotype(pretty)
            color = TISSUE_COLORS["phenotype"]
            size = 32
            symbol = "diamond"
            hover = f"<b>{label}</b><br>Phenotype"
        elif pretty.startswith("M") or pretty.startswith("ME_"):
            mod_id = module_number(pretty)
            label = f"M{mod_id}" if mod_id is not None else pretty
            color = TISSUE_COLORS["module"]
            size = 22
            symbol = "circle"
            hover = f"<b>{label}</b><br>Module"
            if mod_id is not None and category:
                row = module_ann[module_ann["cluster_id"].eq(mod_id)]
                count_col = MODULE_CATEGORY_COUNT.get(category)
                if len(row) and count_col in row:
                    count = int(row[count_col].fillna(0).iloc[0])
                    total = int(row.get("unique_genes", pd.Series([0])).fillna(0).iloc[0])
                    hover += f"<br>{count_col}: {count}"
                    if total:
                        hover += f" / {total}"
                    if count > 0:
                        color = TISSUE_COLORS["highlight"]
                        size = 26
        else:
            label = display_gene_label(raw, gene_ann)
            tissue = tissue_from_raw(raw)
            color = TISSUE_COLORS.get(tissue, TISSUE_COLORS["default"])
            size = 18
            symbol = "circle"
            base = gene_base(raw)
            ann_row = gene_ann[gene_ann["ensembl_gene_id_base"].eq(base)]
            hover = f"<b>{label}</b><br>{base}"
            if len(ann_row):
                cats = ann_row["target_categories"].fillna("").iloc[0]
                score = ann_row["therapeutic_target_score"].fillna(0).iloc[0]
                ad = ann_row["open_targets_ad_score"].fillna(0).iloc[0]
                hover += f"<br>Categories: {cats or 'none'}<br>Therapeutic score: {score}<br>Open Targets AD score: {float(ad):.3f}"
                if category and category in ann_row and bool(ann_row[category].iloc[0]):
                    color = TISSUE_COLORS["highlight"]
                    size = 24

        labels.append(label)
        hovers.append(hover)
        colors.append(color)
        sizes.append(size)
        symbols.append(symbol)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1.2, color="rgba(80,80,80,0.28)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    if edge_text:
        fig.add_trace(
            go.Scatter(
                x=edge_text_x,
                y=edge_text_y,
                mode="text",
                text=edge_text,
                textfont=dict(size=10, color="#334155"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=labels,
            textposition="top center",
            textfont=dict(size=11, color="#111827"),
            hovertext=hovers,
            hoverinfo="text",
            marker=dict(
                size=sizes,
                color=colors,
                symbol=symbols,
                line=dict(width=1.5, color="#ffffff"),
            ),
            showlegend=False,
        )
    )
    fig.update_layout(
        title=title,
        height=720,
        margin=dict(l=10, r=10, t=70, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
        annotations=arrow_annotations[:180],
    )
    return fig


def edge_heatmap(nodes: pd.DataFrame, edges: pd.DataFrame, threshold: float, max_nodes: int = 60) -> go.Figure:
    edges = edges[edges["frequency"] >= threshold].copy()
    if edges.empty:
        return go.Figure().update_layout(title="No edges at selected threshold", height=450)
    id_to_name = dict(zip(nodes["node_id"].astype(int), nodes["pretty_name"].astype(str)))
    active = pd.concat([edges["source_id"], edges["target_id"]]).value_counts()
    keep = active.head(max_nodes).index.astype(int).tolist()
    names = [id_to_name.get(node_id, str(node_id)) for node_id in keep]
    matrix = pd.DataFrame(0.0, index=names, columns=names)
    for _, row in edges.iterrows():
        src = int(row["source_id"])
        dst = int(row["target_id"])
        if src in keep and dst in keep:
            matrix.loc[id_to_name.get(src, str(src)), id_to_name.get(dst, str(dst))] = float(row["frequency"])
    fig = px.imshow(matrix, color_continuous_scale="YlOrRd", aspect="auto", labels=dict(color="Edge frequency"))
    fig.update_layout(height=650, margin=dict(l=10, r=10, t=40, b=10), title="Edge-frequency matrix")
    return fig


def driver_bar(edges: pd.DataFrame, nodes: pd.DataFrame, phenotype: str, top_n: int = 25) -> go.Figure:
    pheno_rows = nodes[nodes["pretty_name"].astype(str).eq(phenotype)]
    if pheno_rows.empty:
        pheno_rows = nodes[nodes["raw_name"].astype(str).eq(phenotype)]
    if pheno_rows.empty:
        return go.Figure().update_layout(title="No phenotype node found")
    pheno_id = int(pheno_rows["node_id"].iloc[0])
    id_to_name = dict(zip(nodes["node_id"].astype(int), nodes["pretty_name"].astype(str)))
    sub = edges[edges["target_id"].astype(int).eq(pheno_id)].copy()
    if sub.empty:
        return go.Figure().update_layout(title="No incoming edges to phenotype")
    sub["source"] = sub["source_id"].astype(int).map(id_to_name)
    sub = sub.sort_values("frequency", ascending=False).head(top_n)
    fig = px.bar(sub.iloc[::-1], x="frequency", y="source", orientation="h", labels={"frequency": "Edge frequency", "source": ""})
    fig.update_layout(height=520, title=f"Top incoming drivers to {clean_phenotype(phenotype)}", margin=dict(l=10, r=10, t=55, b=10))
    return fig


def run_module_view(manifest: dict, category_key: str | None, module_ann: pd.DataFrame, gene_ann: pd.DataFrame) -> None:
    runs = manifest["module_runs"]
    run_labels = {clean_phenotype(run["phenotype"]): run for run in runs}
    selected_label = st.selectbox("Phenotype", sorted(run_labels), key="module_pheno")
    run = run_labels[selected_label]
    nodes = load_tsv(run["nodes"])
    edges = load_tsv(run["edges"])

    threshold = st.slider("Edge frequency threshold", 0.0, 1.0, 0.05, 0.005, key="module_threshold")
    max_edges = st.slider("Maximum displayed edges", 25, 500, 180, 25, key="module_max_edges")

    c1, c2, c3 = st.columns(3)
    c1.metric("Nodes", f"{len(nodes):,}")
    c2.metric("Edges", f"{len(edges):,}")
    c3.metric("Edges above threshold", f"{int((edges['frequency'] >= threshold).sum()):,}")

    fig = build_network_figure(
        nodes,
        edges,
        threshold,
        category_key,
        gene_ann,
        module_ann,
        title=f"Module-level BN: {manifest['label']} / {selected_label}",
        max_edges=max_edges,
    )
    st.plotly_chart(fig, use_container_width=True)

    tab1, tab2, tab3 = st.tabs(["Drivers", "Heatmap", "Selection table"])
    with tab1:
        st.plotly_chart(driver_bar(edges, nodes, run["phenotype"], top_n=30), use_container_width=True)
    with tab2:
        st.plotly_chart(edge_heatmap(nodes, edges, threshold), use_container_width=True)
    with tab3:
        table_key = "selected_modules" if "selected_modules" in manifest["tables"] else "all_module_phenotype_correlations"
        if table_key in manifest["tables"]:
            table = load_tsv(manifest["tables"][table_key]).copy()
            if "module_id" in table.columns:
                table = table.merge(module_ann, left_on="module_id", right_on="cluster_id", how="left")
            st.dataframe(table, use_container_width=True, height=500)
        else:
            st.info("No selection table was available in this snapshot.")


def run_gene_view(manifest: dict, category_key: str | None, module_ann: pd.DataFrame, gene_ann: pd.DataFrame) -> None:
    runs = manifest["gene_runs"]
    if not runs:
        st.info("No gene-level BN runs were found in this dataset snapshot.")
        return
    labels = {
        f"{run['run_group']} / {run['module']} / {clean_phenotype(run['phenotype'])}": run
        for run in runs
    }
    selected = st.selectbox("Gene-level run", sorted(labels), key="gene_run")
    run = labels[selected]
    nodes = load_tsv(run["nodes"])
    edges = load_tsv(run["edges"])

    threshold = st.slider("Edge frequency threshold", 0.0, 1.0, 0.05, 0.005, key="gene_threshold")
    max_edges = st.slider("Maximum displayed edges", 25, 500, 180, 25, key="gene_max_edges")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nodes", f"{len(nodes):,}")
    c2.metric("Edges", f"{len(edges):,}")
    c3.metric("Edges above threshold", f"{int((edges['frequency'] >= threshold).sum()):,}")
    c4.metric("Max edge frequency", f"{edges['frequency'].max():.3f}" if len(edges) else "0")

    fig = build_network_figure(
        nodes,
        edges,
        threshold,
        category_key,
        gene_ann,
        module_ann,
        title=f"Gene-level BN: {selected}",
        max_edges=max_edges,
    )
    st.plotly_chart(fig, use_container_width=True)

    tab1, tab2, tab3 = st.tabs(["Phenotype drivers", "Heatmap", "Node annotations"])
    with tab1:
        st.plotly_chart(driver_bar(edges, nodes, run["phenotype"], top_n=30), use_container_width=True)
    with tab2:
        st.plotly_chart(edge_heatmap(nodes, edges, threshold), use_container_width=True)
    with tab3:
        ann_rows = []
        for _, row in nodes[nodes["is_phenotype"].astype(int).eq(0)].iterrows():
            raw = str(row["raw_name"])
            base = gene_base(raw)
            ann = gene_ann[gene_ann["ensembl_gene_id_base"].eq(base)]
            record = {"node": raw, "label": display_gene_label(raw, gene_ann), "gene_base": base, "tissue": tissue_from_raw(raw)}
            if len(ann):
                for col in ["hgnc_symbol", "hgnc_name", "target_categories", "therapeutic_target_score", "open_targets_ad_score"]:
                    if col in ann:
                        record[col] = ann[col].iloc[0]
                for col in THERAPEUTIC_CATEGORIES.values():
                    if col and col in ann:
                        record[col] = bool(ann[col].iloc[0])
            ann_rows.append(record)
        st.dataframe(pd.DataFrame(ann_rows), use_container_width=True, height=500)


def main() -> None:
    if not (DATA_ROOT / "index.json").exists():
        st.error("Missing data/index.json. Run `python scripts/prepare_data.py` first.")
        return

    index = load_index()
    gene_ann = load_gene_annotations(index["target_annotations"]["genes"])
    module_ann = load_tsv(index["target_annotations"]["modules"])
    if "cluster_id" in module_ann.columns:
        module_ann["cluster_id"] = pd.to_numeric(module_ann["cluster_id"], errors="coerce").astype("Int64")

    dataset_options = {item["label"]: item for item in index["datasets"]}
    with st.sidebar:
        st.title("CINDERellA")
        dataset_label = st.selectbox("Results folder", list(dataset_options))
        view = st.radio("Result level", ["Module-level", "Gene-level"], horizontal=False)
        category_label = st.selectbox("Therapeutic annotation", list(THERAPEUTIC_CATEGORIES))
        category_key = THERAPEUTIC_CATEGORIES[category_label]

    manifest = load_manifest(dataset_options[dataset_label]["manifest"])

    st.title("CINDERellA Bayesian Network Results")
    st.caption(manifest["source_path"])

    if view == "Module-level":
        run_module_view(manifest, category_key, module_ann, gene_ann)
    else:
        run_gene_view(manifest, category_key, module_ann, gene_ann)


if __name__ == "__main__":
    main()

