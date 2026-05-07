from __future__ import annotations

import json
import math
import re
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components


APP_ROOT = Path(__file__).parent
DATA_ROOT = APP_ROOT / "data"

CATEGORY_SPECS = [
    ("Kinase", "is_kinase", "n_kinase", "frac_kinase"),
    ("GPCR", "is_gPCR", "n_GPCR", "frac_GPCR"),
    ("Ion channel", "is_ion_channel", "n_ion_channel", "frac_ion_channel"),
    ("Nuclear receptor", "is_nuclear_receptor", "n_nuclear_receptor", "frac_nuclear_receptor"),
    ("Transporter", "is_transporter", "n_transporter", "frac_transporter"),
    ("Enzyme", "is_enzyme", "n_enzyme", "frac_enzyme"),
    ("Receptor", "is_receptor", "n_receptor", "frac_receptor"),
    ("Transcription factor", "is_transcription_factor", "n_transcription_factor", "frac_transcription_factor"),
    ("Known drug target", "is_known_drug_target", "n_known_drug_target", "frac_known_drug_target"),
    ("Alzheimer evidence", "has_ad_evidence", "n_Alzheimer_evidence", "frac_Alzheimer_evidence"),
]

THERAPEUTIC_CATEGORIES = {"None": None, **{label: gene_col for label, gene_col, _, _ in CATEGORY_SPECS}}
CATEGORY_LABEL_BY_GENE = {gene_col: label for label, gene_col, _, _ in CATEGORY_SPECS}
MODULE_COUNT_BY_GENE = {gene_col: count_col for _, gene_col, count_col, _ in CATEGORY_SPECS}
MODULE_FRAC_BY_GENE = {gene_col: frac_col for _, gene_col, _, frac_col in CATEGORY_SPECS}

DATASET_SOURCES = [
    {
        "Dataset": "HGNC Gene Names",
        "Use in app": "Gene symbols, gene names, and approved human-gene identifiers.",
        "Link": "https://www.genenames.org/",
    },
    {
        "Dataset": "Open Targets Platform",
        "Use in app": "Alzheimer disease target evidence scores and evidence-source labels.",
        "Link": "https://platform.opentargets.org/",
    },
    {
        "Dataset": "IUPHAR/BPS Guide to Pharmacology",
        "Use in app": "Drug-target class annotations such as GPCRs, ion channels, nuclear receptors, and transporters.",
        "Link": "https://www.guidetopharmacology.org/",
    },
]

NODE_COLORS = {
    "phenotype": "#7b3f9b",
    "module": "#607d8b",
    "tissue_specific_cluster": "#2563eb",
    "cross_tissue_cluster": "#dc2626",
    "highlight": "#f59e0b",
    "filter": "#111827",
    "default": "#78909c",
}

TISSUE_PALETTE = [
    "#1f77b4",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#bcbd22",
    "#d62728",
    "#7f7f7f",
]


st.set_page_config(page_title="CINDERellA Results", layout="wide")


@st.cache_data(show_spinner=False)
def load_index() -> dict:
    return json.loads((DATA_ROOT / "index.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_manifest(path: str) -> dict:
    return json.loads((DATA_ROOT / path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def _load_tsv(path: str, mtime_ns: int) -> pd.DataFrame:
    return pd.read_csv(DATA_ROOT / path, sep="\t", low_memory=False)


def load_tsv(path: str) -> pd.DataFrame:
    full_path = DATA_ROOT / path
    return _load_tsv(path, full_path.stat().st_mtime_ns)


@st.cache_data(show_spinner=False)
def load_gene_annotations(path: str) -> pd.DataFrame:
    df = load_tsv(path)
    for _, col, _, _ in CATEGORY_SPECS:
        if col in df:
            df[col] = df[col].fillna(False).astype(bool)
    if "open_targets_ad_score" in df:
        df["open_targets_ad_score"] = pd.to_numeric(df["open_targets_ad_score"], errors="coerce").fillna(0.0)
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


def available_tissues(nodes: pd.DataFrame) -> list[str]:
    values = sorted({tissue_from_raw(raw) for raw in nodes["raw_name"].astype(str) if tissue_from_raw(raw)})
    return values


def tissue_color_map(nodes: pd.DataFrame) -> dict[str, str]:
    return {tissue: TISSUE_PALETTE[idx % len(TISSUE_PALETTE)] for idx, tissue in enumerate(available_tissues(nodes))}


def display_gene_label(raw_name: str, gene_ann: pd.DataFrame) -> str:
    base = gene_base(raw_name)
    row = gene_ann.loc[gene_ann["ensembl_gene_id_base"].eq(base)]
    symbol = row["hgnc_symbol"].iloc[0] if len(row) and pd.notna(row["hgnc_symbol"].iloc[0]) else str(raw_name).rsplit("_", 1)[0]
    tissue = tissue_from_raw(raw_name)
    return f"{symbol} ({tissue})" if tissue else str(symbol)


def run_label(run: dict) -> str:
    if "run_group" in run:
        return f"{run['run_group']} / {run['module']} / {clean_phenotype(run['phenotype'])}"
    return clean_phenotype(run["phenotype"])


def combine_runs(run_items: list[dict], label: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    node_records: dict[str, dict] = {}
    edge_records = []

    for run in run_items:
        nodes = load_tsv(run["nodes"])
        edges = load_tsv(run["edges"])
        id_to_node = nodes.set_index(nodes["node_id"].astype(int)).to_dict("index")
        source_label = run_label(run)

        for _, row in nodes.iterrows():
            raw = str(row["raw_name"])
            if raw not in node_records:
                node_records[raw] = {
                    "raw_name": raw,
                    "pretty_name": str(row["pretty_name"]),
                    "is_phenotype": int(row["is_phenotype"]),
                    "source_runs": set(),
                }
            node_records[raw]["source_runs"].add(source_label)

        for _, edge in edges.iterrows():
            src_id = int(edge["source_id"])
            dst_id = int(edge["target_id"])
            src = id_to_node.get(src_id)
            dst = id_to_node.get(dst_id)
            if src is None or dst is None:
                continue
            edge_records.append(
                {
                    "source_raw": str(src["raw_name"]),
                    "target_raw": str(dst["raw_name"]),
                    "source_name": str(src["pretty_name"]),
                    "target_name": str(dst["pretty_name"]),
                    "frequency": float(edge["frequency"]),
                    "source_run": source_label,
                }
            )

    ordered = sorted(node_records, key=lambda raw: (1 - int(node_records[raw]["is_phenotype"]), raw))
    raw_to_id = {raw: idx + 1 for idx, raw in enumerate(ordered)}
    combined_nodes = pd.DataFrame(
        [
            {
                "node_id": raw_to_id[raw],
                "raw_name": raw,
                "pretty_name": node_records[raw]["pretty_name"],
                "is_phenotype": node_records[raw]["is_phenotype"],
                "source_runs": "; ".join(sorted(node_records[raw]["source_runs"])),
            }
            for raw in ordered
        ]
    )

    if edge_records:
        edge_df = pd.DataFrame(edge_records)
        grouped = (
            edge_df.groupby(["source_raw", "target_raw", "source_name", "target_name"], as_index=False)
            .agg(
                frequency=("frequency", "max"),
                mean_frequency=("frequency", "mean"),
                n_runs=("source_run", "nunique"),
                source_runs=("source_run", lambda vals: "; ".join(sorted(set(vals)))),
            )
            .sort_values("frequency", ascending=False)
        )
        grouped["source_id"] = grouped["source_raw"].map(raw_to_id).astype(int)
        grouped["target_id"] = grouped["target_raw"].map(raw_to_id).astype(int)
        combined_edges = grouped[
            [
                "source_id",
                "target_id",
                "frequency",
                "mean_frequency",
                "n_runs",
                "source_name",
                "target_name",
                "source_raw",
                "target_raw",
                "source_runs",
            ]
        ].copy()
    else:
        combined_edges = pd.DataFrame(columns=["source_id", "target_id", "frequency", "source_name", "target_name", "source_raw", "target_raw"])

    meta = {
        "phenotype": label,
        "n_nodes": int(len(combined_nodes)),
        "n_edges": int(len(combined_edges)),
        "max_frequency": float(combined_edges["frequency"].max()) if len(combined_edges) else 0.0,
        "id": label,
    }
    return combined_nodes, combined_edges, meta


def edge_metrics(nodes: pd.DataFrame, edges: pd.DataFrame, threshold: float) -> pd.DataFrame:
    nodes = nodes.copy()
    nodes["node_id"] = nodes["node_id"].astype(int)
    id_to_raw = dict(zip(nodes["node_id"], nodes["raw_name"].astype(str)))
    id_to_pheno = dict(zip(nodes["node_id"], nodes["is_phenotype"].astype(int).eq(1)))
    sub = edges[edges["frequency"] >= threshold].copy()
    rows = []
    for node_id in nodes["node_id"]:
        incoming = sub[sub["target_id"].astype(int).eq(node_id)]
        outgoing = sub[sub["source_id"].astype(int).eq(node_id)]
        in_from_tissues = incoming["source_id"].astype(int).map(id_to_raw).map(tissue_from_raw)
        out_to_tissues = outgoing["target_id"].astype(int).map(id_to_raw).map(tissue_from_raw)
        to_pheno = outgoing["target_id"].astype(int).map(id_to_pheno).fillna(False)
        from_pheno = incoming["source_id"].astype(int).map(id_to_pheno).fillna(False)
        rows.append(
            {
                "node_id": int(node_id),
                "in_edges": int(len(incoming)),
                "out_edges": int(len(outgoing)),
                "in_weight": float(incoming["frequency"].sum()) if len(incoming) else 0.0,
                "out_weight": float(outgoing["frequency"].sum()) if len(outgoing) else 0.0,
                "to_phenotype_edges": int(to_pheno.sum()),
                "from_phenotype_edges": int(from_pheno.sum()),
                "to_phenotype_weight": float(outgoing.loc[to_pheno.values, "frequency"].sum()) if len(outgoing) else 0.0,
                "from_phenotype_weight": float(incoming.loc[from_pheno.values, "frequency"].sum()) if len(incoming) else 0.0,
                "in_from_tissues": in_from_tissues.value_counts().to_dict(),
                "out_to_tissues": out_to_tissues.value_counts().to_dict(),
            }
        )
    metrics = pd.DataFrame(rows).set_index("node_id")
    metrics["to_phenotype_out_fraction"] = metrics.apply(
        lambda row: row["to_phenotype_weight"] / row["out_weight"] if row["out_weight"] else 0.0,
        axis=1,
    )
    metrics["from_phenotype_in_fraction"] = metrics.apply(
        lambda row: row["from_phenotype_weight"] / row["in_weight"] if row["in_weight"] else 0.0,
        axis=1,
    )
    return metrics


def edge_relation_type(src_id: int, dst_id: int, id_to_pheno: dict[int, bool]) -> str:
    src_pheno = bool(id_to_pheno.get(src_id, False))
    dst_pheno = bool(id_to_pheno.get(dst_id, False))
    if src_pheno and dst_pheno:
        return "phenotype -> phenotype"
    if src_pheno:
        return "phenotype -> gene"
    if dst_pheno:
        return "gene -> phenotype"
    return "gene -> gene"


def edge_relationship_label(
    edge: pd.Series,
    id_to_label: dict[int, str],
    id_to_pheno: dict[int, bool],
) -> str:
    src_id = int(edge["source_id"])
    dst_id = int(edge["target_id"])
    source = id_to_label.get(src_id, str(src_id))
    target = id_to_label.get(dst_id, str(dst_id))
    relation = edge_relation_type(src_id, dst_id, id_to_pheno)
    frequency = float(edge["frequency"])
    return f"{source} -> {target} ({relation}; edge frequency {frequency:.3f})"


def node_relationship_summaries(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    threshold: float,
    gene_ann: pd.DataFrame | None,
    top_n: int = 5,
) -> pd.DataFrame:
    nodes = nodes.copy()
    nodes["node_id"] = nodes["node_id"].astype(int)
    id_to_label = {int(row["node_id"]): node_axis_label(row, gene_ann) for _, row in nodes.iterrows()}
    id_to_pheno = dict(zip(nodes["node_id"], nodes["is_phenotype"].astype(int).eq(1)))
    sub = edges[edges["frequency"] >= threshold].copy()

    rows = []
    for node_id in nodes["node_id"]:
        incoming = sub[sub["target_id"].astype(int).eq(node_id)].sort_values("frequency", ascending=False)
        outgoing = sub[sub["source_id"].astype(int).eq(node_id)].sort_values("frequency", ascending=False)
        to_pheno = outgoing[outgoing["target_id"].astype(int).map(id_to_pheno).fillna(False)]
        from_pheno = incoming[incoming["source_id"].astype(int).map(id_to_pheno).fillna(False)]
        rows.append(
            {
                "node_id": int(node_id),
                "relationship_in_edges": int(len(incoming)),
                "relationship_out_edges": int(len(outgoing)),
                "relationship_in_edge_frequency_sum": float(incoming["frequency"].sum()) if len(incoming) else 0.0,
                "relationship_out_edge_frequency_sum": float(outgoing["frequency"].sum()) if len(outgoing) else 0.0,
                "to_phenotype_edge_frequency": float(to_pheno["frequency"].max()) if len(to_pheno) else 0.0,
                "from_phenotype_edge_frequency": float(from_pheno["frequency"].max()) if len(from_pheno) else 0.0,
                "top_outgoing_relationships": "; ".join(
                    edge_relationship_label(edge, id_to_label, id_to_pheno) for _, edge in outgoing.head(top_n).iterrows()
                ),
                "top_incoming_relationships": "; ".join(
                    edge_relationship_label(edge, id_to_label, id_to_pheno) for _, edge in incoming.head(top_n).iterrows()
                ),
                "direct_to_phenotype_relationships": "; ".join(
                    edge_relationship_label(edge, id_to_label, id_to_pheno) for _, edge in to_pheno.head(top_n).iterrows()
                ),
                "direct_from_phenotype_relationships": "; ".join(
                    edge_relationship_label(edge, id_to_label, id_to_pheno) for _, edge in from_pheno.head(top_n).iterrows()
                ),
            }
        )
    return pd.DataFrame(rows).set_index("node_id")


def category_value(row: pd.Series, category: str, ad_threshold: float) -> bool:
    if category == "has_ad_evidence":
        return float(row.get("open_targets_ad_score", 0.0) or 0.0) >= ad_threshold
    return bool(row.get(category, False))


def gene_annotation_row(raw: str, gene_ann: pd.DataFrame) -> pd.Series | None:
    base = gene_base(raw)
    rows = gene_ann[gene_ann["ensembl_gene_id_base"].eq(base)]
    return rows.iloc[0] if len(rows) else None


def module_annotation_row(pretty: str, module_ann: pd.DataFrame) -> pd.Series | None:
    mod_id = module_number(pretty)
    if mod_id is None or "cluster_id" not in module_ann:
        return None
    rows = module_ann[module_ann["cluster_id"].eq(mod_id)]
    return rows.iloc[0] if len(rows) else None


def module_is_tissue_specific(ann: pd.Series | None) -> bool | None:
    if ann is None:
        return None
    if "is_tissue_specific_095" in ann and pd.notna(ann["is_tissue_specific_095"]):
        value = ann["is_tissue_specific_095"]
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes"}
        return bool(value)
    if "max_tissue_fraction" in ann and pd.notna(ann["max_tissue_fraction"]):
        return float(ann["max_tissue_fraction"]) >= 0.95
    tissue_names = []
    if "tissues" in ann and pd.notna(ann.get("tissues")):
        tissue_names = [tissue for tissue in str(ann.get("tissues")).split(";") if tissue]
    tissue_counts = [float(ann.get(col, 0.0) or 0.0) for col in tissue_names if col in ann]
    total = sum(tissue_counts)
    return max(tissue_counts) / total >= 0.95 if total else None


def module_cluster_color(ann: pd.Series | None) -> str:
    is_specific = module_is_tissue_specific(ann)
    if is_specific is None:
        return NODE_COLORS["module"]
    return NODE_COLORS["tissue_specific_cluster"] if is_specific else NODE_COLORS["cross_tissue_cluster"]


def module_cluster_counts(nodes: pd.DataFrame, module_ann: pd.DataFrame) -> dict[str, int]:
    counts = {"Tissue specific": 0, "Cross tissue": 0, "Unknown": 0}
    module_nodes = nodes[
        nodes["is_phenotype"].astype(int).eq(0)
        & nodes["pretty_name"].astype(str).str.startswith(("M", "ME_"))
    ]
    for pretty in module_nodes["pretty_name"].astype(str):
        ann = module_annotation_row(pretty, module_ann)
        is_specific = module_is_tissue_specific(ann)
        if is_specific is None:
            counts["Unknown"] += 1
        elif is_specific:
            counts["Tissue specific"] += 1
        else:
            counts["Cross tissue"] += 1
    return counts


def module_tissue_options(module_ann: pd.DataFrame) -> list[str]:
    if "tissues" not in module_ann:
        return []
    values: set[str] = set()
    for tissue_text in module_ann["tissues"].dropna().astype(str):
        values.update(tissue for tissue in tissue_text.split(";") if tissue)
    return sorted(values)


def node_condition_controls(prefix: str, nodes: pd.DataFrame, is_gene_level: bool, module_ann: pd.DataFrame | None = None) -> dict:
    with st.expander("Node marking and filtering", expanded=False):
        action = st.radio(
            "Condition action",
            ["Show all nodes", "Mark matching nodes", "Filter to matching nodes"],
            horizontal=True,
            key=f"{prefix}_condition_action",
        )
        selected_categories = st.multiselect(
            "Therapeutic categories that must match",
            [label for label, _, _, _ in CATEGORY_SPECS],
            key=f"{prefix}_condition_categories",
        )
        min_in_edges = st.number_input("Minimum incoming edges", min_value=0, value=0, step=1, key=f"{prefix}_min_in")
        min_out_edges = st.number_input("Minimum outgoing edges", min_value=0, value=0, step=1, key=f"{prefix}_min_out")
        relation = st.selectbox(
            "Phenotype relationship",
            ["No phenotype-edge condition", "Parent of phenotype", "Child of phenotype", "Connected to phenotype"],
            key=f"{prefix}_pheno_relation",
        )
        min_pheno_edges = st.number_input("Minimum phenotype-related edges", min_value=0, value=0, step=1, key=f"{prefix}_min_pheno")

        tissues = available_tissues(nodes)
        selected_tissues = []
        incoming_from_tissues = []
        outgoing_to_tissues = []
        if is_gene_level and tissues:
            selected_tissues = st.multiselect("Gene tissues that must match", tissues, key=f"{prefix}_node_tissues")
            incoming_from_tissues = st.multiselect("Require incoming edges from tissues", tissues, key=f"{prefix}_in_tissues")
            outgoing_to_tissues = st.multiselect("Require outgoing edges to tissues", tissues, key=f"{prefix}_out_tissues")

        module_tissues = []
        min_category_fraction = 0.0
        min_to_pheno_fraction = 0.0
        min_from_pheno_fraction = 0.0
        if not is_gene_level:
            module_tissues = st.multiselect("Module contains tissue label", module_tissue_options(module_ann) if module_ann is not None else [], key=f"{prefix}_module_tissues")
            min_category_fraction = st.slider("Minimum selected-category gene fraction in module", 0.0, 1.0, 0.0, 0.01, key=f"{prefix}_min_cat_frac")
            min_to_pheno_fraction = st.slider("Minimum outgoing-weight fraction to phenotype nodes", 0.0, 1.0, 0.0, 0.01, key=f"{prefix}_to_pheno_frac")
            min_from_pheno_fraction = st.slider("Minimum incoming-weight fraction from phenotype nodes", 0.0, 1.0, 0.0, 0.01, key=f"{prefix}_from_pheno_frac")

    return {
        "action": action,
        "categories": [THERAPEUTIC_CATEGORIES[label] for label in selected_categories],
        "selected_tissues": selected_tissues,
        "incoming_from_tissues": incoming_from_tissues,
        "outgoing_to_tissues": outgoing_to_tissues,
        "min_in_edges": int(min_in_edges),
        "min_out_edges": int(min_out_edges),
        "relation": relation,
        "min_pheno_edges": int(min_pheno_edges),
        "module_tissues": module_tissues,
        "min_category_fraction": float(min_category_fraction),
        "min_to_pheno_fraction": float(min_to_pheno_fraction),
        "min_from_pheno_fraction": float(min_from_pheno_fraction),
    }


def condition_matches(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    threshold: float,
    conditions: dict,
    gene_ann: pd.DataFrame,
    module_ann: pd.DataFrame,
    ad_threshold: float,
    is_gene_level: bool,
) -> set[int]:
    if conditions["action"] == "Show all nodes":
        return set()

    metrics = edge_metrics(nodes, edges, threshold)
    matches: set[int] = set()
    for _, row in nodes.iterrows():
        node_id = int(row["node_id"])
        is_pheno = bool(int(row["is_phenotype"]))
        if is_pheno:
            continue

        ok = True
        raw = str(row["raw_name"])
        pretty = str(row["pretty_name"])
        metric = metrics.loc[node_id]

        if metric["in_edges"] < conditions["min_in_edges"] or metric["out_edges"] < conditions["min_out_edges"]:
            ok = False

        if conditions["relation"] != "No phenotype-edge condition":
            if conditions["relation"] == "Parent of phenotype":
                rel_count = metric["to_phenotype_edges"]
            elif conditions["relation"] == "Child of phenotype":
                rel_count = metric["from_phenotype_edges"]
            else:
                rel_count = metric["to_phenotype_edges"] + metric["from_phenotype_edges"]
            if rel_count < max(1, conditions["min_pheno_edges"]):
                ok = False

        if is_gene_level:
            if conditions["selected_tissues"] and tissue_from_raw(raw) not in conditions["selected_tissues"]:
                ok = False
            if conditions["incoming_from_tissues"]:
                in_counts = metric["in_from_tissues"]
                if not any(in_counts.get(tissue, 0) > 0 for tissue in conditions["incoming_from_tissues"]):
                    ok = False
            if conditions["outgoing_to_tissues"]:
                out_counts = metric["out_to_tissues"]
                if not any(out_counts.get(tissue, 0) > 0 for tissue in conditions["outgoing_to_tissues"]):
                    ok = False
            ann = gene_annotation_row(raw, gene_ann)
            if conditions["categories"]:
                if ann is None or not all(category_value(ann, category, ad_threshold) for category in conditions["categories"]):
                    ok = False
        else:
            ann = module_annotation_row(pretty, module_ann)
            if conditions["module_tissues"] and ann is not None:
                tissue_text = str(ann.get("tissues", ""))
                if not any(tissue in tissue_text for tissue in conditions["module_tissues"]):
                    ok = False
            elif conditions["module_tissues"]:
                ok = False
            if conditions["categories"]:
                if ann is None:
                    ok = False
                else:
                    for category in conditions["categories"]:
                        count_col = MODULE_COUNT_BY_GENE.get(category)
                        frac_col = MODULE_FRAC_BY_GENE.get(category)
                        count = float(ann.get(count_col, 0.0) or 0.0)
                        frac = float(ann.get(frac_col, 0.0) or 0.0)
                        if count <= 0 or frac < conditions["min_category_fraction"]:
                            ok = False
            if metric["to_phenotype_out_fraction"] < conditions["min_to_pheno_fraction"]:
                ok = False
            if metric["from_phenotype_in_fraction"] < conditions["min_from_pheno_fraction"]:
                ok = False

        if ok:
            matches.add(node_id)
    return matches


def apply_node_filter(nodes: pd.DataFrame, edges: pd.DataFrame, matching_ids: set[int], action: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if action != "Filter to matching nodes" or not matching_ids:
        return nodes, edges
    phenotype_ids = set(nodes.loc[nodes["is_phenotype"].astype(int).eq(1), "node_id"].astype(int))
    keep = matching_ids | phenotype_ids
    filtered_edges = edges[edges["source_id"].astype(int).isin(keep) & edges["target_id"].astype(int).isin(keep)].copy()
    connected = set(filtered_edges["source_id"].astype(int)) | set(filtered_edges["target_id"].astype(int))
    keep = keep & (connected | matching_ids)
    filtered_nodes = nodes[nodes["node_id"].astype(int).isin(keep)].copy()
    return filtered_nodes, filtered_edges


def category_hover_lines_for_gene(ann: pd.Series, ad_threshold: float) -> list[str]:
    lines = []
    for label, gene_col, _, _ in CATEGORY_SPECS:
        if gene_col == "has_ad_evidence":
            score = float(ann.get("open_targets_ad_score", 0.0) or 0.0)
            value = score >= ad_threshold
            lines.append(f"{label}: {'yes' if value else 'no'}; score {score:.3f}; threshold {ad_threshold:.3f}")
        else:
            lines.append(f"{label}: {'yes' if bool(ann.get(gene_col, False)) else 'no'}")
    return lines


def category_hover_lines_for_module(ann: pd.Series, ad_threshold: float) -> list[str]:
    total = float(ann.get("unique_genes", 0.0) or 0.0)
    lines = []
    for label, gene_col, count_col, frac_col in CATEGORY_SPECS:
        count = float(ann.get(count_col, 0.0) or 0.0)
        frac = float(ann.get(frac_col, count / total if total else 0.0) or 0.0)
        if gene_col == "has_ad_evidence":
            lines.append(f"{label}: {int(count)} genes; fraction {frac:.3f}; gene-score threshold {ad_threshold:.3f}")
        else:
            lines.append(f"{label}: {int(count)} genes; fraction {frac:.3f}")
    return lines


def add_network_legend(
    fig: go.Figure,
    nodes: pd.DataFrame,
    graph: nx.DiGraph,
    category: str | None,
    is_gene_level: bool,
    marked_ids: set[int],
    tissue_colors: dict[str, str],
) -> None:
    if is_gene_level:
        if any(attrs["is_phenotype"] for _, attrs in graph.nodes(data=True)):
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=12, color=NODE_COLORS["phenotype"], symbol="diamond"),
                    name="Phenotype",
                    hoverinfo="skip",
                )
            )
        for tissue in available_tissues(nodes):
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=12, color=tissue_colors.get(tissue, NODE_COLORS["default"])),
                    name=f"Gene tissue: {tissue}",
                    hoverinfo="skip",
                )
            )
    else:
        if any(str(attrs["pretty"]).startswith(("M", "ME_")) for _, attrs in graph.nodes(data=True)):
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=12, color=NODE_COLORS["tissue_specific_cluster"]),
                    name="Tissue specific module",
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=12, color=NODE_COLORS["cross_tissue_cluster"]),
                    name="Cross tissue module",
                    hoverinfo="skip",
                )
            )

    if category:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=12, color="#ffffff", line=dict(width=3, color=NODE_COLORS["highlight"])),
                name=f"Selected annotation: {CATEGORY_LABEL_BY_GENE.get(category, category)}",
                hoverinfo="skip",
            )
        )
    if marked_ids:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=12, color="#ffffff", line=dict(width=3, color=NODE_COLORS["filter"])),
                name="Marked by node filters",
                hoverinfo="skip",
            )
        )


def build_network_figure(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    threshold: float,
    category: str | None,
    gene_ann: pd.DataFrame,
    module_ann: pd.DataFrame,
    title: str,
    ad_threshold: float,
    marked_ids: set[int] | None = None,
    max_edges: int = 250,
    is_gene_level: bool = False,
) -> go.Figure:
    marked_ids = marked_ids or set()
    nodes = nodes.copy()
    edges = edges[edges["frequency"] >= threshold].copy()
    if edges.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{title}<br><sup>No edges at threshold {threshold:.3f}</sup>", height=620)
        return fig
    edges = edges.sort_values("frequency", ascending=False).head(max_edges)

    keep_ids = set(edges["source_id"].astype(int)) | set(edges["target_id"].astype(int))
    nodes = nodes[nodes["node_id"].astype(int).isin(keep_ids)].copy()
    tissue_colors = tissue_color_map(nodes)

    graph = nx.DiGraph()
    for _, row in nodes.iterrows():
        graph.add_node(
            int(row["node_id"]),
            raw=str(row["raw_name"]),
            pretty=str(row["pretty_name"]),
            is_phenotype=bool(int(row["is_phenotype"])),
            source_runs=str(row.get("source_runs", "")),
        )
    for _, row in edges.iterrows():
        src = int(row["source_id"])
        dst = int(row["target_id"])
        if src in graph and dst in graph:
            graph.add_edge(
                src,
                dst,
                weight=float(row["frequency"]),
                mean_weight=float(row.get("mean_frequency", row["frequency"])),
                n_runs=int(row.get("n_runs", 1)),
                source_runs=str(row.get("source_runs", "")),
            )

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

    node_x, node_y, labels, hovers, colors, sizes, symbols, line_widths, line_colors = [], [], [], [], [], [], [], [], []
    metrics = edge_metrics(nodes, edges, threshold)
    for node_id, attrs in graph.nodes(data=True):
        x, y = pos[node_id]
        raw = attrs["raw"]
        pretty = attrs["pretty"]
        is_pheno = attrs["is_phenotype"]
        metric = metrics.loc[node_id] if node_id in metrics.index else None
        category_match = False
        node_x.append(x)
        node_y.append(y)

        if is_pheno:
            label = clean_phenotype(pretty)
            color = NODE_COLORS["phenotype"]
            size = 34
            symbol = "diamond"
            hover = f"<b>{label}</b><br>Phenotype"
        elif pretty.startswith("M") or pretty.startswith("ME_"):
            mod_id = module_number(pretty)
            label = f"M{mod_id}" if mod_id is not None else pretty
            size = 22
            symbol = "circle"
            hover = f"<b>{label}</b><br>Module"
            ann = module_annotation_row(pretty, module_ann)
            color = module_cluster_color(ann)
            if ann is not None:
                total = int(ann.get("unique_genes", 0) or 0)
                hover += f"<br>Unique genes: {total}"
                if "tissues" in ann:
                    hover += f"<br>Tissues: {ann.get('tissues')}"
                if "cluster_tissue_class_095" in ann:
                    hover += f"<br>Cluster class: {ann.get('cluster_tissue_class_095')}"
                if "dominant_tissue" in ann:
                    hover += f"<br>Dominant tissue: {ann.get('dominant_tissue')}"
                if "max_tissue_fraction" in ann and pd.notna(ann.get("max_tissue_fraction")):
                    hover += f"<br>Max tissue fraction: {float(ann.get('max_tissue_fraction')):.3f}"
                hover += "<br>" + "<br>".join(category_hover_lines_for_module(ann, ad_threshold))
                if category:
                    count = float(ann.get(MODULE_COUNT_BY_GENE.get(category), 0.0) or 0.0)
                    if count > 0:
                        category_match = True
                        size = 27
        else:
            label = display_gene_label(raw, gene_ann)
            tissue = tissue_from_raw(raw)
            color = tissue_colors.get(tissue, NODE_COLORS["default"])
            size = 18
            symbol = "circle"
            base = gene_base(raw)
            ann = gene_annotation_row(raw, gene_ann)
            hover = f"<b>{label}</b><br>{base}<br>Tissue: {tissue or 'none'}"
            if ann is not None:
                cats = ann.get("target_categories", "")
                score = ann.get("therapeutic_target_score", 0)
                hover += f"<br>Categories text: {cats or 'none'}<br>Therapeutic target score: {score}"
                hover += "<br>" + "<br>".join(category_hover_lines_for_gene(ann, ad_threshold))
                if category and category_value(ann, category, ad_threshold):
                    category_match = True
                    size = 24

        if metric is not None:
            hover += (
                f"<br>Incoming edges: {int(metric['in_edges'])}; outgoing edges: {int(metric['out_edges'])}"
                f"<br>To phenotype edges: {int(metric['to_phenotype_edges'])}; from phenotype edges: {int(metric['from_phenotype_edges'])}"
            )
        if attrs.get("source_runs"):
            source_label = "Gene source" if not is_pheno and not (pretty.startswith("M") or pretty.startswith("ME_")) else "Source"
            hover += f"<br>{source_label}: {attrs['source_runs']}"
        if node_id in marked_ids:
            hover += "<br><b>Matches selected node conditions</b>"
            size = max(size, 28)

        labels.append(label)
        hovers.append(hover)
        colors.append(color)
        sizes.append(size)
        symbols.append(symbol)
        line_widths.append(3.6 if node_id in marked_ids or category_match else 1.5)
        if node_id in marked_ids:
            line_colors.append(NODE_COLORS["filter"])
        elif category_match:
            line_colors.append(NODE_COLORS["highlight"])
        else:
            line_colors.append("#ffffff")

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
                line=dict(width=line_widths, color=line_colors),
            ),
            showlegend=False,
        )
    )
    add_network_legend(fig, nodes, graph, category, is_gene_level, marked_ids, tissue_colors)
    fig.update_layout(
        title=title,
        height=720,
        margin=dict(l=10, r=10, t=70, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
        annotations=arrow_annotations[:180],
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1.0),
    )
    return fig


def draggable_network_html(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    threshold: float,
    category: str | None,
    gene_ann: pd.DataFrame,
    module_ann: pd.DataFrame,
    ad_threshold: float,
    marked_ids: set[int] | None = None,
    max_edges: int = 250,
) -> str:
    marked_ids = marked_ids or set()
    nodes = nodes.copy()
    edges = edges[edges["frequency"] >= threshold].copy()
    if edges.empty:
        return "<div style='height:680px;display:flex;align-items:center;justify-content:center;font-family:sans-serif;'>No edges at the selected threshold.</div>"

    edges = edges.sort_values("frequency", ascending=False).head(max_edges)
    keep_ids = set(edges["source_id"].astype(int)) | set(edges["target_id"].astype(int))
    nodes = nodes[nodes["node_id"].astype(int).isin(keep_ids)].copy()
    tissue_colors = tissue_color_map(nodes)

    graph = nx.DiGraph()
    for _, row in nodes.iterrows():
        graph.add_node(
            int(row["node_id"]),
            raw=str(row["raw_name"]),
            pretty=str(row["pretty_name"]),
            is_phenotype=bool(int(row["is_phenotype"])),
            source_runs=str(row.get("source_runs", "")),
        )
    for _, row in edges.iterrows():
        src = int(row["source_id"])
        dst = int(row["target_id"])
        if src in graph and dst in graph:
            graph.add_edge(src, dst, weight=float(row["frequency"]), source_runs=str(row.get("source_runs", "")))

    if not graph.nodes:
        return "<div style='height:680px;display:flex;align-items:center;justify-content:center;font-family:sans-serif;'>No connected nodes to display.</div>"

    pos = nx.spring_layout(
        graph,
        seed=42,
        k=2.4 / math.sqrt(max(graph.number_of_nodes(), 2)),
        iterations=120,
        weight="weight",
    )

    vis_nodes = []
    metrics = edge_metrics(nodes, edges, threshold)
    for node_id, attrs in graph.nodes(data=True):
        raw = attrs["raw"]
        pretty = attrs["pretty"]
        is_pheno = attrs["is_phenotype"]
        metric = metrics.loc[node_id] if node_id in metrics.index else None
        category_match = False
        border_color = "#ffffff"

        if is_pheno:
            label = clean_phenotype(pretty)
            color = NODE_COLORS["phenotype"]
            shape = "diamond"
            size = 24
            title = f"<b>{label}</b><br>Phenotype"
        elif pretty.startswith("M") or pretty.startswith("ME_"):
            mod_id = module_number(pretty)
            label = f"M{mod_id}" if mod_id is not None else pretty
            ann = module_annotation_row(pretty, module_ann)
            color = module_cluster_color(ann)
            shape = "dot"
            size = 20
            title = f"<b>{label}</b><br>Module"
            if ann is not None:
                title += f"<br>Unique genes: {int(ann.get('unique_genes', 0) or 0)}"
                if "cluster_tissue_class_095" in ann:
                    title += f"<br>Cluster class: {ann.get('cluster_tissue_class_095')}"
                if "dominant_tissue" in ann:
                    title += f"<br>Dominant tissue: {ann.get('dominant_tissue')}"
                if "max_tissue_fraction" in ann and pd.notna(ann.get("max_tissue_fraction")):
                    title += f"<br>Max tissue fraction: {float(ann.get('max_tissue_fraction')):.3f}"
                title += "<br>" + "<br>".join(category_hover_lines_for_module(ann, ad_threshold))
                if category and float(ann.get(MODULE_COUNT_BY_GENE.get(category), 0.0) or 0.0) > 0:
                    category_match = True
                    size = 24
        else:
            label = display_gene_label(raw, gene_ann)
            tissue = tissue_from_raw(raw)
            color = tissue_colors.get(tissue, NODE_COLORS["default"])
            shape = "dot"
            size = 17
            ann = gene_annotation_row(raw, gene_ann)
            title = f"<b>{label}</b><br>{gene_base(raw)}<br>Tissue: {tissue or 'none'}"
            if ann is not None:
                title += f"<br>Categories text: {ann.get('target_categories', '') or 'none'}"
                title += "<br>" + "<br>".join(category_hover_lines_for_gene(ann, ad_threshold))
                if category and category_value(ann, category, ad_threshold):
                    category_match = True
                    size = 22
        if category_match:
            border_color = NODE_COLORS["highlight"]

        if metric is not None:
            title += (
                f"<br>Incoming edges: {int(metric['in_edges'])}; outgoing edges: {int(metric['out_edges'])}"
                f"<br>To phenotype edges: {int(metric['to_phenotype_edges'])}; from phenotype edges: {int(metric['from_phenotype_edges'])}"
            )
        if attrs.get("source_runs"):
            source_label = "Gene source" if not is_pheno and not (pretty.startswith("M") or pretty.startswith("ME_")) else "Source"
            title += f"<br>{source_label}: {attrs['source_runs']}"
        if node_id in marked_ids:
            title += "<br><b>Matches selected node conditions</b>"
            border_color = NODE_COLORS["filter"]
            size = max(size, 25)

        x, y = pos[node_id]
        vis_nodes.append(
            {
            "id": int(node_id),
            "label": label,
            "title": title,
            "x": float(x * 900),
            "y": float(y * 650),
            "size": size,
            "shape": shape,
            "borderWidth": 3 if category_match or node_id in marked_ids else 1.5,
            "color": {"background": color, "border": border_color, "highlight": {"background": color, "border": "#111827"}},
            "font": {"size": 15 if is_pheno else 13, "face": "Arial", "color": "#111827"},
        }
        )

    max_w = max((graph[u][v]["weight"] for u, v in graph.edges), default=1.0)
    vis_edges = [
        {
            "from": int(u),
            "to": int(v),
            "value": float(graph[u][v]["weight"]),
            "width": 1.0 + 4.0 * float(graph[u][v]["weight"]) / max_w,
            "title": f"Edge frequency: {float(graph[u][v]['weight']):.3f}",
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.7}},
            "color": {"color": "rgba(80,80,80,0.42)", "highlight": "#111827"},
            "smooth": {"type": "continuous"},
        }
        for u, v in graph.edges
    ]

    nodes_json = json.dumps(vis_nodes)
    edges_json = json.dumps(vis_edges)
    return f"""
    <html>
      <head>
        <script src="https://unpkg.com/vis-network@9.1.9/dist/vis-network.min.js"></script>
        <style>
          body {{ margin: 0; font-family: Arial, sans-serif; }}
          #layout {{ display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 10px; }}
          #network {{ height: 700px; border: 1px solid #e5e7eb; border-radius: 6px; background: #ffffff; }}
          #selectedPanel {{
            height: 700px;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            background: #ffffff;
            padding: 12px;
            overflow: auto;
            color: #111827;
            box-sizing: border-box;
          }}
          #selectedPanel h3 {{ margin: 0 0 8px 0; font-size: 15px; }}
          #selectedPanel .empty {{ color: #64748b; font-size: 13px; line-height: 1.4; }}
          #selectedPanel .nodeMeta {{
            border-top: 1px solid #e5e7eb;
            padding-top: 10px;
            margin-top: 10px;
            font-size: 13px;
            line-height: 1.35;
          }}
          #selectedPanel .nodeMeta:first-of-type {{ border-top: 0; padding-top: 0; margin-top: 0; }}
          #toolbar {{ display: flex; gap: 10px; align-items: center; margin: 0 0 8px 0; color: #374151; font-size: 13px; }}
          button {{ border: 1px solid #cbd5e1; background: #f8fafc; border-radius: 6px; padding: 5px 9px; cursor: pointer; }}
          button:hover {{ background: #eef2f7; }}
          @media (max-width: 900px) {{
            #layout {{ grid-template-columns: 1fr; }}
            #selectedPanel {{ height: 260px; }}
          }}
        </style>
      </head>
      <body>
        <div id="toolbar">
          <button onclick="network.fit({{animation: true}})">Fit</button>
          <button onclick="network.stabilize(80)">Re-layout</button>
          <span>Select nodes with click or drag a node to reposition it. Amber marks the selected sidebar annotation.</span>
        </div>
        <div id="layout">
          <div id="network"></div>
          <div id="selectedPanel">
            <h3>Selected node metadata</h3>
            <div id="selectedInfo" class="empty">Select a node in the network to show its metadata here.</div>
          </div>
        </div>
        <script>
          const nodes = new vis.DataSet({nodes_json});
          const edges = new vis.DataSet({edges_json});
          const container = document.getElementById("network");
          const selectedInfo = document.getElementById("selectedInfo");
          const options = {{
            interaction: {{ hover: true, multiselect: true, navigationButtons: true, keyboard: true, dragNodes: true }},
            physics: {{ enabled: false }},
            edges: {{ selectionWidth: 2 }},
            nodes: {{ borderWidth: 1.5 }}
          }};
          const network = new vis.Network(container, {{ nodes, edges }}, options);
          function updateSelectedPanel(selectedIds) {{
            if (!selectedIds || selectedIds.length === 0) {{
              selectedInfo.className = "empty";
              selectedInfo.innerHTML = "Select a node in the network to show its metadata here.";
              return;
            }}
            selectedInfo.className = "";
            selectedInfo.innerHTML = selectedIds.map((id) => {{
              const node = nodes.get(id);
              return `<div class="nodeMeta">${{node.title || node.label || id}}</div>`;
            }}).join("");
          }}
          network.on("selectNode", function(params) {{
            updateSelectedPanel(params.nodes);
          }});
          network.on("deselectNode", function(params) {{
            updateSelectedPanel(params.nodes);
          }});
          network.on("click", function(params) {{
            updateSelectedPanel(params.nodes);
          }});
          network.fit({{ animation: false }});
        </script>
      </body>
    </html>
    """


def edge_heatmap(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    threshold: float,
    max_nodes: int = 60,
    gene_ann: pd.DataFrame | None = None,
) -> go.Figure:
    edges = edges[edges["frequency"] >= threshold].copy()
    if edges.empty:
        return go.Figure().update_layout(title="No edges at selected threshold", height=450)
    id_to_name = {
        int(row["node_id"]): node_axis_label(row, gene_ann)
        for _, row in nodes.iterrows()
    }
    id_to_raw = dict(zip(nodes["node_id"].astype(int), nodes["raw_name"].astype(str)))
    id_to_pheno = dict(zip(nodes["node_id"].astype(int), nodes["is_phenotype"].astype(int).eq(1)))
    active = pd.concat([edges["source_id"], edges["target_id"]]).value_counts()
    keep = active.head(max_nodes).index.astype(int).tolist()
    names = [id_to_name.get(node_id, str(node_id)) for node_id in keep]
    matrix = pd.DataFrame(0.0, index=names, columns=names)
    hover_text = pd.DataFrame("", index=names, columns=names)
    for src_id in keep:
        src_name = id_to_name.get(src_id, str(src_id))
        for dst_id in keep:
            dst_name = id_to_name.get(dst_id, str(dst_id))
            hover_text.loc[src_name, dst_name] = (
                f"<b>{src_name} -> {dst_name}</b><br>"
                f"Relationship: {edge_relation_type(src_id, dst_id, id_to_pheno)}<br>"
                "No edge above selected threshold"
            )
    for _, row in edges.iterrows():
        src = int(row["source_id"])
        dst = int(row["target_id"])
        if src in keep and dst in keep:
            src_name = id_to_name.get(src, str(src))
            dst_name = id_to_name.get(dst, str(dst))
            frequency = float(row["frequency"])
            relation = edge_relation_type(src, dst, id_to_pheno)
            matrix.loc[src_name, dst_name] = frequency
            hover_text.loc[src_name, dst_name] = (
                f"<b>{src_name} -> {dst_name}</b><br>"
                f"Relationship: {relation}<br>"
                f"BN edge frequency: {frequency:.3f}<br>"
                f"Source raw: {id_to_raw.get(src, src)}<br>"
                f"Target raw: {id_to_raw.get(dst, dst)}"
            )
            if "mean_frequency" in row and pd.notna(row.get("mean_frequency")):
                hover_text.loc[src_name, dst_name] += f"<br>Mean frequency across combined runs: {float(row['mean_frequency']):.3f}"
            if "n_runs" in row and pd.notna(row.get("n_runs")):
                hover_text.loc[src_name, dst_name] += f"<br>Runs containing edge: {int(row['n_runs'])}"
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns.tolist(),
            y=matrix.index.tolist(),
            text=hover_text.values,
            hovertemplate="%{text}<extra></extra>",
            colorscale="YlOrRd",
            colorbar=dict(title="Edge frequency"),
            zmin=0,
            zmax=max(1.0, float(matrix.values.max()) if matrix.size else 1.0),
        )
    )
    fig.update_layout(height=650, margin=dict(l=10, r=10, t=40, b=10), title="BN edge-frequency matrix")
    return fig


def node_axis_label(row: pd.Series, gene_ann: pd.DataFrame | None = None) -> str:
    pretty = str(row["pretty_name"])
    raw = str(row["raw_name"])
    if bool(int(row["is_phenotype"])):
        return clean_phenotype(pretty)
    if pretty.startswith("M") or pretty.startswith("ME_"):
        return pretty
    if gene_ann is not None:
        return display_gene_label(raw, gene_ann)
    return pretty


def driver_bar(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    phenotype: str | None = None,
    top_n: int = 25,
    gene_ann: pd.DataFrame | None = None,
) -> go.Figure:
    id_to_name = {
        int(row["node_id"]): node_axis_label(row, gene_ann)
        for _, row in nodes.iterrows()
    }
    phenotype_ids = set(nodes.loc[nodes["is_phenotype"].astype(int).eq(1), "node_id"].astype(int))
    if phenotype:
        pheno_rows = nodes[nodes["pretty_name"].astype(str).eq(phenotype)]
        if pheno_rows.empty:
            pheno_rows = nodes[nodes["raw_name"].astype(str).eq(phenotype)]
        if pheno_rows.empty:
            return go.Figure().update_layout(title="No phenotype node found")
        phenotype_ids = {int(pheno_rows["node_id"].iloc[0])}

    sub = edges[edges["target_id"].astype(int).isin(phenotype_ids)].copy()
    if sub.empty:
        return go.Figure().update_layout(title="No incoming edges to phenotype")
    sub["source"] = sub["source_id"].astype(int).map(id_to_name)
    sub["phenotype"] = sub["target_id"].astype(int).map(id_to_name).map(clean_phenotype)
    sub["driver"] = sub["source"] + " -> " + sub["phenotype"]
    sub = sub.sort_values("frequency", ascending=False).head(top_n)
    fig = px.bar(sub.iloc[::-1], x="frequency", y="driver", orientation="h", labels={"frequency": "Edge frequency", "driver": ""})
    title = "Top incoming drivers to selected phenotype nodes" if phenotype is None else f"Top incoming drivers to {clean_phenotype(phenotype)}"
    fig.update_layout(height=520, title=title, margin=dict(l=10, r=10, t=55, b=10))
    return fig


def prepare_network_for_view(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    threshold: float,
    conditions: dict,
    gene_ann: pd.DataFrame,
    module_ann: pd.DataFrame,
    ad_threshold: float,
    is_gene_level: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, set[int]]:
    matched = condition_matches(nodes, edges, threshold, conditions, gene_ann, module_ann, ad_threshold, is_gene_level)
    filtered_nodes, filtered_edges = apply_node_filter(nodes, edges, matched, conditions["action"])
    if conditions["action"] == "Filter to matching nodes":
        matched = matched & set(filtered_nodes["node_id"].astype(int))
    return filtered_nodes, filtered_edges, matched


def run_module_view(manifest: dict, category_key: str | None, module_ann: pd.DataFrame, gene_ann: pd.DataFrame, ad_threshold: float) -> None:
    runs = manifest["module_runs"]
    labels = {run_label(run): run for run in runs}
    mode = st.radio("Phenotype display", ["Single phenotype", "Phenotype combination"], horizontal=True, key="module_combo_mode")
    if mode == "Single phenotype":
        selected_label = st.selectbox("Phenotype", sorted(labels), key="module_pheno")
        run = labels[selected_label]
        nodes = load_tsv(run["nodes"])
        edges = load_tsv(run["edges"])
        active_label = selected_label
        driver_pheno = run["phenotype"]
    else:
        defaults = sorted(labels)[: min(3, len(labels))]
        selected = st.multiselect("Phenotypes to combine", sorted(labels), default=defaults, key="module_pheno_combo")
        if not selected:
            st.info("Select at least one phenotype to combine.")
            return
        nodes, edges, run = combine_runs([labels[item] for item in selected], "combined_module_phenotypes")
        active_label = " + ".join(selected)
        driver_pheno = None

    threshold = st.slider("Edge frequency threshold", 0.0, 1.0, 0.05, 0.005, key="module_threshold")
    max_edges = st.slider("Maximum displayed edges", 25, 700, 220, 25, key="module_max_edges")
    conditions = node_condition_controls("module", nodes, is_gene_level=False, module_ann=module_ann)
    nodes, edges, marked = prepare_network_for_view(nodes, edges, threshold, conditions, gene_ann, module_ann, ad_threshold, is_gene_level=False)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nodes", f"{len(nodes):,}")
    c2.metric("Edges", f"{len(edges):,}")
    c3.metric("Edges above threshold", f"{int((edges['frequency'] >= threshold).sum()):,}")
    c4.metric("Marked nodes", f"{len(marked):,}")
    ts_ct_counts = module_cluster_counts(nodes, module_ann)
    st.caption(
        f"Module colors: blue = tissue specific (single-tissue fraction >= 0.95; n={ts_ct_counts['Tissue specific']}), "
        f"red = cross tissue (n={ts_ct_counts['Cross tissue']}). "
        f"Amber outline = selected sidebar annotation. Unknown annotation: {ts_ct_counts['Unknown']}."
    )

    renderer = st.radio("Network renderer", ["Plotly", "Draggable nodes"], horizontal=True, key="module_renderer")
    if renderer == "Draggable nodes":
        components.html(
            draggable_network_html(
                nodes,
                edges,
                threshold,
                category_key,
                gene_ann,
                module_ann,
                ad_threshold=ad_threshold,
                marked_ids=marked,
                max_edges=max_edges,
            ),
            height=750,
            scrolling=False,
        )
    else:
        fig = build_network_figure(
            nodes,
            edges,
            threshold,
            category_key,
            gene_ann,
            module_ann,
            title=f"Module-level BN: {manifest['label']} / {active_label}",
            ad_threshold=ad_threshold,
            marked_ids=marked,
            max_edges=max_edges,
            is_gene_level=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    tab1, tab2, tab3 = st.tabs(["Drivers", "Heatmap", "Selection table"])
    with tab1:
        st.plotly_chart(driver_bar(edges, nodes, driver_pheno, top_n=35), use_container_width=True)
    with tab2:
        st.plotly_chart(edge_heatmap(nodes, edges, threshold, gene_ann=gene_ann), use_container_width=True)
    with tab3:
        table_key = "selected_modules" if "selected_modules" in manifest["tables"] else "all_module_phenotype_correlations"
        if table_key in manifest["tables"]:
            table = load_tsv(manifest["tables"][table_key]).copy()
            if "module_id" in table.columns:
                table = table.merge(module_ann, left_on="module_id", right_on="cluster_id", how="left")
            st.dataframe(table, use_container_width=True, height=500)
        else:
            st.info("No selection table was available in this snapshot.")


def run_gene_view(manifest: dict, category_key: str | None, module_ann: pd.DataFrame, gene_ann: pd.DataFrame, ad_threshold: float) -> None:
    runs = manifest["gene_runs"]
    if not runs:
        st.info("No gene-level BN runs were found in this dataset snapshot.")
        return
    labels = {run_label(run): run for run in runs}
    mode = st.radio("Gene-run display", ["Single run", "Run combination"], horizontal=True, key="gene_combo_mode")
    if mode == "Single run":
        selected = st.selectbox("Gene-level run", sorted(labels), key="gene_run")
        run = labels[selected]
        nodes = load_tsv(run["nodes"])
        gene_source = f"{run.get('run_group', '')} / {run.get('module', '')} / {clean_phenotype(run.get('phenotype', ''))}".strip(" /")
        nodes["source_runs"] = ""
        nodes.loc[nodes["is_phenotype"].astype(int).eq(0), "source_runs"] = gene_source
        edges = load_tsv(run["edges"])
        active_label = selected
        driver_pheno = run["phenotype"]
    else:
        defaults = sorted(labels)[: min(3, len(labels))]
        selected_runs = st.multiselect("Gene-level runs to combine", sorted(labels), default=defaults, key="gene_run_combo")
        if not selected_runs:
            st.info("Select at least one gene-level run to combine.")
            return
        nodes, edges, run = combine_runs([labels[item] for item in selected_runs], "combined_gene_runs")
        active_label = " + ".join(selected_runs)
        driver_pheno = None

    threshold = st.slider("Edge frequency threshold", 0.0, 1.0, 0.05, 0.005, key="gene_threshold")
    max_edges = st.slider("Maximum displayed edges", 25, 700, 220, 25, key="gene_max_edges")
    conditions = node_condition_controls("gene", nodes, is_gene_level=True)
    nodes, edges, marked = prepare_network_for_view(nodes, edges, threshold, conditions, gene_ann, module_ann, ad_threshold, is_gene_level=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nodes", f"{len(nodes):,}")
    c2.metric("Edges", f"{len(edges):,}")
    c3.metric("Edges above threshold", f"{int((edges['frequency'] >= threshold).sum()):,}")
    c4.metric("Marked nodes", f"{len(marked):,}")
    tissue_colors = tissue_color_map(nodes)
    tissue_text = ", ".join(f"{tissue} = {color}" for tissue, color in tissue_colors.items()) or "none"
    st.caption(
        f"Gene colors: phenotype = purple diamond; tissues: {tissue_text}. "
        "Amber outline = selected sidebar annotation; black outline = marked by node filters."
    )

    renderer = st.radio("Network renderer", ["Plotly", "Draggable nodes"], horizontal=True, key="gene_renderer")
    if renderer == "Draggable nodes":
        components.html(
            draggable_network_html(
                nodes,
                edges,
                threshold,
                category_key,
                gene_ann,
                module_ann,
                ad_threshold=ad_threshold,
                marked_ids=marked,
                max_edges=max_edges,
            ),
            height=750,
            scrolling=False,
        )
    else:
        fig = build_network_figure(
            nodes,
            edges,
            threshold,
            category_key,
            gene_ann,
            module_ann,
            title=f"Gene-level BN: {active_label}",
            ad_threshold=ad_threshold,
            marked_ids=marked,
            max_edges=max_edges,
            is_gene_level=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    tab1, tab2, tab3 = st.tabs(["Phenotype drivers", "Heatmap", "Node annotations"])
    with tab1:
        st.plotly_chart(driver_bar(edges, nodes, driver_pheno, top_n=35, gene_ann=gene_ann), use_container_width=True)
    with tab2:
        st.caption("Cells show BN edge frequency between displayed nodes. This snapshot does not include raw pairwise gene-expression correlations.")
        st.plotly_chart(edge_heatmap(nodes, edges, threshold, gene_ann=gene_ann), use_container_width=True)
    with tab3:
        relationship_summary = node_relationship_summaries(nodes, edges, threshold, gene_ann)
        ann_rows = []
        for _, row in nodes[nodes["is_phenotype"].astype(int).eq(0)].iterrows():
            raw = str(row["raw_name"])
            base = gene_base(raw)
            ann = gene_annotation_row(raw, gene_ann)
            node_id = int(row["node_id"])
            record = {"node": raw, "label": display_gene_label(raw, gene_ann), "gene_base": base, "tissue": tissue_from_raw(raw)}
            if node_id in relationship_summary.index:
                record.update(relationship_summary.loc[node_id].to_dict())
            if ann is not None:
                for col in ["hgnc_symbol", "hgnc_name", "target_categories", "therapeutic_target_score", "open_targets_ad_score"]:
                    if col in ann:
                        record[col] = ann[col]
                for _, col, _, _ in CATEGORY_SPECS:
                    if col in ann:
                        record[col] = category_value(ann, col, ad_threshold)
            ann_rows.append(record)
        st.dataframe(pd.DataFrame(ann_rows), use_container_width=True, height=500)


def render_source_info() -> None:
    with st.expander("Annotation dataset sources", expanded=False):
        st.markdown(
            "Therapeutic annotations in this app are copied from the local `target_annotations` folder into the project data snapshot. "
            "The compact tables used by the app are `gene_target_annotations_compact.tsv` and `module_target_summary_level4.tsv`."
        )
        source_df = pd.DataFrame(DATASET_SOURCES)
        st.dataframe(
            source_df,
            use_container_width=True,
            hide_index=True,
            column_config={"Link": st.column_config.LinkColumn("Link")},
        )
        st.markdown(
            "Alzheimer evidence is thresholded interactively from `open_targets_ad_score`. "
            "The raw `has_ad_evidence` flag in the snapshot corresponds to genes with nonzero Open Targets AD evidence, "
            "but the app can mark/filter genes using a stricter score cutoff."
        )


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
        category_label = st.selectbox("Therapeutic annotation to color", list(THERAPEUTIC_CATEGORIES))
        category_key = THERAPEUTIC_CATEGORIES[category_label]
        ad_threshold = st.slider("Open Targets AD score threshold", 0.0, 1.0, 0.05, 0.005)

    manifest = load_manifest(dataset_options[dataset_label]["manifest"])

    st.title("CINDERellA Bayesian Network Results")
    st.caption(manifest["source_path"])
    render_source_info()

    if view == "Module-level":
        run_module_view(manifest, category_key, module_ann, gene_ann, ad_threshold)
    else:
        run_gene_view(manifest, category_key, module_ann, gene_ann, ad_threshold)


if __name__ == "__main__":
    main()
