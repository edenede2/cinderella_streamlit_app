#!/usr/bin/env python3
"""Create a compact, repo-friendly CINDERellA data snapshot for the Streamlit app."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


EDEN = Path("/media/psylab-6028/DATA/Eden")
APP_ROOT = EDEN / "cinderella_streamlit_app"
DATA_ROOT = APP_ROOT / "data"

SOURCE_DATASETS = {
    "therapeutic_targets": {
        "label": "Therapeutic target selected modules",
        "path": EDEN / "CINDERellA/notebooks/results/se2_4_parallel_phenos_therapeutic_targets",
    },
    "parallel_phenos_40": {
        "label": "Correlation top-40 selected modules",
        "path": EDEN / "CINDERellA/notebooks/results/se2_4_parallel_phenos_40",
    },
    "signed_80": {
        "label": "Signed 80-module run",
        "path": EDEN / "CINDERellA/notebooks/results/se2_4_signed_80",
    },
    "muscles_therapeutic_targets": {
        "label": "Muscle tissues therapeutic target selected modules",
        "path": EDEN / "CINDERellA/notebooks/results/se2_3_muscles_parallel_phenos_therapeutic_targets",
    },
}

TARGET_DIR = (
    EDEN
    / "inter-tissue-CoExpression/data/proccessed/se2_filtered/"
    / "se2_rosmap_full_signed_alt/target_annotations"
)
SOURCE_ROOT = TARGET_DIR.parent


def copy_file(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def read_nodes(path: Path) -> pd.DataFrame:
    nodes = pd.read_csv(path, sep="\t")
    nodes["node_id"] = nodes["node_id"].astype(int)
    return nodes


def read_edges(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["source_id", "target_id", "frequency"],
        engine="python",
    )


def write_edge_bundle(src_run: Path, dst_run: Path) -> dict | None:
    edge_path = src_run / "edgefrq.txt"
    node_path = src_run / "cinderella_selected_nodes.tsv"
    if not edge_path.exists() or not node_path.exists():
        return None

    nodes = read_nodes(node_path)
    edges = read_edges(edge_path)
    id_to_pretty = dict(zip(nodes["node_id"], nodes["pretty_name"]))
    id_to_raw = dict(zip(nodes["node_id"], nodes["raw_name"]))
    edges["source_name"] = edges["source_id"].map(id_to_pretty)
    edges["target_name"] = edges["target_id"].map(id_to_pretty)
    edges["source_raw"] = edges["source_id"].map(id_to_raw)
    edges["target_raw"] = edges["target_id"].map(id_to_raw)

    dst_run.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(dst_run / "nodes.tsv", sep="\t", index=False)
    edges.to_csv(dst_run / "edges.tsv", sep="\t", index=False)

    phenotype_nodes = nodes[nodes["is_phenotype"].astype(int) == 1]
    phenotype = phenotype_nodes["pretty_name"].iloc[0] if len(phenotype_nodes) else src_run.name
    return {
        "phenotype": phenotype,
        "n_nodes": int(len(nodes)),
        "n_edges": int(len(edges)),
        "max_frequency": float(edges["frequency"].max()) if len(edges) else 0.0,
        "nodes": str((dst_run / "nodes.tsv").relative_to(DATA_ROOT)),
        "edges": str((dst_run / "edges.tsv").relative_to(DATA_ROOT)),
    }


def module_id_from_name(name: str) -> int | None:
    text = str(name).replace("ME_", "").replace("M", "")
    return int(text) if text.isdigit() else None


def gene_base_from_raw(raw_name: str) -> str:
    gene = str(raw_name).rsplit("_", 1)[0]
    return gene.split(".", 1)[0]


def gene_run_id(run_dir: Path, gene_root: Path) -> tuple[str, str, str]:
    phenotype = run_dir.name
    parent = run_dir.parents[1]
    run_type = parent.parent.name
    module = parent.name.replace("_genes", "").replace("cross_module_", "cross_")
    if module == "cinderella_final":
        module = parent.name
    rel = run_dir.relative_to(gene_root)
    safe = "__".join(part for part in rel.parts if part != "cinderella_final")
    safe = safe.replace("_genes", "").replace("PHENO_", "")
    return safe, module, phenotype


def copy_gene_runs(src_root: Path, dst_root: Path) -> list[dict]:
    gene_root = src_root / "gene_level_BN"
    if not gene_root.exists():
        return []
    runs = []
    for run_dir in sorted(gene_root.glob("**/cinderella_final/PHENO_*")):
        if not (run_dir / "edgefrq.txt").exists():
            continue
        run_id, module, phenotype = gene_run_id(run_dir, gene_root)
        bundle = write_edge_bundle(run_dir, dst_root / "gene_runs" / run_id)
        if bundle is None:
            continue
        hub_candidates = list(run_dir.parents[1].glob(f"hub_genes_{module}_{phenotype.replace('PHENO_', '')}.tsv"))
        if hub_candidates:
            copy_file(hub_candidates[0], dst_root / "gene_runs" / run_id / "hub_genes.tsv")
            bundle["hub_genes"] = str((dst_root / "gene_runs" / run_id / "hub_genes.tsv").relative_to(DATA_ROOT))
        bundle.update({"id": run_id, "module": module, "phenotype": phenotype, "run_group": run_dir.relative_to(gene_root).parts[0]})
        runs.append(bundle)
    return runs


def copy_dataset(slug: str, spec: dict) -> dict:
    src = spec["path"]
    dst = DATA_ROOT / "datasets" / slug
    dst.mkdir(parents=True, exist_ok=True)

    manifest = {
        "slug": slug,
        "label": spec["label"],
        "source_path": str(src),
        "module_runs": [],
        "gene_runs": [],
        "tables": {},
    }

    table_sources = {
        "all_phenotypes_edge_summary": src / "all_phenotypes_edge_summary.tsv",
        "normality": src / "normality_test_results.tsv",
        "selected_modules": src / "reduced_modules/reduced_modules_table.tsv",
        "all_module_phenotype_correlations": src / "reduced_modules/all_module_phenotype_correlations.tsv",
        "driver_genes_all_phenotypes": src / "gene_level_BN/driver_genes_all_phenotypes.tsv",
        "driver_genes_therapeutic": src / "gene_level_BN/visualizations_therapeutic/driver_genes_therapeutic_summary.tsv",
        "driver_genes_summary": src / "gene_level_BN/visualizations_gene_symbols/driver_genes_summary.tsv",
    }
    for key, path in table_sources.items():
        if copy_file(path, dst / "tables" / path.name):
            manifest["tables"][key] = str((dst / "tables" / path.name).relative_to(DATA_ROOT))

    for run_dir in sorted((src / "cinderella_final").glob("PHENO_*")):
        bundle = write_edge_bundle(run_dir, dst / "module_runs" / run_dir.name)
        if bundle is None:
            continue
        bundle["id"] = run_dir.name
        manifest["module_runs"].append(bundle)

    manifest["gene_runs"] = copy_gene_runs(src, dst)
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def write_target_snapshot() -> None:
    dst = DATA_ROOT / "target_annotations"
    dst.mkdir(parents=True, exist_ok=True)

    gene_ann = pd.read_csv(TARGET_DIR / "gene_target_annotations.tsv", sep="\t", low_memory=False)
    keep_gene_cols = [
        "ensembl_gene_id_base",
        "hgnc_symbol",
        "hgnc_name",
        "open_targets_ad_score",
        "open_targets_ad_datasources",
        "is_kinase",
        "is_gPCR",
        "is_ion_channel",
        "is_nuclear_receptor",
        "is_transporter",
        "is_enzyme",
        "is_receptor",
        "is_transcription_factor",
        "is_known_drug_target",
        "has_ad_evidence",
        "target_categories",
        "therapeutic_target_score",
    ]
    gene_ann = gene_ann[[c for c in keep_gene_cols if c in gene_ann.columns]].copy()
    gene_ann.to_csv(dst / "gene_target_annotations_compact.tsv", sep="\t", index=False)

    cluster = pd.read_csv(TARGET_DIR / "se2_cluster_target_summary_level4.tsv", sep="\t", low_memory=False)
    details = pd.read_csv(SOURCE_ROOT / "se2_details_filtered_4.csv")
    tissue_cols = [col for col in ["AC", "MFBA9BA46", "PCGBA23"] if col in details.columns]
    details = details.rename(
        columns={
            "Cluster ID": "cluster_id",
            "Cluster Size": "cluster_size",
            "Cluster Type": "original_cluster_type",
            "Dominant Tissue": "dominant_tissue",
        }
    )
    for col in tissue_cols:
        details[col] = pd.to_numeric(details[col], errors="coerce").fillna(0)
    details["max_tissue_fraction"] = details[tissue_cols].max(axis=1) / pd.to_numeric(
        details["cluster_size"], errors="coerce"
    ).replace(0, pd.NA)
    details["is_tissue_specific_095"] = details["max_tissue_fraction"].fillna(0).ge(0.95)
    details["cluster_tissue_class_095"] = details["is_tissue_specific_095"].map(
        {True: "Tissue specific", False: "Cross tissue"}
    )
    for col in tissue_cols:
        details[f"frac_{col}"] = details[col] / pd.to_numeric(details["cluster_size"], errors="coerce").replace(0, pd.NA)
    detail_keep = [
        "cluster_id",
        "cluster_size",
        "original_cluster_type",
        "dominant_tissue",
        "max_tissue_fraction",
        "is_tissue_specific_095",
        "cluster_tissue_class_095",
        *tissue_cols,
        *[f"frac_{col}" for col in tissue_cols],
    ]
    cluster = cluster.merge(details[detail_keep], on="cluster_id", how="left")
    keep_cluster_cols = [
        "cluster_id",
        "unique_genes",
        "tissues",
        "cluster_size",
        "original_cluster_type",
        "dominant_tissue",
        "max_tissue_fraction",
        "is_tissue_specific_095",
        "cluster_tissue_class_095",
        "AC",
        "MFBA9BA46",
        "PCGBA23",
        "frac_AC",
        "frac_MFBA9BA46",
        "frac_PCGBA23",
        "n_kinase",
        "frac_kinase",
        "n_GPCR",
        "frac_GPCR",
        "n_ion_channel",
        "frac_ion_channel",
        "n_nuclear_receptor",
        "frac_nuclear_receptor",
        "n_transporter",
        "frac_transporter",
        "n_enzyme",
        "frac_enzyme",
        "n_receptor",
        "frac_receptor",
        "n_transcription_factor",
        "frac_transcription_factor",
        "n_known_drug_target",
        "frac_known_drug_target",
        "n_Alzheimer_evidence",
        "frac_Alzheimer_evidence",
        "n_any_therapeutic_category",
        "frac_any_therapeutic_category",
        "target_category_score",
        "representative_target_genes",
        "target_rank",
    ]
    cluster = cluster[[c for c in keep_cluster_cols if c in cluster.columns]].copy()
    cluster.to_csv(dst / "module_target_summary_level4.tsv", sep="\t", index=False)


def main() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    write_target_snapshot()
    manifests = [copy_dataset(slug, spec) for slug, spec in SOURCE_DATASETS.items()]
    index = {
        "datasets": [
            {
                "slug": item["slug"],
                "label": item["label"],
                "manifest": str((DATA_ROOT / "datasets" / item["slug"] / "manifest.json").relative_to(DATA_ROOT)),
            }
            for item in manifests
        ],
        "target_annotations": {
            "genes": "target_annotations/gene_target_annotations_compact.tsv",
            "modules": "target_annotations/module_target_summary_level4.tsv",
        },
    }
    (DATA_ROOT / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Wrote data snapshot to {DATA_ROOT}")
    for manifest in manifests:
        print(
            f"- {manifest['slug']}: "
            f"{len(manifest['module_runs'])} module runs, {len(manifest['gene_runs'])} gene runs"
        )


if __name__ == "__main__":
    main()
