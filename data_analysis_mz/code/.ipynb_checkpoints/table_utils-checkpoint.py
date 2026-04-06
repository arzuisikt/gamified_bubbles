# TABLE_UTILS.PY

# Helper functions to format regression results and summary statistics into LaTeX tables.
# Used by: regression_code.py (not run directly)

from pathlib import Path
import numpy as np
import pandas as pd


def stars(p):
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def fmt_num(x, digits=3):
    if pd.isna(x):
        return ""
    return f"{x:.{digits}f}"


def model_to_series(model, model_name, variable_order=None, variable_labels=None, digits=3):
    params = model.params.copy()
    bse = model.bse.copy()
    pvalues = model.pvalues.copy()

    if variable_order is None:
        variable_order = list(params.index)

    out = []
    for var in variable_order:
        if var not in params.index:
            continue

        label = variable_labels.get(var, var) if variable_labels else var
        coef = f"{fmt_num(params[var], digits)}{stars(pvalues[var])}"
        se = f"({fmt_num(bse[var], digits)})"

        out.append((label, coef, model_name))
        out.append((f"{label}__se", se, model_name))

    n = getattr(model, "nobs", np.nan)
    out.append(("Observations", str(int(n)) if pd.notna(n) else "", model_name))

    if hasattr(model, "rsquared"):
        out.append(("R-squared", fmt_num(model.rsquared, digits), model_name))
    elif hasattr(model, "prsquared"):
        out.append(("Pseudo R-squared", fmt_num(model.prsquared, digits), model_name))

    return pd.DataFrame(out, columns=["row", "value", "model"])


def regression_table_to_latex(
    models,
    model_names,
    file_path,
    title="Regression Results",
    variable_order=None,
    variable_labels=None,
    notes="Standard errors in parentheses. * p<0.10, ** p<0.05, *** p<0.01.",
    digits=3,
):
    pieces = []
    for model, name in zip(models, model_names):
        if model is None:
            continue
        pieces.append(
            model_to_series(
                model=model,
                model_name=name,
                variable_order=variable_order,
                variable_labels=variable_labels,
                digits=digits,
            )
        )

    if not pieces:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("% No estimable models.\n")
        return

    df = pd.concat(pieces, axis=0)
    table = df.pivot(index="row", columns="model", values="value").fillna("")

    latex = []
    latex.append("\\begin{table}[!htbp]")
    latex.append("\\centering")
    latex.append(f"\\caption{{{title}}}")
    latex.append("\\begin{tabular}{l" + "c" * len(table.columns) + "}")
    latex.append("\\hline")
    latex.append(" & " + " & ".join(table.columns) + " \\\\")
    latex.append("\\hline")

    for idx, row in table.iterrows():
        if idx.endswith("__se"):
            clean_idx = ""
        else:
            clean_idx = idx
        vals = " & ".join(str(v) for v in row.values)
        latex.append(f"{clean_idx} & {vals} \\\\")

    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append(f"\\begin{{flushleft}}\\footnotesize {notes}\\end{{flushleft}}")
    latex.append("\\end{table}")

    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex))


def summary_stats_to_latex(df, file_path, columns=None, title="Summary Statistics", digits=3):
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    desc = df[columns].describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]].copy()
    desc = desc.round(digits)

    latex = []
    latex.append("\\begin{table}[!htbp]")
    latex.append("\\centering")
    latex.append(f"\\caption{{{title}}}")
    latex.append(desc.to_latex())
    latex.append("\\end{table}")

    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex))