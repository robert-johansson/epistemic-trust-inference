"""Generate the tables and code listings of the paper's Supplementary Material from the committed logs and
source files, so that nothing in the supplement is typed by hand.

    uv run python -I tools/make_supplement.py      -> supplement/*.tex (input by the supplement's LaTeX file)
"""
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "intentions_game" / "results"
OUT = ROOT / "supplement"


def sci(x):
    """'4.63e-13' -> '$4.6 \\times 10^{-13}$'."""
    m, e = f"{float(x):.1e}".split("e")
    return f"${m} \\times 10^{{{int(e)}}}$"


def tidy(cell):
    """Typeset signs and very small p values in plain-text cells (math cells are left alone)."""
    if "\\times" in cell or "$" in cell:
        return cell
    cell = re.sub(r"(?<![\w])-(?=\d)", "$-$", cell)
    cell = re.sub(r"(?<![\w])\+(?=\d)", "$+$", cell)
    return cell.replace("(0.000)", "($<$.001)").replace("%", "\\%")


def table(caption, label, colspec, header, rows, note=None):
    lines = ["\\begin{table}[htbp]", "\\centering", "\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\renewcommand{\\arraystretch}{1.1}", f"\\begin{{tabular}}{{{colspec.replace('p{', '>{\\raggedright\\arraybackslash}p{')}}}", "\\hline",
             " & ".join(f"\\textbf{{{h}}}" for h in header) + " \\\\", "\\hline"]
    lines += [" & ".join(tidy(c) for c in r) + " \\\\" for r in rows]
    lines += ["\\hline", "\\end{tabular}"]
    if note:
        lines += ["", "\\vspace{2pt}", f"\\begin{{minipage}}{{0.95\\textwidth}}\\footnotesize {note}\\end{{minipage}}"]
    lines += ["\\end{table}", ""]
    return "\n".join(lines)


# ---------------- Table S1: validation ----------------
def validation():
    rows = []
    t = (RES / "validate_reference.log").read_text()
    for m in re.finditer(r"(full_sim_\w+)\.mat, (\w+), (\d+) participants, (real responses|simulated actions)[^\n]*\n\s+"
                         r"max \|P\(option 1\) diff\| per trial = ([\d.e+-]+)(?:; max \|total log-likelihood diff\| = ([\d.e+-]+))?",
                         t):
        f, model, n, kind, p, ll = m.groups()
        rows.append([f"NumPy translation vs.\\ authors' MATLAB output ({f.replace('_', '\\_')}.mat, {model}; "
                     f"{'real responses' if kind.startswith('real') else 'simulated actions'})", n, sci(p),
                     sci(ll) if ll else "--"])
    t = (RES / "validate_memo.log").read_text()
    for g, label in (("g1", "BPD"), ("g2", "controls")):
        vals = re.findall(rf"^{g} (\w+)\s*: max \|netp diff\| ([\d.e+-]+)\s+max \|loglik diff\| ([\d.e+-]+)", t, re.M)
        p = max(float(v[1]) for v in vals); ll = max(float(v[2]) for v in vals)
        rows.append([f"memo port vs.\\ NumPy translation (Beta, M1--M4; {label})", "50" if g == "g1" else "53",
                     sci(p), sci(ll)])
    w = re.search(r"WORST: ([\d.e+-]+)", (RES / "validate_hier.log").read_text()).group(1)
    rows.append(["Model H: memo specification vs.\\ factorized implementation", "4", sci(w), "--"])
    m = re.search(r"WORST: predictions ([\d.e+-]+), attributions ([\d.e+-]+)", (RES / "validate_intent.log").read_text())
    rows.append(["Model I: memo vs.\\ independent NumPy implementation (predictions; attributions)", "4",
                 f"{sci(m.group(1))}; {sci(m.group(2))}", "--"])
    t = (RES / "validate_hier_intent.log").read_text()
    a = re.search(r"\(1\).*= ([\d.e+-]+)", t).group(1); b = re.search(r"\(2\).*= ([\d.e+-]+)", t).group(1)
    rows.append(["Model C: memo specification vs.\\ factorized; full grid vs.\\ direct JAX", "4", f"{sci(a)}; {sci(b)}", "--"])
    return table("Numerical validation of the Intentions Game models. Maximum absolute differences in per-trial choice "
                 "probabilities and in total log-likelihood. Participants: number of participants checked (all 103 for the "
                 "published models; four participants under several parameter settings for H, I and C).",
                 "tab:validation", "p{7.2cm} p{2.1cm} p{3.6cm} p{2.6cm}",
                 ["Comparison", "Participants", "Per trial", "Log-likelihood"], rows,
                 "Logs: \\texttt{validate\\_reference.log}, \\texttt{validate\\_memo.log}, \\texttt{validate\\_hier.log}, "
                 "\\texttt{validate\\_intent.log}, \\texttt{validate\\_hier\\_intent.log}.")


# ---------------- Table S2: protected model comparison, complete ----------------
LINE = re.compile(r"^\s{2}(.+?)\s{2,}winner (\S+)\s+freq ([\d.]+)\s+xp ([\d.]+)\s+pxp ([\w.]+)(?:\s+BOR ([\d.]+))?")


def comparisons():
    t = (RES / "pxp_report.log").read_text()
    blocks = dict(re.findall(r"=== (BPD|Controls)\n(.*?)(?=\n===|\Z)", t, re.S))
    parsed = {g: [LINE.match(l).groups() for l in blocks[g].splitlines() if LINE.match(l)] for g in ("BPD", "Controls")}
    rows = []
    for (lab, w1, f1, x1, p1, b1), (_, w2, f2, x2, p2, b2) in zip(parsed["BPD"], parsed["Controls"]):
        lab = lab.replace("family: ", "Family: ").replace("authors HBI, 5 models", "Authors' HBI, 5 models")
        fmt = lambda w, f, x, p, b: f"{w}: {f}, {x}, " + ("--" if p == "nan" else p) + (f", {b}" if b else ", --")
        rows.append([lab, fmt(w1, f1, x1, p1, b1), fmt(w2, f2, x2, p2, b2)])
    return table("Complete protected model comparison (per-participant Laplace fits; the first row is the authors' hierarchical "
                 "fit, which has no protected values). Each cell: preferred model or family: estimated frequency, exceedance "
                 "probability, protected exceedance probability, Bayesian omnibus risk.",
                 "tab:bms-full", "p{5.4cm} p{4.9cm} p{4.9cm}", ["Comparison", "BPD ($n = 50$)", "Controls ($n = 53$)"],
                 rows, "Log: \\texttt{pxp\\_report.log}. Monte Carlo exceedance probabilities use $10^6$ samples (fixed seed).")


def between_group():
    t = (RES / "family_report.log").read_text()
    block = t.split("=== Between-group")[1]
    pretty = lambda n: re.sub(r"\bvs\b", "vs.", n.strip()).replace(",", ", ").replace("M1-M4", "M1--M4")
    rows = [[pretty(n), f"{float(e):.2f}", f"{float(v):.2f}"]
            for n, e, v in re.findall(r"^\s{2}(\S.+?)\s{2,}([\d.]+)\s+([\d.]+)\s*$", block, re.M)]
    return table("Posterior probability that BPD and controls differ in model frequencies (equal prior odds), computed "
                 "exactly by numerical integration over the model frequency and, for comparison, with the variational "
                 "approximation used by the VBA toolbox. The variational version understates the probability of a difference.",
                 "tab:between", "p{7.4cm} p{2.6cm} p{2.6cm}", ["Comparison", "Exact", "Variational"], rows,
                 "Log: \\texttt{family\\_report.log}.")


# ---------------- Tables S4-S6: recovery ----------------
def recovery_h():
    t = (RES / "recovery_H_analysis.log").read_text()
    names = {"a (alpha mean)": "$a$ (own $\\alpha$, mean)", "b (beta mean)": "$b$ (own $\\beta$, mean)",
             "log v_a": "log $v_\\alpha$", "log v_b": "log $v_\\beta$", "log tau_a": "log $\\tau_\\alpha$",
             "log tau_b": "log $\\tau_\\beta$"}
    rows = [[names[p.strip()], *v]
            for p, *v in re.findall(r"^(\w[\w ()]+?)\s{2,}([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*$",
                                     t.split("Trade-offs")[0], re.M)]
    m = re.search(r"Spearman = ([\d.]+); Pearson = ([\d.]+)", t)
    bpd = re.search(r"BPD-derived.*Spearman = ([\d.]+)", t).group(1); con = re.search(r"control-derived.*Spearman = ([\d.]+)", t).group(1)
    return table("Parameter recovery for model H: 103 parameter sets (one per participant), data simulated from H and refitted "
                 "blind. Correlations between true and recovered values (Pearson, Spearman), bias, root-mean-square error and "
                 "the standard deviation of the true values.", "tab:rec-h", "p{3.6cm} p{1.7cm} p{1.7cm} p{1.5cm} p{1.5cm} p{1.6cm}",
                 ["Parameter", "Pearson", "Spearman", "Bias", "RMSE", "True SD"], rows,
                 f"Overall coupling (mean log $\\tau$): Spearman {m.group(1)}, Pearson {m.group(2)}; within sets derived from BPD "
                 f"and from control participants: Spearman {bpd} and {con}. Log: \\texttt{{recovery\\_H\\_analysis.log}}.")


def recovery_i():
    t = (RES / "recovery_I_analysis.log").read_text()
    names = {"alpha (self)": "$\\alpha$ (self)", "beta (self)": "$\\beta$ (self)", "log alpha_v": "log $\\alpha_v$",
             "log beta_v": "log $\\beta_v$", "mu_omega": "$\\mu_\\omega$", "mu_rho": "$\\mu_\\rho$",
             "log sd_omega": "log $\\sigma_\\omega$", "log sd_rho": "log $\\sigma_\\rho$"}
    rows = []
    num = r"\s+([-\d.]+)"
    for raw, tex in names.items():
        v = re.search(rf"^{re.escape(raw)}{num * 7}\s*$", t, re.M).groups()   # Pearson Spearman bias RMSE SD BPD CON
        rows.append([tex, v[0], v[1], v[5], v[6], v[2], v[3]])
    g = re.search(r"true BPD ([-+\d.]+) vs CON ([-+\d.]+) \(p = ([\d.]+)\); recovered BPD ([-+\d.]+) vs CON ([-+\d.]+)", t).groups()
    return table("Parameter recovery for model I (103 parameter sets, simulated and refitted blind): correlations between true "
                 "and recovered values overall and within sets derived from each group, bias and root-mean-square error.",
                 "tab:rec-i", "p{2.6cm} p{1.5cm} p{1.7cm} p{1.9cm} p{1.9cm} p{1.3cm} p{1.3cm}",
                 ["Parameter", "Pearson", "Spearman", "Spearman BPD", "Spearman controls", "Bias", "RMSE"], rows,
                 f"Group medians of $\\mu_\\rho$: true BPD {g[0]} vs.\\ controls {g[1]} ($p = {g[2]}$); recovered BPD {g[3]} vs.\\ "
                 f"controls {g[4]}. Log: \\texttt{{recovery\\_I\\_analysis.log}}.")


def model_recovery():
    t = (RES / "model_recovery_analysis.log").read_text()
    rows = [list(r) for r in re.findall(r"^(H|M1)\s+(\d+%)\s+(\d+%)\s+(\d+%)\s+(\d+%)\s+([-\d.]+)\s+([\d.]+)\s*$", t, re.M)]
    p = dict(re.findall(r"generated by (H|M1)\s+winner \S+\s+freq [\d.]+\s+xp [\d.]+\s+pxp ([\d.]+)", (RES / "pxp_report.log").read_text()))
    rows = [r + [p[r[0]]] for r in rows]
    return table("Model recovery, H versus M1: 52 parameter sets per generating model, simulated and refitted blind with both "
                 "models. Per-participant outcomes and group-level random-effects selection.", "tab:rec-model",
                 "p{1.9cm} p{1.1cm} p{1.1cm} p{1.9cm} p{1.9cm} p{2.4cm} p{1.0cm} p{1.0cm}",
                 ["Generated by", "H wins", "M1 wins", "$|\\log\\mathrm{BF}| > 1$, correct", "$|\\log\\mathrm{BF}| > 1$, wrong",
                  "Median log BF, generating vs.\\ other", "xp", "pxp"], rows,
                 "Logs: \\texttt{model\\_recovery\\_analysis.log}, \\texttt{pxp\\_report.log}.")


# ---------------- Table S7: credulity ----------------
def credulity():
    t = (RES / "etmcq_check.log").read_text()
    rows = []
    for head, body in re.findall(r"### (.+?)\n(.*?)(?=\n###|\Z)", t, re.S):
        for dim, sym in (("alpha", "$\\alpha$"), ("beta", "$\\beta$")):
            m = re.search(rf"Credulity vs \|d {dim}_par\|\s+pooled ([-+\d.]+) \(p=([\d.]+)\) \| BPD ([-+\d.]+) \(p=([\d.]+)\) \| controls "
                          rf"([-+\d.]+) \(p=([\d.]+)\) \| group-controlled ([-+\d.]+) \(p=([\d.]+)\)", body)
            v = m.groups()
            rows.append([head if dim == "alpha" else "", sym] + [f"{v[k]} ({v[k + 1]})" for k in (0, 2, 4, 6)])
    return table("Self-reported credulity (ETMCQ) and the size of belief updating about the partner's preferences in phase 2, "
                 "for each model used to compute updating. Spearman correlations ($p$); the last column controls for group.",
                 "tab:credulity", "p{4.5cm} p{0.8cm} p{2.4cm} p{2.4cm} p{2.4cm} p{2.4cm}",
                 ["Updating computed under", "Dim.", "Pooled", "BPD", "Controls", "Group-controlled"], rows,
                 "Log: \\texttt{etmcq\\_check.log} ($n = 100$ with ETMCQ data).")


# ---------------- code listings ----------------
LISTINGS = [
    ("intentions_game/memo_models.py", ["own_choice", "partner_choice", "predict_partner"],
     "Published models M1--M4 (the Beta model uses \\texttt{\\_beta} versions of the same programs on the $\\beta$ dimension only)"),
    ("intentions_game/hier_spec.py", ["partner_likelihood", "predict_partner", "own_choice"],
     "Model H: specification with an explicit population"),
    ("intentions_game/intent.py", ["intent_choice", "predict_intent", "attribute_intent", "own_choice_intent"],
     "Model I: the partner's intent, and the attributions it implies"),
    ("intentions_game/hier_intent_spec.py", ["own_choice", "partner_likelihood"],
     "Model C (its \\texttt{predict\\_partner} is shown in Box 2 of the article)"),
    ("advice_taking/bart_et_advisor.py", ["update_reliability"],
     "Advisor model: belief update after the outcome (\\texttt{choose\\_pumps} is shown in Box 1 of the article)"),
    ("development/corriveau_trust.py", ["ask", "endorse", "update"], "Developmental case: asking, endorsing and learning"),
    ("development/corriveau_routes.py", ["believes"], "Developmental case: the child's private belief"),
]


def memo_programs(path, names):
    src = (ROOT / path).read_text()
    lines = src.splitlines()
    out = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in names and any(
                getattr(d, "id", None) == "memo" for d in node.decorator_list):
            start = min(d.lineno for d in node.decorator_list)
            out[node.name] = "\n".join(lines[start - 1:node.end_lineno])
    missing = set(names) - set(out)
    assert not missing, (path, missing)
    return [out[n] for n in names]


def listings():
    parts = []
    for path, names, title in LISTINGS:
        code = "\n\n".join(memo_programs(path, names))
        bad = {c for c in code if ord(c) > 127}
        assert not bad, (path, bad)
        parts += [f"\\subsection*{{{title}}}", f"\\noindent\\texttt{{{path.replace('_', '\\_')}}}", "",
                  "\\begin{lstlisting}", code, "\\end{lstlisting}", ""]
    return "\n".join(parts)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    files = {"tab_validation.tex": validation(), "tab_bms_full.tex": comparisons(), "tab_between.tex": between_group(),
             "tab_recovery_h.tex": recovery_h(), "tab_recovery_i.tex": recovery_i(),
             "tab_model_recovery.tex": model_recovery(), "tab_credulity.tex": credulity(), "listings.tex": listings()}
    for name, text in files.items():
        (OUT / name).write_text("% generated by tools/make_supplement.py from the committed logs and sources -- do not edit\n" + text)
        print(f"wrote supplement/{name}")
