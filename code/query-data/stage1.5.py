import re

# ─── Your formula dictionary ─────────────────────────────────────────────────
formula_dict = {'Net Profit Margin': 'Net Profit / Revenue', 
                'Return on Assets (ROA)': 'Net Profit / Total Assets', 
                'Return on Equity (ROE)': "Net Profit / Shareholder's Equity", 
                'Return on Capital Employed (ROCE)': 'EBIT / Capital Employed', 
                'EBITDA Margin': 'EBITDA / Revenue', 
                'Current Ratio': 'Current Assets / Current Liabilities', 
                'Quick Ratio (Acid Test)': '(Current Assets - Inventory) / Current Liabilities', 
                'Cash Ratio': 'Cash & Equivalents / Current Liabilities', 
                'Inventory Turnover': 'COGS / Average Inventory', 
                'Receivables Turnover': 'Revenue / Accounts Receivable', 
                'Payables Turnover': 'COGS / Accounts Payable', 
                'Asset Turnover': 'Revenue / Total Assets', 
                'Working Capital Turnover': 'Revenue / Working Capital', 
                'Debt-to-Equity Ratio': 'Total Debt / Shareholder’s Equity', 
                'Debt Ratio': 'Total Debt / Total Assets', 
                'Interest Coverage Ratio': 'EBIT / Interest Expense', 
                'Equity Ratio': 'Equity / Total Assets', 
                'Capital Gearing Ratio': '(Debt / (Debt + Equity))', 
                'Earnings Per Share (EPS)': 'Net Income / No. of Shares', 
                'Price-to-Earnings (P/E) Ratio': 'Market Price / Earnings Per Share (EPS)', 
                'Price-to-Book (P/B) Ratio': 'Market Price / Book Value per Share', 
                'Dividend Yield': 'Dividend per Share / Market Price', 
                'Dividend Payout Ratio': 'Dividend / Net Profit', 
                'Enterprise Value (EV)': 'Market Cap + Debt - Cash', 
                'EV/EBITDA': 'Enterprise Value (EV) / EBITDA', 
                'Working Capital': 'Current Assets - Current Liabilities.', 
                'Gross Profit': 'Revenue - COGS.', 
                'Free Cash Flow': 'Operating Cash Flow - CapEx', 
                'Equity': 'Assets - Liabilities'
        }


# ─── Regex extractor ──────────────────────────────────────────────────────────
def extract_vars_regex(formula_str):
    """
    Split on + - * / and parentheses to pull out variable names,
    preserving spaces/punctuation inside them.
    """
    # Split on any operator or paren
    tokens = re.split(r"[+\-*/\(\)]", formula_str)
    # Strip whitespace and trailing dots, filter out empties
    return {tok.strip().rstrip(".") for tok in tokens if tok.strip()}

# ─── Recursive resolver (using regex extractor) ──────────────────────────────
def resolve_terms(term, seen=None):
    """
    If `term` has a formula, use regex to extract its variables,
    then recurse into each. Otherwise return {term}.
    """
    if seen is None:
        seen = set()
    if term in seen:
        raise RuntimeError(f"Cyclic dependency detected on '{term}'")
    seen.add(term)

    if term not in formula_dict:
        return {term}

    formula = formula_dict[term]
    vars_ = extract_vars_regex(formula)
    atoms = set()
    for v in vars_:
        # Normalize internal whitespace
        name = " ".join(v.split())
        atoms |= resolve_terms(name, seen.copy())
    return atoms

# ─── Interactive test harness ───────────────────────────────────────────────
if __name__ == "__main__":
    print("Stage 1.5 Term Resolver (regex‐based)")
    print("Enter a glossary term (blank to exit):")
    while True:
        inp = input("> ").strip()
        if not inp:
            break

        if inp not in formula_dict:
            print(f"↳ '{inp}' has no formula → go to Stage 2 lookup\n")
            continue

        formula = formula_dict[inp]
        print(f"↳ '{inp}' formula: {formula}")
        try:
            atoms = resolve_terms(inp)
            print(f"  → atomic terms needed: {sorted(atoms)}\n")
        except Exception as e:
            print(f"  ! Error: {e}\n")
