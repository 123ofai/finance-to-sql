import os
import json
import re
import ast
import operator as op
import tempfile
import calendar
from datetime import datetime
from contextlib import contextmanager

import torch
import logging
import pandas as pd
import sqlalchemy
from sshtunnel import SSHTunnelForwarder
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from rapidfuzz import process, fuzz


# ─── LOGGER CONFIGURATION ───────────────────────────────────────────────────
logger = logging.getLogger("inference")  
logger.setLevel(logging.INFO)       

# ─── CONFIG & CONSTANTS ─────────────────────────────────────────────────────
DEFAULT_SCHEMA            = "epm1-replica.finalyzer.info_100032"
TABLE             = "fbi_entity_analysis_report"
DEFAULT_ENTITY_ID = 6450
DEFAULT_TAXONOMY  = 71
DEFAULT_CURRENCY  = "INR"
TOP_K             = 5
W_SIM1, W_RERANK1 = 0.5, 0.5
W_SIM2, W_RERANK2 = 0.6, 0.4

# ─── SECRETS FROM ENV ────────────────────────────────────────────────────────
ssh_conf   = {
    "tunnel_host": "3.110.31.32",
    "tunnel_port": 22,
    "ssh_username": "ec2-user",
}

pg_conf    = {
    "host": "10.200.51.243",  
    "port": 3306,
    "dbname": "superset",
    "user": "superset_user",
    "password": "FINadmin123#"
}

# ─── SETUP FUNCTIONS ─────────────────────────────────────────────────────────
def setup_financial_formulas():
    """Setup formula dictionary and computation utilities"""
    formula_dict = {
        #"Equity": "Assets - Liabilities",
        "Net Worth": "Current Assets - Current Liabilities",
        #"Gross Profit": "Revenue - Cost of Goods Sold (COGS)",
        #"Free Cash Flow": "Operating Cash Flow - CapEx",
        #"Gross Profit Margin": "Gross Profit / Revenue",
        "Operating Profit Margin": "Operating Profit (EBIT) / Revenue",
        "Net Profit Margin": "Net Profit (PAT) / Revenue",
        #"Return on Assets (ROA)": "Net Profit (PAT) / Total Assets",
        #"Return on Equity (ROE)": "Net Profit (PAT) / Shareholder's Equity",
        #"Return on Capital Employed (ROCE)": "Operating Profit (EBIT) / Capital Employed",
        #"EBITDA Margin": "EBITDA / Revenue",
        "Current Ratio": "Current Assets / Current Liabilities",
        "Quick Ratio (Acid Test)": "(Current Assets - Inventory) / Current Liabilities",
        #"Cash Ratio": "Cash & Equivalents / Current Liabilities",
        #"Inventory Turnover": "Cost of Goods Sold (COGS) / Average Inventory",
        #"Receivables Turnover": "Revenue / Accounts Receivable",
        #"Payables Turnover": "Cost of Goods Sold (COGS) / Accounts Payable",
        #"Asset Turnover": "Revenue / Total Assets",
        #"Working Capital Turnover": "Revenue / Working Capital",
        #"Debt-to-Equity Ratio": "Total Debt / Shareholder's Equity",
        #"Debt Ratio": "Total Debt / Total Assets",
        #"Interest Coverage Ratio": "Operating Profit (EBIT) / Interest Expense",
        #"Equity Ratio": "Equity / Total Assets",
        #"Capital Gearing Ratio": "Debt / (Debt + Equity)",
        #"Earnings Per Share (EPS)": "Net Income / No. of Shares",
        #"Price-to-Earnings (P/E) Ratio": "Market Price / Earnings Per Share (EPS)",
        #"Price-to-Book (P/B) Ratio": "Market Price / Book Value per Share",
        #"Dividend Yield": "Dividend per Share / Market Price",
        #"Dividend Payout Ratio": "Dividend / Net Profit",
        #"Enterprise Value (EV)": "Market Cap + Debt - Cash",
        #"EV/EBITDA": "Enterprise Value (EV) / EBITDA"
    }

    OPS = {
        ast.Add:  op.add,
        ast.Sub:  op.sub,
        ast.Mult: op.mul,
        ast.Div:  op.truediv,
        ast.USub: op.neg,
    }

    def extract_vars_improved(formula_str):
        """
        Improved variable extraction that handles parentheses properly
        and preserves complete terms like 'Cost of Goods Sold (COGS)' and 'No. of Shares'
        """
        try:
            # First try AST parsing for simple cases
            tree = ast.parse(formula_str, mode="eval")
            variables = set()
            
            def extract_names(node):
                if isinstance(node, ast.Name):
                    variables.add(node.id)
                elif hasattr(node, '_fields'):
                    for field_name in node._fields:
                        field_value = getattr(node, field_name)
                        if isinstance(field_value, list):
                            for item in field_value:
                                if isinstance(item, ast.AST):
                                    extract_names(item)
                        elif isinstance(field_value, ast.AST):
                            extract_names(field_value)
            
            extract_names(tree.body)
            return variables
            
        except SyntaxError:
            # Fallback for complex variable names
            variables = set()
            
            # Split by operators while being careful about parentheses
            # First, let's handle parentheses that are part of variable names vs. grouping parentheses
            
            # Step 1: Split by main arithmetic operators (+, -, *, /) but be smart about parentheses
            # We need to distinguish between:
            # - Parentheses that are part of variable names: "Cost of Goods Sold (COGS)"
            # - Parentheses used for grouping: "(Current Assets - Inventory)"
            
            # Use a more sophisticated splitting approach
            tokens = []
            current_token = ""
            paren_depth = 0
            i = 0
            
            while i < len(formula_str):
                char = formula_str[i]
                
                if char == '(':
                    paren_depth += 1
                    current_token += char
                elif char == ')':
                    paren_depth -= 1
                    current_token += char
                elif char in '+-*/' and paren_depth == 0:
                    # This is a top-level operator
                    if current_token.strip():
                        tokens.append(current_token.strip())
                    current_token = ""
                    # Skip the operator (don't add it to tokens)
                else:
                    current_token += char
                
                i += 1
            
            # Add the last token
            if current_token.strip():
                tokens.append(current_token.strip())
            
            # Step 2: Clean up tokens and handle nested parentheses
            for token in tokens:
                cleaned_token = token.strip()
                
                # Remove outer parentheses if they wrap the entire expression
                if cleaned_token.startswith('(') and cleaned_token.endswith(')'):
                    # Check if these are grouping parentheses (wrap the whole expression)
                    inner = cleaned_token[1:-1]
                    # If the inner part has operators, this might be a grouping parenthesis
                    if any(op in inner for op in '+-*/'):
                        # This looks like a grouping parenthesis, recursively extract from inner
                        inner_vars = extract_vars_improved(inner)
                        variables.update(inner_vars)
                    else:
                        # This might be part of a variable name, keep as is
                        variables.add(cleaned_token)
                else:
                    if cleaned_token:
                        variables.add(cleaned_token)
            
            # Step 3: Clean up the results
            final_variables = set()
            for var in variables:
                # Remove extra whitespace
                clean_var = ' '.join(var.split())
                if clean_var and not re.match(r'^[\s\(\)]*$', clean_var):
                    final_variables.add(clean_var)
            
            return final_variables

    def resolve_terms(term, seen=None):
        """Recursively resolve all atomic terms for a given term"""
        if seen is None:
            seen = set()
        if term in seen:
            raise RuntimeError(f"Cyclic dependency on '{term}'")
        seen.add(term)
        
        if term not in formula_dict:
            return {term}
        
        atoms = set()
        formula_vars = extract_vars_improved(formula_dict[term])
        
        for var in formula_vars:
            # Clean the variable name
            clean_var = " ".join(var.split())
            atoms |= resolve_terms(clean_var, seen.copy())
        
        return atoms

    def eval_node(node, vars_, formula_dict_ref):
        """Enhanced eval_node that can handle nested formulas"""
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.Constant):  # For newer Python versions
            return node.value
        elif isinstance(node, ast.Name):
            var_name = node.id
            if var_name in vars_:
                return vars_[var_name]
            elif var_name in formula_dict_ref:
                # This is a nested formula, compute it recursively
                return compute_formula_recursive(formula_dict_ref[var_name], vars_, formula_dict_ref)
            else:
                raise KeyError(f"Unknown variable '{var_name}'")
        elif isinstance(node, ast.BinOp):
            L = eval_node(node.left, vars_, formula_dict_ref)
            R = eval_node(node.right, vars_, formula_dict_ref)
            return OPS[type(node.op)](L, R)
        elif isinstance(node, ast.UnaryOp):
            return OPS[type(node.op)](eval_node(node.operand, vars_, formula_dict_ref))
        else:
            raise TypeError(f"Unsupported AST node {type(node)}")

    def compute_formula_recursive(formula_str, variables, formula_dict_ref):
        """Recursively compute formulas, handling nested formula dependencies"""
        try:
            # Try to parse as a valid Python expression
            tree = ast.parse(formula_str, mode="eval")
            return eval_node(tree.body, variables, formula_dict_ref)
        except SyntaxError:
            # Handle cases where formula contains terms that aren't valid Python identifiers
            # We need to create a mapping from clean variable names to placeholder names
            
            # Extract all variables from the formula
            formula_vars = extract_vars_improved(formula_str)
            
            # Create a mapping from original terms to Python-safe identifiers
            var_mapping = {}
            safe_formula = formula_str
            
            for i, var in enumerate(sorted(formula_vars, key=len, reverse=True)):  # Sort by length to avoid substring issues
                safe_name = f"var_{i}"
                var_mapping[safe_name] = var
                safe_formula = safe_formula.replace(var, safe_name)
            
            # Create safe_variables dict with mapped names
            safe_variables = {}
            for safe_name, original_name in var_mapping.items():
                clean_original = " ".join(original_name.split())
                if clean_original in variables:
                    safe_variables[safe_name] = variables[clean_original]
                elif clean_original in formula_dict_ref:
                    # Recursively compute nested formula
                    safe_variables[safe_name] = compute_formula_recursive(
                        formula_dict_ref[clean_original], variables, formula_dict_ref
                    )
                else:
                    raise KeyError(f"Unknown variable '{clean_original}'")
            
            # Parse and evaluate the safe formula
            tree = ast.parse(safe_formula, mode="eval")
            return eval_node(tree.body, safe_variables, formula_dict_ref)

    def compute_formula(formula_str, variables):
        """Main interface for formula computation"""
        return compute_formula_recursive(formula_str, variables, formula_dict)

    return {
        'formula_dict': formula_dict,
        'resolve_terms': resolve_terms,
        'compute_formula': compute_formula,
        'extract_vars_regex': extract_vars_improved
    }

# ─── DATABASE CONNECTION HELPER ─────────────────────────────────────────────
@contextmanager
def get_db_connection(key_path):
    """Context manager for database connections to avoid resource leaks"""
    tunnel = None
    engine = None
    try:
        tunnel = SSHTunnelForwarder(
            (ssh_conf['tunnel_host'], ssh_conf['tunnel_port']),
            ssh_username=ssh_conf['ssh_username'],
            ssh_pkey=key_path,
            remote_bind_address=(pg_conf['host'], pg_conf['port'])
        )
        tunnel.start()
        
        conn_str = (
            f"postgresql://{pg_conf['user']}:{pg_conf['password']}@"
            f"127.0.0.1:{tunnel.local_bind_port}/{pg_conf['dbname']}"
        )
        engine = sqlalchemy.create_engine(conn_str)
        yield engine
    finally:
        if engine:
            engine.dispose()
        if tunnel:
            tunnel.stop()

# ─── DATABASE FETCH HELPER ───────────────────────────────────────────────────
def fetch_metric(gid, period_id, key_path, input_data):
    """Fetch metric value from database - direct, efficient approach"""
    sql = f"""
        SELECT value FROM "{input_data["schema"]}"."{TABLE}"
        WHERE entity_id={input_data["entity_id"]}
        AND grouping_id={gid}
        AND period_id='{period_id}'
        AND nature_of_report='{input_data["nature"]}'
        AND scenario='{input_data["scenario"]}'
        AND taxonomy_id={input_data["taxonomy"]}
        AND reporting_currency='{input_data["currency"]}';
        """
    
    with get_db_connection(key_path) as engine:
        df = pd.read_sql(sql, engine)
    
    if df.empty:
        return None
    
    ret_value = df['value'].iat[0].strip().replace(',', '')
    return 0.0 if ret_value == '-' else float(ret_value)

def extract_glossary(nl: str, resources) -> str:
    bi_encoder = resources['bi_encoder']
    term_embs = resources['term_embs']
    term_texts = resources['term_texts']
    reranker_1 = resources['reranker_1']
    
    q_emb = bi_encoder.encode(nl, convert_to_tensor=True, normalize_embeddings=True)
    sims = util.cos_sim(q_emb, term_embs)[0]
    idx = torch.topk(sims, k=TOP_K).indices.tolist()
    tops = [term_texts[i] for i in idx]
    top_sims = [sims[i].item() for i in idx]
    rerank = reranker_1.predict([(nl, t) for t in tops])
    scores = [W_SIM1 * s + W_RERANK1 * r for s, r in zip(top_sims, rerank)]
    best = int(torch.tensor(scores).argmax().item())
    return tops[best].split(" can be defined as ")[0]

def lookup_grouping(gloss: str, resources):
    """
    Simplified lookup_grouping that only uses similarity scoring (no reranker)
    Returns the top similarity match directly
    """
    bi_encoder = resources['bi_encoder']
    label_embs = resources[resources['cur_schema'] + '_label_embs']
    label_texts = resources[resources['cur_schema'] + '_label_texts']
    group_df = resources[resources['cur_schema'] + '_group_df']
    
    # Encode the glossary term
    q_emb = bi_encoder.encode(gloss, convert_to_tensor=True, normalize_embeddings=True)
    
    # Calculate similarity scores with all labels
    sims = util.cos_sim(q_emb, label_embs)[0]
    
    # Get the index of the highest similarity score
    best_idx = int(sims.argmax().item())
    
    # Get the best matching label
    best_label = label_texts[best_idx]
    
    # Find the corresponding grouping_id
    gid = int(group_df.loc[group_df['grouping_label'] == best_label, 'grouping_id'].iat[0])
    
    return best_label, gid

# ─── CONSOLIDATED PERIOD RESOLUTION FUNCTIONS ───────────────────────────────
def construct_period_id(nl: str, resources) -> str:
    """
    Consolidated function to parse natural language and construct period_id
    Returns: period_id in format "YYYY_NATURE_VIEW_SEQUENCE"
    """
    # Get the period encoder (same as bi_encoder)
    period_encoder = resources['bi_encoder']
    view_embs = resources['view_embs']
    
    # Keywords for view detection
    FTP_KEYWORDS = [
        r"\bfor the period\b", r"\bfor that period\b", r"\bjust that month\b",
        r"\bonly that quarter\b", r"\bas at\b", r"\bas at\s+(?:month|quarter)\b",
    ]
    PRD_KEYWORDS = [
        r"\byear to date\b", r"\bytd\b", r"\bso far\b", r"\bcumulative\b",
        r"\bthrough\b", r"\bup to\b", r"\bas of\b", r"\bto date\b",
        r"\bsince the start of the year\b", r"\bmonth to date\b", r"\bmtd\b",
        r"\bquarter to date\b", r"\bqtd\b", r"\bthrough end of\b",
        r"\bthrough end-of-period\b"
    ]

    # Build regex for any period unit
    months = [m.lower() for m in calendar.month_name if m]
    month_regex = r"(?:{})".format("|".join(months))
    quarter_regex = r"(?:q[1-4]|quarter\s*[1-4])"
    half_regex = r"(?:h1|h2|first half|second half|half-year\s*[12])"
    fy_regex = r"(?:fy\s*\d{2,4}|financial year)"
    period_unit_regex = rf"{month_regex}|{quarter_regex}|{half_regex}|{fy_regex}"

    # "Verbose" prototypes for semantic fallback
    FTP_PROTOS_VERB = [
        "FTP can be defined as 'for the period' meaning only that month or quarter",
        "FTP can be defined as 'for that period only' meaning the single slice of time",
        "FTP can be defined as 'at month end' meaning only that month",
        "FTP can be defined as 'year ended' meaning the year‐end snapshot",
    ]
    PRD_PROTOS_VERB = [
        "PRD can be defined as 'to date' meaning cumulative up until now",
        "PRD can be defined as 'year to date' meaning aggregated so far this fiscal year",
        "PRD can be defined as 'month to date' meaning cumulative this month",
        "PRD can be defined as 'quarter to date' meaning cumulative this quarter",
        "PRD can be defined as 'so far' meaning sum of all periods up until date",
    ]
    VIEW_PROTOS = FTP_PROTOS_VERB + PRD_PROTOS_VERB

    def detect_view(nl: str) -> str:
        low = nl.lower()

        # 1) If any explicit PRD keyword, return PRD
        for pat in PRD_KEYWORDS:
            if re.search(pat, low):
                return "PRD"

        # 2) If "for <period-unit>" pattern without PRD keyword, return FTP
        if re.search(rf"\bfor\b.*\b{period_unit_regex}\b", low):
            return "FTP"

        # 3) Semantic fallback
        q_emb = period_encoder.encode(nl, convert_to_tensor=True, normalize_embeddings=True)
        sims = util.cos_sim(q_emb, view_embs)[0]
        best_idx = int(sims.argmax().item())
        best_score = sims[best_idx].item()

        # 4) If best semantic score is low, default to FTP
        if best_score < 0.45:
            return "FTP"

        best_proto = VIEW_PROTOS[best_idx]
        return "FTP" if best_proto in FTP_PROTOS_VERB else "PRD"

    def extract_year(nl: str) -> int:
        m = re.search(r"\b(19|20)\d{2}\b", nl)
        return int(m.group(0)) if m else datetime.now().year

    def extract_nature(nl: str) -> str:
        low = nl.lower()
        if re.search(r"\bquarter\b|\bq[1-4]\b", low): return "FQ"
        if re.search(r"\bhalf\b|\bh1\b|\bh2\b", low): return "FH"
        if re.search(r"\bfinancial year\b|\bfy\b", low): return "FY"
        return "M"

    def extract_sequence(nl: str, nat: str) -> int:
        # Candidates for fuzzy matching
        month_names_full = [m.lower() for m in calendar.month_name if m]
        month_names_abbr = [m[:3].lower() for m in calendar.month_name if m]
        all_month_candidates = list(set(month_names_full + month_names_abbr))

        quarter_candidates = [
            "q1","q2","q3","q4",
            "quarter 1","quarter 2","quarter 3","quarter 4",
            "1st quarter","2nd quarter","3rd quarter","4th quarter"
        ]
        half_candidates = ["h1","h2","first half","second half","half-year 1","half-year 2"]

        ORDINAL_MAP = {
            "first":1,"1st":1,
            "second":2,"2nd":2,
            "third":3,"3rd":3,
            "fourth":4,"4th":4
        }
        NUM_WORD_MAP = {
            "one":1,"two":2,"three":3,"four":4,
            "five":5,"six":6,"seven":7,"eight":8,
            "nine":9,"ten":10,"eleven":11,"twelve":12
        }

        def fuzzy_match_token(token: str, candidates: list, threshold=75):
            match, score, _ = process.extractOne(token, candidates, scorer=fuzz.ratio)
            return match if score >= threshold else None

        low = nl.lower()

        # 1) EXPLICIT MONTH
        if nat == "M":
            # (a) Exact full month
            for i, name in enumerate(calendar.month_name[1:], start=1):
                if name.lower() in low:
                    return i
            # (b) Exact 3-letters
            for i, name in enumerate(calendar.month_name[1:], start=1):
                if name.lower()[:3] in low:
                    return i
            # (c) Numeric word "one" → 1 … "twelve" → 12
            for word, val in NUM_WORD_MAP.items():
                if re.search(rf"\b{word}\b", low) and 1 <= val <= 12:
                    return val
            # (d) Regex "month <number>"
            m = re.search(r"month\s+(\d{1,2})", low)
            if m:
                num = int(m.group(1))
                if 1 <= num <= 12:
                    return num
            # (e) Fuzzy-match tokens
            tokens = re.findall(r"\w+", low)
            for t in tokens:
                fm = fuzzy_match_token(t, all_month_candidates)
                if fm:
                    if fm in month_names_full:
                        return month_names_full.index(fm) + 1
                    if fm in month_names_abbr:
                        return month_names_abbr.index(fm) + 1
            # (f) "last month" or "last N months"
            if re.search(r"last\s+month", low):
                seq = datetime.now().month - 1
                return seq if seq >= 1 else 1
            m = re.search(r"last\s+(\d+)\s+months?", low)
            if m:
                n = int(m.group(1))
                seq = datetime.now().month - n
                return seq if seq >= 1 else 1

        # 2) EXPLICIT QUARTER
        if nat == "FQ":
            m = re.search(r"q([1-4])", low)
            if m:
                return int(m.group(1))
            for w, val in ORDINAL_MAP.items():
                if f"{w} quarter" in low:
                    return val
            tokens = re.findall(r"\w+", low)
            for t in tokens:
                fm = fuzzy_match_token(t, quarter_candidates)
                if fm:
                    num = re.search(r"([1-4])", fm)
                    if num:
                        return int(num.group(1))
            if re.search(r"last\s+quarter", low):
                anchor = (datetime.now().month - 1)//3 + 1
                seq = anchor - 1
                return seq if seq >= 1 else 1
            m = re.search(r"last\s+(\d+)\s+quarters?", low)
            if m:
                n = int(m.group(1))
                anchor = (datetime.now().month - 1)//3 + 1
                seq = anchor - n
                return seq if seq >= 1 else 1

        # 3) EXPLICIT HALF
        if nat == "FH":
            if re.search(r"h1\b|first\s+half", low):
                return 1
            if re.search(r"h2\b|second\s+half", low):
                return 2
            for w, val in ORDINAL_MAP.items():
                if f"{w} half" in low:
                    return val
            tokens = re.findall(r"\w+", low)
            for t in tokens:
                fm = fuzzy_match_token(t, half_candidates)
                if fm:
                    if "1" in fm or "first" in fm:
                        return 1
                    if "2" in fm or "second" in fm:
                        return 2
            # Fallback half by current month
            month = datetime.now().month
            return 1 if month <= 6 else 2

        # 4) FISCAL YEAR → always 1
        if nat == "FY":
            return 1

        # 5) FALLBACK: current period
        month = datetime.now().month
        if nat == "M":
            return month
        if nat == "FQ":
            return (month - 1)//3 + 1
        if nat == "FH":
            return 1 if month <= 6 else 2
        return 1

    # Extract components
    year = extract_year(nl)
    period_nature = extract_nature(nl)
    view = detect_view(nl)
    seq = extract_sequence(nl, period_nature)
    
    return f"{year}_{period_nature}_{view}_{seq}"

def model_fn(model_dir, *args):
    """
    1st function called by SageMaker to load the model and any other artifacts.
    - Should load/return all resources that are required during the predict phase
    """

    # Setup financial formulas
    formula_ops = setup_financial_formulas()

    # Creating Label Term Infra
    gloss_df = pd.read_csv(os.path.join(model_dir, "data", "glossary.csv"))
    def build_full_text(r):
        txt = f"{r['Glossary']} can be defined as {r['Description']}"
        return txt
    term_texts = gloss_df.apply(build_full_text, axis=1).tolist()

    # Loading other trained models
    print('Loading Models')
    #device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    device = 'cpu'
    print('CUDA Availability: ', torch.cuda.is_available())
    bi_encoder = SentenceTransformer('BAAI/bge-large-en-v1.5', device=device)
    s1_dir = os.path.join(model_dir, "models", "stage1_cross_encoder_finetuned_bge_balanced_data_top10")
    #s2_dir = os.path.join(model_dir, "models", "stage2_cross_encoder_finetuned_MiniLM_new_top5")
    reranker_1 = CrossEncoder(s1_dir, num_labels=1, device=device)
    #reranker_2 = CrossEncoder(s2_dir, num_labels=1, device=device)
    print('Stage 1,2 models loaded')

    # Loading Stage 0
    print('Loading Stage 0 tokeniser')
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, "models", "stage0_model"), use_fast=False)
    print('Loading stage 0 model')
    model = AutoModelForSequenceClassification.from_pretrained(os.path.join(model_dir, "models", "stage0_model"))
    # choose device: GPU if available, else CPU
    #device = 0 if torch.cuda.is_available() else -1
    device = 0
    # instantiate HF pipeline for text-classification
    stage0_clf = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)
    print('Pipeline loaded - stage 0')

    with torch.no_grad():
        term_embs = bi_encoder.encode(term_texts, convert_to_tensor=True, normalize_embeddings=True)
        view_embs = bi_encoder.encode(
            [
                "FTP can be defined as 'for the period' meaning only that month or quarter",
                "FTP can be defined as 'for that period only' meaning the single slice of time",
                "FTP can be defined as 'at month end' meaning only that month",
                "FTP can be defined as 'year ended' meaning the year‐end snapshot",
                "PRD can be defined as 'to date' meaning cumulative up until now",
                "PRD can be defined as 'year to date' meaning aggregated so far this fiscal year",
                "PRD can be defined as 'month to date' meaning cumulative this month",
                "PRD can be defined as 'quarter to date' meaning cumulative this quarter",
                "PRD can be defined as 'so far' meaning sum of all periods up until date"
            ],
            convert_to_tensor=True,
            normalize_embeddings=True
        )
    
    # Loading default schema's embeddings - for caching
    with get_db_connection(os.path.join(model_dir,'data', 'private_key.pem')) as engine:
        group_df = pd.read_sql(f'SELECT grouping_id, grouping_label FROM "{DEFAULT_SCHEMA}".fbi_grouping_master', con=engine)
        print('Connection done - default group_df loaded')

        # ✅ Clean the grouping_label column (important!)
        group_df = group_df.dropna(subset=['grouping_label'])
        group_df['grouping_label'] = (
            group_df['grouping_label']
                .astype(str)
                .str.strip()
                .str.replace(r'\s+', ' ', regex=True)
            )
        group_df = group_df[group_df['grouping_label'] != '']  # remove empty strings

    with torch.no_grad():
        label_texts = group_df['grouping_label'].tolist()
        label_embs = bi_encoder.encode(label_texts, convert_to_tensor=True, normalize_embeddings=True, batch_size=128)
    print('Grouping Embeddings Loaded')

    return {
        "gloss_df": gloss_df,
        "bi_encoder": bi_encoder,
        "reranker_1": reranker_1,
        #"reranker_2": reranker_2,
        "term_texts": term_texts,
        "term_embs": term_embs,
        "view_embs": view_embs,
        "stage0_clf": stage0_clf,
        "model_dir": model_dir,
        # Add formula operations
        "formula_dict": formula_ops['formula_dict'],
        "resolve_terms": formula_ops['resolve_terms'],
        "compute_formula": formula_ops['compute_formula'],
        DEFAULT_SCHEMA + '_group_df': group_df,
        DEFAULT_SCHEMA + "_label_texts": label_texts,
        DEFAULT_SCHEMA + "_label_embs": label_embs
    }

def input_fn(request_body, content_type="application/json"):
    """
    Deserialize the incoming payload to a Python dictionary.
    """
    if content_type == "application/json":
        return json.loads(request_body)
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, resources):
    """
    Call the handler in inference_log module with the deserialized input.
    input_data = {
        "query": "Find the PAT for the period jan 2025",
        "taxonomy": "", 
        "currency": "", 
        "schema": "" , 
        "entity_id":"", 
        "scenario": "",
        "nature": ""
        }
    """

    # Parse Inputs
    query = input_data["query"]

    ###### 0) Load relevant resources #####
    stage0_clf = resources["stage0_clf"]
    key_path = os.path.join(resources["model_dir"],'data', 'private_key.pem')

    # Load context variables with proper defaults
    input_data["taxonomy"] = input_data.get("taxonomy") or DEFAULT_TAXONOMY
    input_data["currency"] = input_data.get("currency") or DEFAULT_CURRENCY
    input_data["entity_id"] = input_data.get("entity_id") or DEFAULT_ENTITY_ID
    
    # Will pick from query if mentioned, else fallback to 'Actual'
    if not input_data.get("scenario"):
        input_data["scenario"] = 'Forecast' if 'forecast' in query.lower() or 'budget' in query.lower() else \
                    ('Cashflow' if 'cash' in query.lower() else 'Actual')
    
    input_data["nature"] = input_data.get("nature") or 'Standalone'
    input_data["schema"] = input_data.get("schema") or DEFAULT_SCHEMA
    
    # Handle schema-specific resources
    if input_data["schema"] != DEFAULT_SCHEMA and input_data["schema"] + '_group_df' not in resources:
        # Caching group_df, embedding results for future use
        with get_db_connection(key_path) as engine:
            group_df = pd.read_sql(f'SELECT grouping_id, grouping_label FROM "{input_data["schema"]}".fbi_grouping_master', con=engine)
            print('Connection done - given group_df available')
        
        if group_df.empty:
            return "Invalid Schema"

        # Clean data
        group_df = group_df.dropna(subset=['grouping_label'])
        group_df['grouping_label'] = (
            group_df['grouping_label']
                .astype(str)
                .str.strip()
                .str.replace(r'\s+', ' ', regex=True)
            )
        group_df = group_df[group_df['grouping_label'] != '']

        with torch.no_grad():
            label_texts = group_df['grouping_label'].tolist()
            label_embs = resources["bi_encoder"].encode(label_texts, convert_to_tensor=True, normalize_embeddings=True, batch_size=128)

        resources[input_data["schema"] + "_group_df"] = group_df
        resources[input_data["schema"] + "_label_texts"] = label_texts
        resources[input_data["schema"] + "_label_embs"] = label_embs

    resources['cur_schema'] = input_data['schema']

    ######## PIPELINE BEGINS #########
    # 1) Classification + logging
    print('Stage 0 - Classification starting')
    cls_out = stage0_clf(query, top_k=None)[0]
    label_idx = int(cls_out["label"].split("_")[-1])
    logger.info(json.dumps({
        "event": "classification",
        "query": query,
        "predicted_label": label_idx,
        "score": cls_out["score"]
    }))

    # 2) Short‑circuit non‑label‑0
    if label_idx != 0:
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": (
                    "This question belongs to the type of query we are currently working on; "
                    "we will come back with a solution as soon as possible. "
                    "Thank you for your patience."
                ),
                "predicted_label": label_idx
            })
        }

    # 3) Original inference logic for label 0
    logger.info(json.dumps({
        "event": "request_received: label 0 complete",
        "query": query,
    }))

    # Stage 1: Extract glossary term
    gloss = extract_glossary(query, resources)
    logger.info(json.dumps({
        "event": "stage1_complete",
        "glossary_term": gloss
    }))

    print('Query:', query)
    print('Stage 0 - Classification:', label_idx)
    print('Glossary:', gloss)

    # Construct period_id using consolidated function
    period_id = construct_period_id(query, resources)
    print('Period:', period_id)
    logger.info(json.dumps({
        "event": "period_constructed",
        "period_id": period_id
    }))

    # Check if formula calculation is needed
    group_df = resources[input_data["schema"] + '_group_df']
    label2id = dict(zip(group_df['grouping_label'].str.strip(), group_df['grouping_id']))
    
    sql = None
    
    if gloss in resources['formula_dict']:
        # Formula calculation path
        atoms = resources['resolve_terms'](gloss)
        
        vals = {}
        atom_details = {}  # ← Add this to store grouping details
        for atom in atoms:
            # Find the best grouping label & its ID for this atom
            grouping_label, gid = lookup_grouping(atom, resources)
            # Store the grouping details
            atom_details[atom] = {
                "grouping_label": grouping_label,
                "grouping_id": gid
            }
            # Fetch the metric value
            vals[atom] = fetch_metric(gid, period_id, key_path, input_data)
        
        result = resources['compute_formula'](resources['formula_dict'][gloss], vals)
        
        response = {
            "glossary_term": gloss,
            "formula": resources['formula_dict'][gloss],
            "atoms": list(atoms),
            "atom_values": vals,
            "atom_details": atom_details,  # ← Add this line
            "period_id": period_id,
            "value": result
        }

    else:
        # Direct lookup path
        # Stage 2: Lookup grouping
        label, gid = lookup_grouping(gloss, resources)
        print('Grouping label & ID:', label, gid)
        logger.info(json.dumps({
            "event": "stage2_complete",
            "glossary_term": gloss,
            "grouping_label": label,
            "grouping_id": gid
        }))

        # Generate SQL
        sql = f"""
                SELECT value FROM "{input_data["schema"]}"."{TABLE}"
                WHERE entity_id={input_data["entity_id"]}
                AND grouping_id={gid}
                AND period_id='{period_id}'
                AND nature_of_report='{input_data["nature"]}'
                AND scenario='{input_data["scenario"]}'
                AND taxonomy_id={input_data["taxonomy"]}
                AND reporting_currency='{input_data["currency"]}';
                """.strip()
        
        logger.info(json.dumps({
            "event": "sql_generated",
            "sql": sql
        }))
        print('SQL:', sql)

        # Fetch result
        result = fetch_metric(gid, period_id, key_path, input_data)

        response = {
            "glossary_term": gloss,
            "grouping_label": label,
            "grouping_id": gid,
            "sql": sql,
            "period_id": period_id,
            "nature": input_data["nature"],
            "scenario": input_data["scenario"],
            "value": result
        }

    logger.info(json.dumps({
        "event": "result_ready",
        "value": result
    }))

    return response

def output_fn(prediction, accept="application/json"):
    """
    Serialize the prediction output to a JSON string.
    """
    return json.dumps(prediction), accept