from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import json
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
from PIL import Image
import sqlite3
from itertools import combinations

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# =====================================================================
# PREPROCESSING PIPELINE
# =====================================================================

class PreprocessingPipeline:
    def __init__(self, data):
        self.data = data.copy()
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.outlier_count = 0
        self.encoded_cols = 0
        self.selected_features = 0

    def handle_missing_values(self):
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            self.data[numeric_cols] = self.imputer.fit_transform(self.data[numeric_cols])
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            self.data[col] = self.data[col].fillna(
                self.data[col].mode()[0] if len(self.data[col].mode()) > 0 else 'Unknown')
        return self

    def handle_outliers(self):
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            mask = (self.data[col] >= Q1 - 1.5 * IQR) & (self.data[col] <= Q3 + 1.5 * IQR)
            self.outlier_count += int((~mask).sum())
            self.data = self.data[mask]
        return self

    def label_encoding(self):
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        self.encoded_cols = len(categorical_cols)
        for col in categorical_cols:
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col].astype(str))
            self.label_encoders[col] = le
        return self

    def normalization(self):
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            self.data[numeric_cols] = self.scaler.fit_transform(self.data[numeric_cols])
        return self

    def feature_selection(self, target_col=None):
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if target_col and target_col in numeric_cols:
            numeric_cols.remove(target_col)
        if len(numeric_cols) > 8:
            X = self.data[numeric_cols]
            selector = SelectKBest(f_classif, k=8)
            try:
                y = self.data[target_col] if target_col and target_col in self.data.columns else np.zeros(len(self.data))
                selector.fit(X, y)
                selected = [numeric_cols[i] for i in np.argsort(selector.scores_)[-8:]]
                keep = selected + ([target_col] if target_col and target_col in self.data.columns else [])
                self.data = self.data[keep]
                self.selected_features = 8
            except:
                self.selected_features = len(numeric_cols)
        else:
            self.selected_features = len(numeric_cols)
        return self

    def fit_transform(self, target_col=None):
        return (self.handle_missing_values()
                    .handle_outliers()
                    .label_encoding()
                    .normalization()
                    .feature_selection(target_col))


# =====================================================================
# APRIORI MODEL
# =====================================================================

class AprioriModel:
    def __init__(self, min_support=0.2, min_confidence=0.5):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.itemsets = []
        self.rules = []

    def _discretize(self, df, target_col=None):
        """
        Diskretisasi fitur numerik → _HIGH/_LOW.
        Kolom target TIDAK di-diskretisasi — nilainya dipakai langsung sebagai label.
        """
        transactions = []
        medians = {col: df[col].median()
                   for col in df.columns
                   if col != target_col and pd.api.types.is_numeric_dtype(df[col])}
        for _, row in df.iterrows():
            t = []
            for col, val in row.items():
                if pd.isna(val):
                    continue
                if col == target_col:
                    # Simpan label asli, format "Disease=<label>"
                    t.append(f"Disease={val}")
                elif col in medians:
                    tag = f"{col}_HIGH" if val >= medians[col] else f"{col}_LOW"
                    t.append(tag)
                else:
                    t.append(f"{col}_{val}")
            transactions.append(t)
        return transactions

    def fit(self, df, target_col):
        # ── Preprocessing: hanya fitur (bukan target) yang di-encode/scale ──
        feat_cols = [c for c in df.columns if c != target_col]
        df_feats = df[feat_cols].copy()

        num_cols = df_feats.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            imp = SimpleImputer(strategy='mean')
            df_feats[num_cols] = imp.fit_transform(df_feats[num_cols])

        # Tambahkan kembali kolom target asli (string)
        processed_df = df_feats.copy()
        if target_col in df.columns:
            processed_df[target_col] = df[target_col].values

        transactions = self._discretize(processed_df, target_col)
        n = len(transactions)
        min_count = int(np.ceil(self.min_support * n))

        # Count 1-itemsets
        item_count = {}
        for t in transactions:
            for item in set(t):
                item_count[item] = item_count.get(item, 0) + 1

        freq_items = {frozenset([k]): v for k, v in item_count.items() if v >= min_count}

        # Count 2-itemsets
        freq_items_2 = {}
        item_list = [list(k)[0] for k in freq_items]
        for i in range(len(item_list)):
            for j in range(i+1, len(item_list)):
                pair = frozenset([item_list[i], item_list[j]])
                count = sum(1 for t in transactions if item_list[i] in t and item_list[j] in t)
                if count >= min_count:
                    freq_items_2[pair] = count

        self.itemsets = list(freq_items.keys()) + list(freq_items_2.keys())

        # Generate rules from 2-itemsets
        # Pisahkan: disease_rules (fitur → Disease) dan feat_rules (fitur → fitur)
        self.rules        = []   # semua rules (untuk /predict route)
        self.disease_rules = []  # HANYA fitur → Disease (untuk prediksi label)

        for itemset, count in freq_items_2.items():
            items = list(itemset)
            support = count / n
            has_disease = any(x.startswith('Disease=') for x in items)
            for i in range(len(items)):
                ant  = items[i]
                cons = items[1-i]
                # Skip rule "Disease → fitur" (direction terbalik, tidak berguna untuk prediksi)
                if ant.startswith('Disease='):
                    continue
                ant_sup = item_count.get(ant, 0) / n
                conf    = support / ant_sup if ant_sup > 0 else 0
                # Untuk disease rules, turunkan threshold sedikit agar lebih banyak match
                threshold = self.min_confidence * 0.5 if has_disease else self.min_confidence
                if conf >= threshold:
                    display_cons = cons.replace('Disease=', '') if cons.startswith('Disease=') else cons
                    rule = {
                        'antecedent': ant,
                        'consequent': display_cons,
                        'support':    round(support, 4),
                        'confidence': round(conf, 4),
                        'lift':       round(conf / (item_count.get(cons, 1) / n), 4)
                    }
                    self.rules.append(rule)
                    if has_disease:
                        self.disease_rules.append(rule)
        return self

    def get_summary(self):
        confs = [r['confidence'] for r in self.rules] if self.rules else [0]
        return {
            'total_itemsets': len(self.itemsets),
            'total_rules': len(self.rules),
            'avg_confidence': round(float(np.mean(confs)), 4),
            'top_rules': sorted(self.rules, key=lambda x: x['confidence'], reverse=True)[:5]
        }


# =====================================================================
# APRIORI TID MODEL
# =====================================================================
# Diskritisasi: _H (≥ median), _L (< median)  — konsisten dengan AprioriTid.ipynb
# Prediksi    : TID co-occurrence scoring (bukan rule antecedent matching)
#               karena AprioriTid menyimpan frequent patterns + TID sets,
#               bukan association rules ber-confidence seperti Apriori biasa.
# =====================================================================

class AprioriTidModel:
    def __init__(self, min_support=0.2, min_confidence=0.5):
        self.min_support    = min_support
        self.min_confidence = min_confidence
        # tid_lists: item_tag → set of TIDs   (1-itemset)
        self.tid_lists      = {}
        # pair_tids: frozenset{a,b} → set of TIDs (2-itemset)
        self.pair_tids      = {}
        # rules: list of dicts dengan confidence/lift (dari 2-itemset)
        self.rules          = []
        self.n_transactions = 0

    # ── diskritisasi median → _H / _L ─────────────────────────────
    def _tag(self, col, val, median):
        return f"{col}_H" if val >= median else f"{col}_L"

    def fit(self, df, target_col):
        # ── Preprocessing: hanya fitur, bukan target ──────────────
        feat_cols = [c for c in df.columns if c != target_col]
        df_feats  = df[feat_cols].copy()
        num_cols  = df_feats.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            imp = SimpleImputer(strategy='mean')
            df_feats[num_cols] = imp.fit_transform(df_feats[num_cols])

        # Gabungkan kembali kolom target asli (string)
        processed_df = df_feats.copy()
        if target_col in df.columns:
            processed_df[target_col] = df[target_col].values

        n = len(processed_df)
        self.n_transactions = n
        min_count = int(np.ceil(self.min_support * n))

        # Median hanya untuk fitur numerik (bukan target)
        medians = {col: processed_df[col].median()
                   for col in processed_df.columns
                   if col != target_col and pd.api.types.is_numeric_dtype(processed_df[col])}

        # ── Bangun TID lists per 1-itemset ─────────────────────────
        self.tid_lists = {}
        for tid, (_, row) in enumerate(processed_df.iterrows()):
            tags = []
            for col, val in row.items():
                if pd.isna(val):
                    continue
                if col == target_col:
                    tag = f"Disease={val}"   # label asli, bukan di-discretize
                elif col in medians:
                    tag = self._tag(col, val, medians[col])
                else:
                    tag = f"{col}_{val}"
                tags.append(tag)
                if tag not in self.tid_lists:
                    self.tid_lists[tag] = set()
                self.tid_lists[tag].add(tid)

        # Filter ke frequent 1-itemsets
        freq1 = {k: v for k, v in self.tid_lists.items() if len(v) >= min_count}

        # ── 2-itemset via TID intersection ─────────────────────────
        self.pair_tids  = {}
        self.rules       = []
        self.disease_rules = []
        freq_keys = list(freq1.keys())

        for i in range(len(freq_keys)):
            for j in range(i + 1, len(freq_keys)):
                a, b = freq_keys[i], freq_keys[j]
                inter = freq1[a] & freq1[b]
                if len(inter) < min_count:
                    continue
                pair = frozenset([a, b])
                self.pair_tids[pair] = inter
                has_disease = a.startswith('Disease=') or b.startswith('Disease=')
                # Turunkan threshold untuk pasangan yang melibatkan Disease
                threshold = self.min_confidence * 0.5 if has_disease else self.min_confidence

                # a → b  (skip jika a adalah Disease tag)
                if not a.startswith('Disease='):
                    conf_ab = len(inter) / len(freq1[a])
                    if conf_ab >= threshold:
                        display_b = b.replace('Disease=', '') if b.startswith('Disease=') else b
                        rule = {
                            'antecedent': a,
                            'consequent': display_b,
                            'support':    round(len(inter) / n, 4),
                            'confidence': round(conf_ab, 4),
                            'lift':       round(conf_ab / (len(freq1[b]) / n), 4),
                            'tid_count':  len(inter),
                        }
                        self.rules.append(rule)
                        if has_disease:
                            self.disease_rules.append(rule)

                # b → a  (skip jika b adalah Disease tag)
                if not b.startswith('Disease='):
                    conf_ba = len(inter) / len(freq1[b])
                    if conf_ba >= threshold:
                        display_a = a.replace('Disease=', '') if a.startswith('Disease=') else a
                        rule = {
                            'antecedent': b,
                            'consequent': display_a,
                            'support':    round(len(inter) / n, 4),
                            'confidence': round(conf_ba, 4),
                            'lift':       round(conf_ba / (len(freq1[a]) / n), 4),
                            'tid_count':  len(inter),
                        }
                        self.rules.append(rule)
                        if has_disease:
                            self.disease_rules.append(rule)
        return self

    def predict_tid(self, input_tags):
        """
        input_tags : list/set of str, format "ColName_H" atau "ColName_L"
                     (sudah di-strip dari prefix "ColName=" oleh caller)
        Returns    : list of dict diurut confidence desc, hanya consequent
                     berupa label penyakit (Disease=...) jika tersedia.
        """
        input_set = set(input_tags)

        # Kumpulkan TID transaksi yang mengandung SEMUA item input
        matching_tids = None
        for tag in input_set:
            if tag in self.tid_lists:
                tids = self.tid_lists[tag]
                matching_tids = set(tids) if matching_tids is None else matching_tids & tids

        if not matching_tids:
            return []

        # Hitung co-occurrence: item di luar input yang muncul di TID cocok
        n = self.n_transactions
        cooc = {}
        for tag, tids in self.tid_lists.items():
            if tag in input_set:
                continue
            overlap = set(tids) & matching_tids
            if overlap:
                cooc[tag] = len(overlap)

        if not cooc:
            return []

        # Prioritaskan consequent berupa label penyakit (Disease=...)
        disease_cooc = {k: v for k, v in cooc.items() if k.startswith('Disease=')}
        candidate_cooc = disease_cooc if disease_cooc else cooc

        # Buat pseudo-rules dari co-occurrence
        results = []
        for cons, cnt in sorted(candidate_cooc.items(), key=lambda x: x[1], reverse=True):
            rel_sup  = cnt / n
            rel_conf = cnt / len(matching_tids)
            cons_sup = len(self.tid_lists.get(cons, set())) / n
            lift     = rel_conf / cons_sup if cons_sup > 0 else 1.0
            # Strip "Disease=" prefix untuk tampilan
            display_cons = cons.replace('Disease=', '') if cons.startswith('Disease=') else cons
            results.append({
                'antecedent':    ' + '.join(sorted(input_set)),
                'consequent':    display_cons,
                'support':       round(rel_sup, 4),
                'confidence':    round(rel_conf, 4),
                'lift':          round(lift, 4),
                'tid_count':     cnt,
                'matching_tids': len(matching_tids),
            })

        return results

    def get_summary(self):
        confs = [r['confidence'] for r in self.rules] if self.rules else [0]
        return {
            'total_patterns': len(self.tid_lists),
            'total_rules':    len(self.rules),
            'avg_support':    round(float(np.mean([r['support'] for r in self.rules])) if self.rules else 0, 4),
            'avg_confidence': round(float(np.mean(confs)), 4),
            'top_rules':      sorted(self.rules, key=lambda x: x['confidence'], reverse=True)[:5]
        }


# =====================================================================
# AIS ALGORITHM
# =====================================================================

class AISAlgorithm:
    def __init__(self, num_intervals=3, min_support=0.15, min_confidence=0.3):
        self.num_intervals = num_intervals
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.rules = []
        self.intervals = {}

    def discretize_continuous(self, df):
        df_d = df.copy()
        for col in df_d.select_dtypes(include=[np.number]).columns:
            labels = [f"{col}_L", f"{col}_M", f"{col}_H"]
            try:
                res, bins = pd.cut(df_d[col], bins=self.num_intervals, labels=labels, retbins=True)
                df_d[col] = res
                self.intervals[col] = bins.tolist()
            except:
                df_d[col] = labels[0]
                self.intervals[col] = []
        return df_d

    def generate_itemsets(self, df_d):
        n = len(df_d)
        min_count = int(np.ceil(self.min_support * n))

        # 1-itemsets
        items_count = {}
        for col in df_d.columns:
            for val in df_d[col].dropna().unique():
                key = f"{col}={val}"
                cnt = int((df_d[col] == val).sum())
                if cnt >= min_count:
                    items_count[key] = cnt

        itemsets = {1: items_count}

        # 2-itemsets: hitung support aktual dari data
        target_keys = [k for k in items_count if k.startswith("Disease=")]
        feat_keys   = [k for k in items_count if not k.startswith("Disease=")]

        def key_to_mask(key, df_d):
            col, val = key.split("=", 1)
            return df_d[col].astype(str) == str(val)

        itemsets_2 = {}

        # Fitur ↔ Disease (penting untuk prediksi label penyakit)
        for fk in feat_keys:
            for dk in target_keys:
                mask = key_to_mask(fk, df_d) & key_to_mask(dk, df_d)
                cnt = int(mask.sum())
                if cnt >= min_count:
                    itemsets_2[f"{fk} AND {dk}"] = cnt

        # Fitur ↔ Fitur
        for i in range(len(feat_keys)):
            for j in range(i+1, min(i+3, len(feat_keys))):
                mask = key_to_mask(feat_keys[i], df_d) & key_to_mask(feat_keys[j], df_d)
                cnt = int(mask.sum())
                if cnt >= min_count:
                    itemsets_2[f"{feat_keys[i]} AND {feat_keys[j]}"] = cnt

        if itemsets_2:
            itemsets[2] = itemsets_2
        return itemsets

    def generate_rules(self, itemsets, n, items_count):
        rules = []
        for level, items in itemsets.items():
            for itemset, support_count in items.items():
                if ' AND ' in itemset:
                    parts = itemset.split(' AND ')
                    for ant in parts:
                        cons = " AND ".join(p for p in parts if p != ant)
                        ant_count = items_count.get(ant.strip(), support_count)
                        conf = round(support_count / ant_count, 4) if ant_count > 0 else 0
                        cons_count = items_count.get(cons.strip(), 1)
                        cons_sup = cons_count / n if n > 0 else 1
                        if conf >= self.min_confidence:
                            rules.append({
                                'antecedent': ant.strip(),
                                'consequent': cons.strip(),
                                'support': round(support_count / n, 4),
                                'confidence': conf,
                                'lift': round(conf / cons_sup, 4) if cons_sup > 0 else 1.0
                            })
        return rules

    def fit(self, df, target_col):
        # Pisahkan fitur numerik dan target (jangan encode target)
        feat_cols = [c for c in df.columns if c != target_col]
        df_feats = df[feat_cols].copy()

        # Hanya impute fitur numerik, tanpa label-encode target
        num_cols = df_feats.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            imp = SimpleImputer(strategy='mean')
            df_feats[num_cols] = imp.fit_transform(df_feats[num_cols])

        # Gabungkan kembali dengan kolom target asli (string label)
        processed_df = df_feats.copy()
        if target_col in df.columns:
            processed_df[target_col] = df[target_col].values

        n = len(processed_df)
        df_d = self.discretize_continuous(processed_df)
        itemsets = self.generate_itemsets(df_d)
        items_count = itemsets.get(1, {})
        self.rules = self.generate_rules(itemsets, n, items_count)
        return self

    def predict(self, input_conditions):
        """Cocokkan input_conditions dengan rules, return best match."""
        matched = []
        for rule in self.rules:
            conds = [c.strip() for c in rule['antecedent'].split('AND')]
            if all(c in input_conditions for c in conds):
                matched.append(rule)
        if not matched:
            return None
        return sorted(matched, key=lambda x: x['confidence'], reverse=True)

    def get_summary(self):
        confs = [r['confidence'] for r in self.rules] if self.rules else [0]
        return {
            'num_intervals': self.num_intervals,
            'total_rules': len(self.rules),
            'avg_confidence': round(float(np.mean(confs)), 4),
            'top_rules': sorted(self.rules, key=lambda x: x['confidence'], reverse=True)[:5]
        }


# =====================================================================
# SET-ORIENTED MINING
# =====================================================================

class SetOrientedMiningModel:
    def __init__(self, min_support=0.2, min_confidence=0.5):
        self.min_support    = min_support
        self.min_confidence = min_confidence
        self.itemsets_count = 0
        self.rules          = []
        self.disease_rules  = []

    def fit(self, df, target_col):
        # ── Preprocessing: hanya fitur, bukan target ──────────────
        feat_cols = [c for c in df.columns if c != target_col]
        df_feats  = df[feat_cols].copy()
        num_cols  = df_feats.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            imp = SimpleImputer(strategy='mean')
            df_feats[num_cols] = imp.fit_transform(df_feats[num_cols])

        # Gabungkan kembali kolom target asli (string)
        processed_df = df_feats.copy()
        if target_col in df.columns:
            processed_df[target_col] = df[target_col].values

        n = len(processed_df)
        min_count = int(np.ceil(self.min_support * n))

        # ── Diskretisasi fitur → _HIGH/_LOW, target tetap string ──
        medians = {col: processed_df[col].median()
                   for col in feat_cols
                   if pd.api.types.is_numeric_dtype(processed_df[col])}

        def discretize_row(row):
            tags = []
            for col, val in row.items():
                if pd.isna(val):
                    continue
                if col == target_col:
                    tags.append(f"Disease={val}")
                elif col in medians:
                    tags.append(f"{col}_HIGH" if val >= medians[col] else f"{col}_LOW")
                else:
                    tags.append(f"{col}_{val}")
            return tags

        transactions = [discretize_row(row) for _, row in processed_df.iterrows()]

        # ── Bangun item counts (SQLite in-memory) ──────────────────
        conn = sqlite3.connect(':memory:')
        try:
            cur = conn.cursor()
            cur.execute('CREATE TABLE tid_items (tid INT, item TEXT)')
            for tid, t in enumerate(transactions):
                for item in set(t):
                    cur.execute('INSERT INTO tid_items VALUES (?,?)', (tid, item))
            cur.execute('CREATE INDEX idx_item ON tid_items(item)')
            conn.commit()

            # 1-itemset counts
            cur.execute('''
                SELECT item, COUNT(DISTINCT tid) as cnt
                FROM tid_items
                GROUP BY item
                HAVING cnt >= ?
                ORDER BY cnt DESC
            ''', (min_count,))
            freq_items = {row[0]: row[1] for row in cur.fetchall()}
            self.itemsets_count = len(freq_items)

            # Pisahkan fitur dan label
            disease_items = {k: v for k, v in freq_items.items() if k.startswith('Disease=')}
            feat_items    = {k: v for k, v in freq_items.items() if not k.startswith('Disease=')}

            self.rules        = []
            self.disease_rules = []

            # 2-itemset: fitur ↔ Disease (untuk prediksi label penyakit)
            # Turunkan threshold agar lebih banyak disease rules ter-generate
            disease_threshold = self.min_confidence * 0.5
            for fk, f_cnt in feat_items.items():
                for dk, d_cnt in disease_items.items():
                    cur.execute('''
                        SELECT COUNT(DISTINCT a.tid)
                        FROM tid_items a JOIN tid_items b ON a.tid = b.tid
                        WHERE a.item = ? AND b.item = ?
                    ''', (fk, dk))
                    pair_cnt = cur.fetchone()[0]
                    if pair_cnt < min_count:
                        continue
                    sup     = pair_cnt / n
                    conf_fd = pair_cnt / f_cnt
                    if conf_fd >= disease_threshold:
                        d_sup        = d_cnt / n
                        display_cons = dk.replace('Disease=', '') if dk.startswith('Disease=') else dk
                        rule = {
                            'antecedent': fk,
                            'consequent': display_cons,
                            'support':    round(sup, 4),
                            'confidence': round(conf_fd, 4),
                            'lift':       round(conf_fd / d_sup if d_sup > 0 else 1.0, 4),
                        }
                        self.rules.append(rule)
                        self.disease_rules.append(rule)

            # 2-itemset: fitur ↔ fitur (tetap berguna untuk rule umum)
            feat_list = list(feat_items.keys())
            for i in range(min(len(feat_list), 20)):
                for j in range(i + 1, min(i + 5, len(feat_list))):
                    ant, cons = feat_list[i], feat_list[j]
                    cur.execute('''
                        SELECT COUNT(DISTINCT a.tid)
                        FROM tid_items a JOIN tid_items b ON a.tid = b.tid
                        WHERE a.item = ? AND b.item = ?
                    ''', (ant, cons))
                    pair_cnt = cur.fetchone()[0]
                    if pair_cnt < min_count:
                        continue
                    sup  = pair_cnt / n
                    conf = pair_cnt / freq_items[ant]
                    if conf >= self.min_confidence:
                        cons_sup = freq_items[cons] / n
                        self.rules.append({
                            'antecedent': ant,
                            'consequent': cons,
                            'support':    round(sup, 4),
                            'confidence': round(conf, 4),
                            'lift':       round(conf / cons_sup if cons_sup > 0 else 1.0, 4),
                        })
        finally:
            conn.close()
        return self


    def get_summary(self):
        confs = [r['confidence'] for r in self.rules] if self.rules else [0]
        return {
            'total_itemsets': self.itemsets_count,
            'sql_queries': 3,
            'total_rules': len(self.rules),
            'avg_confidence': round(float(np.mean(confs)), 4),
            'top_rules': sorted(self.rules, key=lambda x: x['confidence'], reverse=True)[:5]
        }


# =====================================================================
# HELPER: PRIORITAS DISEASE RULES
# =====================================================================

DISEASE_LABELS = {'Bacterial leaf blight', 'Brown spot', 'Leaf smut'}

def pick_disease_prediction(model, input_tags, model_name=''):
    """
    Pilih rules prediksi yang consequent-nya adalah label penyakit.
    Selalu mengembalikan list terurut confidence DESC dengan consequent
    berupa nama penyakit (bukan nama fitur).

    Urutan prioritas:
      1. disease_rules (fitur → Disease) yang antecedent-nya ada di input_tags
      2. Semua disease_rules terurut confidence (tanpa filter input)
      3. Semua rules biasa yang consequent-nya label penyakit
      4. Fallback: semua rules yang antecedent-nya ada di input_tags

    Untuk AprioriTid: gunakan predict_tid() langsung karena scoring berbeda.
    """
    disease_labels = DISEASE_LABELS

    # AprioriTid — sudah mengembalikan list dengan Disease consequent
    if model_name == 'AprioriTid':
        results = model.predict_tid(input_tags)
        disease = [r for r in results if r['consequent'] in disease_labels]
        return disease if disease else results

    # Untuk Apriori dan SetOrientedMining
    disease_rules = getattr(model, 'disease_rules', [])

    # 1. disease_rules dengan antecedent cocok input
    matched_disease = sorted(
        [r for r in disease_rules if r['antecedent'] in input_tags],
        key=lambda x: x['confidence'], reverse=True
    )
    if matched_disease:
        return matched_disease

    # 2. Semua disease_rules (tanpa filter input) — voting berdasarkan confidence
    if disease_rules:
        return sorted(disease_rules, key=lambda x: x['confidence'], reverse=True)

    # 3. Fallback: rules biasa dengan consequent = penyakit
    from_all = sorted(
        [r for r in model.rules if r['consequent'] in disease_labels and r['antecedent'] in input_tags],
        key=lambda x: x['confidence'], reverse=True
    )
    if from_all:
        return from_all

    # 4. Last resort: semua rules yang antecedent cocok
    return sorted(
        [r for r in model.rules if r['antecedent'] in input_tags],
        key=lambda x: x['confidence'], reverse=True
    )


# =====================================================================
# DATA LOADING
# =====================================================================

def extract_image_features(image_path, size=(32, 32)):
    try:
        img = Image.open(image_path).convert('RGB').resize(size)
        a = np.array(img)
        return [a[:,:,0].mean(), a[:,:,1].mean(), a[:,:,2].mean(),
                a[:,:,0].std(),  a[:,:,1].std(),  a[:,:,2].std()]
    except:
        return [0, 0, 0, 0, 0, 0]

def load_rice_leaf_dataset():
    path = 'Dataset/Dataset1_Citra_RiceLeafDiseasesDataset'
    if not os.path.exists(path):
        return None, None
    data_list, labels = [], []
    for disease in ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']:
        dp = os.path.join(path, disease)
        if os.path.exists(dp):
            for f in os.listdir(dp):
                if f.lower().endswith(('.jpg','.png','.jpeg')):
                    data_list.append(extract_image_features(os.path.join(dp, f)))
                    labels.append(disease)
    if data_list:
        df = pd.DataFrame(data_list, columns=['R_mean','G_mean','B_mean','R_std','G_std','B_std'])
        df['Disease'] = labels
        return df, list(df.columns)
    return None, None

DATASET_PATHS = {
    'Dataset1': ('image', 'Dataset/Dataset1_Citra_RiceLeafDiseasesDataset'),
    'Dataset2': ('csv',   'Dataset/Dataset2_CSV_PlantGrowthDataClassification/plant_growth_data.csv'),
    'Dataset3': ('csv',   'Dataset/Dataset3_CSV_Agriculture&Farming/agriculture_dataset.csv'),
    'Dataset4': ('csv',   'Dataset/Dataset4_CSV_AgriSeshatAgricultureDataset/Agriculture.csv'),
}

DATASET_TARGETS = {
    'Dataset1': 'Disease',
    'Dataset2': 'Growth_Milestone',
    'Dataset3': 'Crop_Yield',
    'Dataset4': 'Value From',
}

def load_dataset(dataset_name):
    if dataset_name not in DATASET_PATHS:
        return None, None
    dtype, path = DATASET_PATHS[dataset_name]
    if dtype == 'image':
        return load_rice_leaf_dataset()
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    return df, list(df.columns)

def run_model(model_name, df, target_col):
    models = {
        'Apriori': AprioriModel,
        'AprioriTid': AprioriTidModel,
        'AgrawalImielinskiSwami': AISAlgorithm,
        'SetOrientedMining': SetOrientedMiningModel,
    }
    if model_name not in models:
        return {'status': 'error', 'message': f'Model {model_name} tidak dikenal'}
    try:
        m = models[model_name]()
        m.fit(df, target_col)
        summary = m.get_summary()
        return {'status': 'success', 'model': model_name,
                'data_shape': str(df.shape), **summary}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


# =====================================================================
# FLASK ROUTES
# =====================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    df, features = load_dataset(data['dataset'])
    if df is None:
        return jsonify({'status': 'error', 'message': f"Dataset {data['dataset']} tidak ditemukan. Pastikan folder Dataset ada di direktori yang sama dengan Web.py"})
    target = data.get('target') or DATASET_TARGETS.get(data['dataset'], features[0])
    return jsonify(run_model(data['model'], df, target))

@app.route('/compare', methods=['POST'])
def compare():
    data = request.get_json()
    df, features = load_dataset(data['dataset'])
    if df is None:
        return jsonify({'error': f"Dataset {data['dataset']} tidak ditemukan"})
    target = data.get('target') or DATASET_TARGETS.get(data['dataset'], features[0])
    results = {}
    for m in ['Apriori', 'AprioriTid', 'AgrawalImielinskiSwami', 'SetOrientedMining']:
        results[m] = run_model(m, df, target)
    return jsonify(results)

# ── /predict — input manual per dataset (semua model) ─────────────────
@app.route('/predict', methods=['POST'])
def predict():
    """
    Input manual per dataset untuk simulasi prediksi semua model.

    Format conditions per model:
      AIS               : ["G_mean=G_mean_L", "R_std=R_std_H"]       (_L/_M/_H, 3 bin via pd.cut)
      Apriori           : ["G_mean=G_mean_HIGH", "R_std=R_std_LOW"]   (_HIGH/_LOW, median split)
      AprioriTid        : ["G_mean=G_mean_H", "R_std=R_std_H"]        (_H/_L, median split)
                          ⚠️  BERBEDA dengan Apriori — pakai _H/_L bukan _HIGH/_LOW
                          Prediksi via TID co-occurrence scoring (bukan confidence rules)
      SetOrientedMining : ["G_mean=G_mean_HIGH", "R_std=R_std_LOW"]   (_HIGH/_LOW, sama dengan Apriori)
                          Antecedent tunggal, SQLite in-memory, confidence-based matching

    Ringkasan format:
      AIS            → _L / _M / _H   (3 bin)
      Apriori        → _HIGH / _LOW   (2 bin, median)
      AprioriTid     → _H  / _L       (2 bin, median — BEDA dari Apriori!)
      SetOriented    → _HIGH / _LOW   (2 bin, median — sama dengan Apriori)

    Body JSON:
      { "dataset": "Dataset1",
        "model": "SetOrientedMining",
        "conditions": ["G_mean=G_mean_LOW", "R_std=R_std_HIGH"] }
    """
    data = request.get_json()
    dataset_name = data.get('dataset', 'Dataset1')
    model_name   = data.get('model', 'AgrawalImielinskiSwami')
    conditions   = data.get('conditions', [])

    df, features = load_dataset(dataset_name)
    if df is None:
        return jsonify({'status': 'error', 'message': f'Dataset {dataset_name} tidak ditemukan'})

    target = data.get('target') or DATASET_TARGETS.get(dataset_name, features[0])

    # ── AIS ──────────────────────────────────────────────────────────
    if model_name == 'AgrawalImielinskiSwami':
        m = AISAlgorithm()
        m.fit(df, target)
        matched = m.predict(conditions)  # returns list or None

    # ── Apriori ──────────────────────────────────────────────────────
    elif model_name == 'Apriori':
        m = AprioriModel()
        m.fit(df, target)
        input_tags = set(c.split('=',1)[1] if '=' in c else c for c in conditions)
        matched = pick_disease_prediction(m, input_tags, 'Apriori')

    # ── AprioriTid ────────────────────────────────────────────────
    elif model_name == 'AprioriTid':
        m = AprioriTidModel()
        m.fit(df, target)
        input_tags = set(c.split('=',1)[1] if '=' in c else c for c in conditions)
        matched = pick_disease_prediction(m, input_tags, 'AprioriTid')

    # ── SetOrientedMining ─────────────────────────────────────────────
    elif model_name == 'SetOrientedMining':
        m = SetOrientedMiningModel()
        m.fit(df, target)
        input_tags = set(c.split('=',1)[1] if '=' in c else c for c in conditions)
        matched = pick_disease_prediction(m, input_tags, 'SetOrientedMining')

    else:
        return jsonify({'status': 'error', 'message': f'Model {model_name} tidak dikenal'})

    # ── Response ──────────────────────────────────────────────────────
    if not matched:
        tip_map = {
            'AgrawalImielinskiSwami': 'Coba ubah interval fitur (misalnya _L → _M atau _H)',
            'Apriori':                'Coba ubah interval fitur (misalnya _HIGH → _LOW atau sebaliknya)',
            'AprioriTid':             'Coba ubah interval fitur _H ↔ _L (AprioriTid pakai median split biner: _H / _L)',
            'SetOrientedMining':      'Coba ubah interval fitur (misalnya _HIGH → _LOW atau sebaliknya)',
        }
        return jsonify({
            'status': 'no_match',
            'message': 'Tidak ada rule yang cocok dengan kombinasi input tersebut.',
            'tip': tip_map.get(model_name, 'Coba kombinasi kondisi lain.')
        })

    best = matched[0]
    response = {
        'status':           'success',
        'dataset':          dataset_name,
        'model':            model_name,
        'input_conditions': conditions,
        'prediction':       best['consequent'],
        'confidence':       best['confidence'],
        'lift':             best['lift'],
        'support':          best['support'],
        'top_rules':        matched[:3],
    }
    # Field tambahan khusus AprioriTid
    if model_name == 'AprioriTid':
        response['tid_count']      = best.get('tid_count', 0)
        response['matching_tids']  = best.get('matching_tids', 0)
        response['score_method']   = 'TID co-occurrence'
    return jsonify(response)

@app.route('/predict_image', methods=['POST'])
def predict_image():
    """
    Terima gambar daun padi, ekstrak fitur RGB, prediksi penyakit
    via association rules murni (tanpa KNN).
    """
    import io, base64

    body_json = request.get_json(silent=True) or {}
    model_name = request.form.get('model') or body_json.get('model', 'AgrawalImielinskiSwami')

    # ── Ambil gambar ──────────────────────────────────────────────────
    try:
        if request.files.get('image'):
            img_bytes = request.files['image'].read()
        else:
            b64 = body_json.get('image_b64', '')
            if ',' in b64:
                b64 = b64.split(',', 1)[1]
            img_bytes = base64.b64decode(b64)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB').resize((32, 32))
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Gagal membaca gambar: {e}'})

    # ── Ekstrak fitur ─────────────────────────────────────────────────
    a = np.array(img)
    feats = {
        'R_mean': float(a[:,:,0].mean()),
        'G_mean': float(a[:,:,1].mean()),
        'B_mean': float(a[:,:,2].mean()),
        'R_std':  float(a[:,:,0].std()),
        'G_std':  float(a[:,:,1].std()),
        'B_std':  float(a[:,:,2].std()),
    }

    # ── Load dataset ──────────────────────────────────────────────────
    df, _ = load_dataset('Dataset1')
    if df is None:
        return jsonify({'status': 'error', 'message': 'Dataset1 tidak ditemukan di server.'})

    target = 'Disease'
    disease_labels = {'Bacterial leaf blight', 'Brown spot', 'Leaf smut'}

    # ── Fit model & buat conditions dari fitur gambar ─────────────────
    conditions = []
    matched = []

    if model_name == 'AgrawalImielinskiSwami':
        m = AISAlgorithm()
        m.fit(df, target)
        for col, val in feats.items():
            bins = m.intervals.get(col, [])
            if len(bins) >= 4:
                suffix = 'L' if val <= bins[1] else ('M' if val <= bins[2] else 'H')
            else:
                suffix = 'M'
            conditions.append(f"{col}={col}_{suffix}")
        all_matched = m.predict(conditions) or []
        # Ambil rule yang consequent-nya mengandung label penyakit
        # (format: 'Disease=Bacterial leaf blight')
        matched = [r for r in all_matched if any(d in r['consequent'] for d in disease_labels)]
        if not matched:
            matched = all_matched

    elif model_name in ('Apriori', 'SetOrientedMining'):
        ModelClass = AprioriModel if model_name == 'Apriori' else SetOrientedMiningModel
        m = ModelClass()
        m.fit(df, target)
        medians = df[['R_mean','G_mean','B_mean','R_std','G_std','B_std']].median()
        input_tags = set()
        for col, val in feats.items():
            suffix = 'HIGH' if val >= medians[col] else 'LOW'
            conditions.append(f"{col}={col}_{suffix}")
            input_tags.add(f"{col}_{suffix}")
        matched = pick_disease_prediction(m, input_tags, model_name)

    elif model_name == 'AprioriTid':
        m = AprioriTidModel()
        m.fit(df, target)
        medians = df[['R_mean','G_mean','B_mean','R_std','G_std','B_std']].median()
        input_tags = set()
        for col, val in feats.items():
            suffix = 'H' if val >= medians[col] else 'L'
            conditions.append(f"{col}={col}_{suffix}")
            input_tags.add(f"{col}_{suffix}")
        matched = pick_disease_prediction(m, input_tags, 'AprioriTid')

    else:
        return jsonify({'status': 'error', 'message': f'Model {model_name} tidak dikenal'})

    if not matched:
        return jsonify({
            'status': 'no_match',
            'message': 'Tidak ada rule yang cocok dengan fitur gambar tersebut.',
            'features': {k: round(v, 2) for k, v in feats.items()},
            'conditions': conditions,
        })

    best = matched[0]
    # Strip 'Disease=' prefix jika ada
    raw_pred = best['consequent']
    prediction = raw_pred.replace('Disease=', '') if raw_pred.startswith('Disease=') else raw_pred
    return jsonify({
        'status':     'success',
        'model':      model_name,
        'features':   {k: round(v, 2) for k, v in feats.items()},
        'conditions': conditions,
        'prediction': prediction,
        'confidence': best['confidence'],
        'lift':       best['lift'],
        'support':    best['support'],
        'top_rules':  matched[:3],
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Association Rule Mining — Web Interface")
    print("="*60)
    print("\n📍 Buka: http://localhost:5000")
    print("\n✨ Endpoints:")
    print("  GET  /                   → UI")
    print("  POST /analyze            → Analisis single model")
    print("  POST /compare            → Bandingkan 4 model")
    print("  POST /predict            → Simulasi input manual (JSON)")
    print("  POST /predict_notebook   → Simulasi input manual (plain-text notebook style)")
    print("")
    print("📐 Format interval per model:")
    print("  AIS          → _L / _M / _H  (3 bin, pd.cut)")
    print("  Apriori      → _HIGH / _LOW  (2 bin, median)")
    print("  AprioriTid   → _H  / _L      (2 bin, median — BEDA dari Apriori!)")
    print("  SetOriented  → _HIGH / _LOW  (2 bin, median — sama dengan Apriori)")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)


# ── /predict_notebook — output teks mirip notebook cell ───────────────
@app.route('/predict_notebook', methods=['POST'])
def predict_notebook():
    """
    Sama dengan /predict, tetapi response berupa plain-text formatted
    seperti output notebook cell — cocok untuk debugging di terminal.

    Body JSON: sama dengan /predict
    """
    import json as _json
    data = request.get_json()
    dataset_name = data.get('dataset', 'Dataset1')
    model_name   = data.get('model', 'AgrawalImielinskiSwami')
    conditions   = data.get('conditions', [])

    df, features = load_dataset(dataset_name)
    if df is None:
        return f"❌ Dataset {dataset_name} tidak ditemukan", 404, {'Content-Type': 'text/plain'}

    target = data.get('target') or DATASET_TARGETS.get(dataset_name, features[0])

    # Reuse existing logic
    resp = app.test_client().post('/predict',
        data=_json.dumps(data), content_type='application/json')
    result = _json.loads(resp.data)

    lines = []
    sep = "=" * 60
    fmt_map = {
        'AgrawalImielinskiSwami': '_L / _M / _H  (3 bin, pd.cut)',
        'Apriori':                '_HIGH / _LOW  (2 bin, median split)',
        'AprioriTid':             '_H / _L       (2 bin, median split · TID Score)',
        'SetOrientedMining':      '_HIGH / _LOW  (2 bin, median split · SQL Set)',
    }
    lines += [sep, f"🔍 {model_name} SIMULASI PREDIKSI — {dataset_name}", sep,
              f"📥 Input Kondisi : {conditions}",
              f"📐 Diskritisasi  : {fmt_map.get(model_name, '—')}", "-" * 60]

    if result.get('status') == 'success':
        lines += [
            f"✅ Hasil Prediksi Terkuat  : → {result['prediction']}",
            f"   Confidence              : {result['confidence']:.4f}",
            f"   Lift                    : {result['lift']:.4f}",
            f"   Support                 : {result['support']:.4f}",
        ]
        if model_name == 'AprioriTid':
            lines += [f"   TID Score               : {result.get('tid_count','—')}",
                      f"   Matching TIDs           : {result.get('matching_tids','—')}"]
        lines.append("")
        lines.append("📋 Top 3 Rules yang Cocok:")
        for i, r in enumerate(result.get('top_rules', [])[:3], 1):
            meta = f"Conf: {r['confidence']:.4f}  Lift: {r['lift']:.4f}  Sup: {r['support']:.4f}"
            if model_name == 'AprioriTid' and 'tid_count' in r:
                meta += f"  TID: {r['tid_count']}"
            lines += [f"  {i}. [{r['antecedent']}] => [{r['consequent']}]", f"     {meta}"]
    elif result.get('status') == 'no_match':
        lines += [f"⚠️  {result['message']}", f"   Tips: {result.get('tip','')}"]
    else:
        lines.append(f"❌ Error: {result.get('message','Unknown error')}")

    lines.append(sep)
    return '\n'.join(lines), 200, {'Content-Type': 'text/plain; charset=utf-8'}
