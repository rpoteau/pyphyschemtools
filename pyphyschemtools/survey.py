######################################################################
#                       Survey
######################################################################
from .visualID_Eng import fg, bg, hl
from .core import centerTitle, centertxt

import os, json, yaml, pandas as pd
from datetime import datetime
from IPython.display import display
from ipywidgets import VBox, HTML, Button, IntSlider, Text, Textarea, Layout, HBox, Dropdown
from textwrap import wrap
import numpy as np
import matplotlib.pyplot as plt

class SurveyApp:
    def __init__(self, mode="participant", base_dir="ML-survey"):
        self.mode = mode
        self.base_dir = base_dir
        self.responses_dir = os.path.join(base_dir, "responses")
        self.summary_dir = os.path.join(base_dir, "summary")
        os.makedirs(self.responses_dir, exist_ok=True)
        os.makedirs(self.summary_dir, exist_ok=True)
        self.questions, self.blocks = self.load_questions()

    def enable_slider_css(self):
        """Inject CSS for hover/active color effects on sliders."""
        from IPython.display import HTML, display
        display(HTML("""
        <style>
        /* Hover: track + rail */
        .jp-InputSlider:hover .MuiSlider-track,
        .jp-InputSlider:hover .MuiSlider-rail {
            background-color: #1E90FF55 !important;
        }
    
        /* Hover: thumb */
        .jp-InputSlider:hover .MuiSlider-thumb {
            background-color: #1E90FF !important;
            box-shadow: 0px 0px 4px #1E90FF !important;
        }
    
        /* Active: thumb when clicked or dragged */
        .jp-InputSlider .MuiSlider-thumb.Mui-active {
            background-color: #FF4500 !important;
            box-shadow: 0px 0px 6px #FF4500 !important;
        }
        </style>
        """))

    def get_or_create_user_id(self):
        """Return a persistent anonymous ID (stored in .survey_id)."""
        id_path = os.path.join(self.base_dir, ".survey_id")
    
        # If ID file already exists, read it
        if os.path.exists(id_path):
            with open(id_path, "r") as f:
                user_id = f.read().strip()
            if user_id:
                return user_id
    
        # Otherwise, create a new one
        import secrets
        user_id = f"UID_{datetime.now().strftime('%Y%m%d')}_{secrets.token_hex(3).upper()}"
        with open(id_path, "w") as f:
            f.write(user_id)
        return user_id

    def load_questions(self):
        yaml_path = os.path.join(self.base_dir, "survey_questions.yaml")
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
    
        questions, blocks = {}, {}
        
        for b, v in data["blocks"].items():
            blocks[b] = (v["title"], v["subtitle"])
            
            for qid, qinfo in v["questions"].items():
                questions[qid] = {
                    "text": qinfo["text"],
                    "required": qinfo.get("required", True)  # default = required
                }
                
        return questions, blocks

    # === Helper: Print Summary ===
    def print_questions_summary(self):
        """Affiche la liste des questions par bloc et leur type (Numérique/Texte)."""
        print("\n#####################################################")
        print("#         RÉPARTITION DES QUESTIONS PAR BLOC        #")
        print("#####################################################")

        num_total, text_total = 0, 0

        for block_id, (title, subtitle) in self.blocks.items():
            print(f"\n--- {block_id}. {title} ---")

            num_in_block, text_in_block = 0, 0

            # Filtre les questions appartenant à ce bloc
            block_questions = {
                qid: qinfo for qid, qinfo in self.questions.items() 
                if qid.startswith(block_id)
            }

            for qid, qinfo in block_questions.items():
                text = qinfo["text"]

                # Reproduction de la logique de détection des types
                if "(1 =" in text:
                    q_type = "NUMÉ(Slider)"
                    num_in_block += 1
                else:
                    q_type = "TEXTE(Libre)"
                    text_in_block += 1

                print(f"  [{qid:4}] {q_type:12} : {text.split('(1 =')[0].strip()}")

            num_total += num_in_block
            text_total += text_in_block

        print("\n-----------------------------------------------------")
        print(f"TOTAL : {num_total} questions numériques, {text_total} questions à champ libre.")
        print("-----------------------------------------------------")
    
    # === UI Builder ===
    def run(self):
        if self.mode == "participant":
            self.build_participant_form()
        elif self.mode == "admin":
            self.build_admin_dashboard()

    # === Participant Mode ===
    def build_participant_form(self):
        self.enable_slider_css()   # ← inject CSS automatically
        colors = ["#f7f9fc", "#f0f0f0"]
        base_styles = {
            "title": "font-size:18px;font-weight:bold;margin-top:5px;",
            "subtitle": "color:#444;font-style:italic;font-size:13px;margin-bottom:8px;",
            "warn": "color:#CC0000;font-size:12px;font-style:italic;",
        }
        
        self.user_id = self.get_or_create_user_id()
        self.full_form = [
            HTML(f"<b>🆔 Your anonymous ID:</b> <code>{self.user_id}</code><br>"
                 f"<span style='color:#777;font-size:12px'>(This ID is stored locally in a hidden file .survey_id)</span>")
        ]
        self.input_controls, self.warn_labels = [], []

        block_index = 0
        for block in self.blocks.keys():
            color = colors[block_index % len(colors)]
            title, subtitle = self.blocks[block]
            header_html = f"""
            <div style='background-color:{color};border:1px solid #ccc;border-radius:8px;padding:15px 20px;margin:12px 0'>
            <div style='{base_styles['title']}color:#1E90FF'>{title}</div>
            <div style='{base_styles['subtitle']}'>{subtitle}</div><div style='margin-left:15px;'>
            """
            footer_html = "</div></div>"
            block_widgets = [HTML(header_html)]
            for q, qinfo in self.questions.items():
                if q.startswith(block):      # ← IMPORTANT, à garder absolument
                    txt = qinfo["text"]
                    required = qinfo["required"]
                    
                    # Affichage + astérisque
                    star = "<span style='color:#a00'>*</span>" if required else ""
                    block_widgets.append(HTML(f"<b>{txt}</b> {star}"))
            
                    # Détection slider vs textarea (inchangée)
                    if "(1 =" in txt:
                        w = IntSlider(
                            value=0, min=0, max=5, step=1,
                            description='', layout=Layout(width="35%")
                        )
                        w.slider_behavior = "drag-tap"
                    else:
                        w = Textarea(
                            placeholder="Write your answer here...",
                            layout=Layout(width="85%", height="60px")
                        )
            
                    warn = HTML("")
            
                    # Stockage widget + required
                    self.input_controls.append((w, required))
                    self.warn_labels.append(warn)
            
                    # Ajout dans le layout
                    block_widgets.extend([w, warn])
            block_widgets.append(HTML(footer_html))
            self.full_form.extend(block_widgets)
            block_index += 1

        # === Buttons ===
        btn_layout = Layout(width="220px", height="40px", margin="3px 6px 3px 0")
        self.save_button = Button(description="💾 Save draft", button_style="info", layout=btn_layout)
        self.load_button = Button(description="📂 Load selected draft", button_style="warning", layout=btn_layout)
        self.submit_button = Button(description="✅ Submit", button_style="success", layout=btn_layout)
        self.status_label = HTML(value="", layout=Layout(margin="10px 0px"))
        self.draft_status_label = HTML(value="", layout=Layout(margin="5px 0px"))

        # === Dropdown to select which draft to load ===
        self.draft_dropdown = Dropdown(
            options=self.list_drafts(),
            description="Drafts:",
            layout=Layout(width="70%")
        )

        self.save_button.on_click(self.save_draft)
        self.load_button.on_click(self.load_draft)
        self.submit_button.on_click(self.submit_form)

        self.full_form.append(
            VBox([
                self.save_button,
                HBox([self.load_button, self.draft_dropdown]),           # ✅ ici à la place de self.load_button
                self.draft_status_label,
                self.submit_button,
                self.status_label
            ])
        )
        display(VBox(self.full_form))


    # === Helper: list available drafts ===
    def list_drafts(self):
        if not os.path.exists(self.responses_dir):
            return ["No drafts found"]
        drafts = sorted([f for f in os.listdir(self.responses_dir) if f.endswith(".json")])
        return ["Select a draft to load and then click on the Load Selected Draft button"] + drafts if drafts else ["No drafts found"]
    
    # === Actions ===
    def save_draft(self, b):
        data = self._collect_data()
        base_name = f"FallSchool_Draft_{self.user_id}"
        existing = [f for f in os.listdir(self.responses_dir) if f.startswith(base_name)]
        filename = os.path.join(self.responses_dir, f"{base_name}_v{len(existing)+1}.json")
        with open(filename, "w") as f: json.dump(data, f, indent=2)
        self.status_label.value = f"<div style='background:#fff4e5;color:#b35900;padding:6px;border:1px solid #b35900;border-radius:6px'>💾 Draft saved as <code>{os.path.basename(filename)}</code></div>"
        self.draft_dropdown.options = self.list_drafts()

    def load_draft(self, b):
        selected = self.draft_dropdown.value
        # --- Sécurité : rien sélectionné ou placeholder ---
        if not selected or selected.startswith("Select") or selected.startswith("No drafts"):
            self.status_label.value = (
                "<div style='color:#a00'>⚠ Please select a valid draft from the dropdown.</div>"
            )
            return
        filename = os.path.join(self.responses_dir, selected)

        with open(filename, "r") as f:
            data = json.load(f)

        if "id" in data:
            self.user_id = data["id"]
    
        for i, (q, _) in enumerate(self.questions.items()):
            if q in data:
                w, required = self.input_controls[i]
                val = data[q]
                if isinstance(w, IntSlider): w.value = int(val)
                else: w.value = str(val)
        self.status_label.value = (f"<div style='background:#fff4e5;color:#b35900;padding:6px;"
                                   f"border:1px solid #b35900;border-radius:6px'>📂 Loaded "
                                   f"{os.path.basename(filename)}</div>")

    def submit_form(self, b):
        incomplete = False
        data = {}
    
        for i, (q, _) in enumerate(self.questions.items()):
            w, required = self.input_controls[i]
            val = w.value
            warn_label = self.warn_labels[i]  # 🔴 label d’avertissement sous chaque question
    
            # --- Vérification des sliders ---
            if isinstance(w, IntSlider):
                if required and val == 0:
                    warn_label.value = (
                        "<span style='color:#a00;font-size:12px;font-style:italic;'>⚠ Please answer this question.</span>"
                    )
                    w.style.handle_color = "red"
                    incomplete = True
                else:
                    warn_label.value = ""
                    w.style.handle_color = None
                data[q] = int(val)
    
            # --- Vérification des champs texte ---
            else:
                if required and not str(val).strip():
                    warn_label.value = (
                        "<span style='color:#a00;font-size:12px;font-style:italic;'>⚠ Please provide an answer.</span>"
                    )
                    incomplete = True
                else:
                    warn_label.value = ""
                data[q] = val
    
        data["id"] = getattr(self, "user_id", "Anonymous")
    
        # === Si des réponses manquent ===
        if incomplete:
            self.status_label.value = (
                "<div style='background:#ffe6e6;color:#a00;border:1px solid #a00;"
                "padding:8px;border-radius:6px;'>❌ Some questions are missing. "
                "Please check the red warnings above.</div>"
            )
            return
    
        # === Si tout est rempli ===
        filename = os.path.join(
            self.responses_dir,
            f"Response_{data['id']}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        )
        pd.DataFrame([data]).to_csv(filename, index=False)
        self.status_label.value = (
            f"<div style='background:#e6ffe6;color:#060;border:1px solid #060;"
            f"padding:8px;border-radius:6px;'>✅ Response saved to "
            f"<code>{os.path.basename(filename)}</code></div>"
        )


    def _collect_data(self):
        data = {}
        for q, (w, required) in zip(self.questions.keys(), self.input_controls):
            data[q] = w.value
        data["id"] = self.user_id
        return data

    # === Admin mode ===================================================================================
    #=== Helper
    # === Admin mode ===================================================================================

    def plot_spider_multi(self, df, title="Participant and Mean Scores per Block", savepath=None, figsize=(12,8)):
        """
        Draw radar (spider) chart with per-participant transparency
        and block names instead of A–F.
        """
    
        # --- Compute averages ---
        avg = df.mean(axis=0)
        labels = avg.index.tolist()
        N = len(labels)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += [angles[0]]
    
        # === Replace A–F with block titles ===
        # → only use the first sentence (shortened title)
        label_map = {b: self.blocks[b][0].replace(f"Block {b}. ", "") for b in self.blocks.keys()}
        display_labels = [label_map.get(lbl, lbl) for lbl in labels]
        
        # === Auto linebreak: split labels into two roughly equal parts ===
        def split_label(text):
            words = text.split()
            if len(words) <= 2:
                return text
            mid = len(words) // 2
            return " ".join(words[:mid]) + "\n" + " ".join(words[mid:])    
            
        display_labels = [split_label(lbl) for lbl in display_labels]

        # --- Create figure ---
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
        # --- Plot all participants ---
        for i in range(len(df)):
            values = df.iloc[i].values.tolist()
            values += [values[0]]
            ax.plot(angles, values, linewidth=1, alpha=0.25, color="gray")
            ax.fill(angles, values, alpha=0.05, color="gray")
    
        # --- Mean polygon ---
        mean_values = avg.values.tolist() + [avg.values[0]]
        ax.plot(angles, mean_values, color='navy', linewidth=2.5)
        ax.fill(angles, mean_values, color='navy', alpha=0.25)
    
        # --- Axis style ---
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(display_labels, fontsize=11, fontweight='bold', wrap=True)
        ax.set_yticks([1,2,3,4,5])
        ax.set_yticklabels(["1","2","3","4","5"], fontsize=10, fontweight='bold', color="gray")
        ax.set_ylim(0,5)
        ax.set_title(title, size=14, weight='bold', pad=25)

        # --- Grid and outer circle ---
        ax.grid(True, linestyle='--', color='gray', alpha=0.4, linewidth=0.8)
        ax.spines['polar'].set_visible(False)  # remove the black frame
        outer_circle = plt.Circle((0,0), 5, transform=ax.transData._b, fill=False, lw=5, color="red", alpha=0.4)
        ax.add_artist(outer_circle)
        
        plt.tight_layout()
    
        # --- Save plot if requested ---
        if savepath:
            plt.savefig(savepath, dpi=300, bbox_inches='tight')
            print(f"💾 Saved radar plot to {savepath}")
    
        plt.show()

    def summarize_by_block(self, df):
        """Compute average score per block (A–F) for numeric questions."""
        import re
        num_df = df.select_dtypes(include=["number"])
        block_means = {}
        for col in num_df.columns:
            match = re.match(r"([A-F])\d+", col)
            if match:
                block = match.group(1)
                block_means.setdefault(block, []).append(num_df[col])
        # Mean per block (ignores missing NaN)
        block_avg = {b: pd.concat(cols, axis=1).mean(axis=1) for b, cols in block_means.items()}
        return pd.DataFrame(block_avg)
    

    ############################################################
    # 🔍 TEXTUAL & SEMANTIC ANALYSIS METHODS
    ############################################################

    def load_all_responses(self):
        """Load and merge all .csv survey responses into a DataFrame."""
        import pandas as pd, os
        files = [f for f in os.listdir(self.responses_dir) if f.endswith(".csv")]
        if not files:
            print("⚠ No responses found.")
            return None
        df = pd.concat([pd.read_csv(os.path.join(self.responses_dir, f)) for f in files], ignore_index=True)
        df.reset_index(drop=True, inplace=True)
        print(f"✅ Loaded {len(df)} responses ({len(df.columns)} columns)")
        return df


    def analyze_text_columns(self, df=None, columns=None, top_n=20):
        """
        Basic textual analysis: show frequent words, word clouds, and per-question summary.
        """
        import matplotlib.pyplot as plt
        from sklearn.feature_extraction.text import CountVectorizer
        from wordcloud import WordCloud
        import pandas as pd
        import os

        if df is None:
            df = self.load_all_responses()
        if df is None:
            return

        # auto-detect textual columns if not provided
        if columns is None:
            columns = [c for c in df.columns if df[c].dtype == 'object']
        if not columns:
            print("⚠ No text columns found.")
            return

        os.makedirs(self.summary_dir, exist_ok=True)
        print(f"🧩 Textual questions detected: {columns}")

        for col in columns:
            texts = df[col].dropna().astype(str)
            if len(texts) == 0:
                continue

            # vectorize text
            vectorizer = CountVectorizer(stop_words='english')
            X = vectorizer.fit_transform(texts)
            word_freq = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out()).sum().sort_values(ascending=False)

            # show top words
            print(f"\n📝 Top {top_n} words for '{col}':")
            display(word_freq.head(top_n))

            # wordcloud
            wc = WordCloud(width=800, height=400, background_color='white').generate(" ".join(texts))
            plt.figure(figsize=(8, 4))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            plt.title(f"Word Cloud: {col}")
            savepath = os.path.join(self.summary_dir, f"WordCloud_{col}.png")
            plt.savefig(savepath, dpi=300, bbox_inches="tight")
            print(f"💾 Saved {savepath}")
            plt.show()


    def semantic_analysis(self, df=None, columns=None, n_clusters=3):
        """
        Perform semantic clustering on open-ended responses using sentence-transformers.
        """
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import KMeans
        import matplotlib.pyplot as plt
        import umap
        import numpy as np
        import os

        if df is None:
            df = self.load_all_responses()
        if df is None:
            return

        if columns is None:
            columns = [c for c in df.columns if df[c].dtype == 'object']
        texts = []
        meta = []
        for col in columns:
            for t in df[col].dropna():
                texts.append(str(t))
                meta.append(col)

        if len(texts) < 2:
            print("⚠ Not enough text to perform semantic analysis.")
            return

        print(f"🧠 Encoding {len(texts)} responses from {len(columns)} text questions...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(texts)

        reducer = umap.UMAP(random_state=0)
        emb_2d = reducer.fit_transform(embeddings)

        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        labels = kmeans.fit_predict(embeddings)

        plt.figure(figsize=(8, 6))
        plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
        plt.title("Semantic Clusters of Open Responses", fontsize=14, weight='bold')
        for i, (x, y) in enumerate(emb_2d):
            plt.text(x, y, meta[i], fontsize=8, alpha=0.6)
        plt.tight_layout()

        savepath = os.path.join(self.summary_dir, "SemanticClusters.png")
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
        print(f"💾 Saved semantic clustering plot to {savepath}")
        plt.show()


    def build_admin_dashboard(self):

        # === APPEL AJOUTÉ ICI ! ===
        self.print_questions_summary()

        # === Load all responses ===
        df = self.load_all_responses()
        if df is None:
            return

        # --- CODE POUR SAUVEGARDER EN EXCEL ---
        excel_path = os.path.join(self.summary_dir, f"All_Responses_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx")
        df.to_excel(excel_path, index=False)
        print(f"✅ Saved all responses to Excel: {excel_path}")
        # ---------------------------------------------

        display(HTML("<h4>📊 All collected responses</h3>"))
        display(df)

        # === Summary statistics ===
        display(HTML("<h4>📈 Summary statistics</h4>"))
        display(df.describe())

        # === Missing values report ===
        html_summary = "<h4>🕳 Missing values per column:</h4><div style='font-family:monospace;font-size:14px;'>"
        missing = df.isna().sum()
        for col, val in missing.items():
            if val > 0:
                html_summary += f"<span style='color:red;font-weight:bold;'>{col}={val}</span> | "
            else:
                html_summary += f"{col}=0 | "
        html_summary = html_summary.rstrip(" | ") + "</div>"
        display(HTML(html_summary))

        # === 🧩 Textual analysis ===
        text_cols = [c for c in df.columns if df[c].dtype == 'object' and c not in ['id']]
        if text_cols:
            display(HTML("<h4>🧠 Textual Analysis</h4>"))
            self.analyze_text_columns(df=df, columns=text_cols, top_n=15)
        else:
            print("ℹ️ No open-ended text columns found for analysis.")

        # 🕸 Radar plot
        block_avg_df = self.summarize_by_block(df)
        self.plot_spider_multi(
            block_avg_df,
            title="",
            savepath=os.path.join(self.summary_dir, "Radar_BlockScores.png")
        )

        # === 🧭 Semantic map of text answers ===
        display(HTML("<h4>🧭 Semantic Clustering Map</h4>"))
        try:
            self.semantic_analysis(df=df, columns=text_cols, n_clusters=4)
        except Exception as e:
            print(f"⚠️ Skipped semantic clustering (reason: {e})")

        display(HTML(
            "<h4>✅ Dashboard summary saved in:</h4>"
            f"<code>{os.path.abspath(self.summary_dir)}</code>"
        ))

