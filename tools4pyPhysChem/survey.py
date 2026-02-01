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

############################################################
#                       Absorption spectra
############################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc

class SpectrumSimulator:

    def __init__(self, sigma_ev=0.3, plotWH=(12,8), \
                 fontSize_axisText=14, fontSize_axisLabels=14, fontSize_legends=12,
                 fontsize_peaks=12,
                 colorS='#3e89be',colorVT='#469cd6'
                ):
        """
        Initializes the spectrum simulator

        Args:
            - sigma_ev (float): Gaussian half-width at half-maximum in electron-volts (eV).
                                Default is 0.3 eV (GaussView default is 0.4 eV).
            - plotWH (tuple(int,int)): Width and Height of the matplotlib figures in inches. Default is (12,8).
            - colorS: color of the simulated spectrum (default ='#3e89be')
            - colorVT: color of the vertical transition line (default = '#469cd6')

        Returns:
            None: This method initializes the instance attributes.
        Calculates:
            - sigmanm = half-width of the Gaussian band, in nm
        """
        self.sigma_ev = sigma_ev
        # Conversion constante eV -> nm sigma
        self.ev2nm_const = (sc.h * sc.c) * 1e9 / sc.e
        self.sigmanm = self.ev2nm_const / self.sigma_ev
        self.plotW = plotWH[0]
        self.plotH = plotWH[1]
        self.colorS = colorS
        self.colorVT = colorVT
        self.fig = None
        self.graph = None
        self.fontSize_axisText = fontSize_axisText
        self.fontSize_axisLabels = fontSize_axisLabels
        self.fontSize_legends = fontSize_legends
        self.fontsize_peaks = fontsize_peaks

        print(f"sigma = {sigma_ev} eV -> sigmanm = {self.sigmanm:.1f} nm")
    
    def _initializePlot(self):
        fig, graph = plt.subplots(figsize=(self.plotW,self.plotH))
        plt.subplots_adjust(wspace=0)
        plt.xticks(fontsize=self.fontSize_axisText,fontweight='bold')
        plt.yticks(fontsize=self.fontSize_axisText,fontweight='bold')
        return fig, graph
    
    def _calc_epsiG(self,lambdaX,lambdai,fi):
        '''
        calculates a Gaussian band shape around a vertical transition
        input:
            - lambdaX = wavelength variable, in nm
            - lambdai = vertical excitation wavelength for i_th state, in nm
            - fi = oscillator strength for state i (dimensionless)
        output :
            molar absorption coefficient, in L mol-1 cm-1
        '''
        import scipy.constants as sc
        import numpy as np
        c = sc.c*1e2 #cm-1
        NA = sc.N_A #mol-1
        me = sc.m_e*1000 #g
        e = sc.e*sc.c*10 #elementary charge in esu 
        pf = np.sqrt(np.pi)*e**2*NA/(1000*np.log(10)*c**2*me)
        nubarX = 1e7 / lambdaX # nm to cm-1
        nubari = 1e7 / lambdai
        sigmabar = 1e7 / self.sigmanm
        epsi = pf * (fi / sigmabar) * np.exp(-((nubarX - nubari)/sigmabar)**2)
        return epsi
    
    def _Absorbance(self,eps,opl,cc):
        '''
        Calculates the Absorbance with the Beer-Lambert law
        input:
            - eps = molar absorption coefficient, in L mol-1 cm-1
            - opl = optical path length, in cm
            - cc = concentration of the attenuating species, in mol.L-1
        output :
            Absorbance, A (dimensionless)
        '''
        return eps*opl*cc
    
    def _sumStatesWithGf(self,wavel,wavelTAB,feTAB):
        '''
        '''
        import numpy as np
        sumInt = np.zeros(len(wavel))
        for l in wavel:
            for i in range(len(wavelTAB)):
                sumInt[np.argwhere(l==wavel)[0][0]] += self._calc_epsiG(l,wavelTAB[i],feTAB[i])
        return sumInt
    
    def _FindPeaks(self,sumInt,height,prom=1):
        '''
        Finds local maxima within the spectrum based on height and prominence.
        
        Prominence is crucial when switching between linear and logarithmic scales:
        - In Linear mode: A large prominence (e.g., 1 to 1000) filters out noise.
        - In Log mode: Data is compressed into a range of ~0 to 5. A large 
          prominence will 'hide' real peaks. A smaller value (0.01 to 0.1) 
          is required to detect shoulders and overlapping bands.

        Input:
            - sumInt: Array of intensities (Epsilon or Absorbance).
            - height: Minimum height a peak must reach to be considered.
            - prom: Required vertical distance between the peak and its lowest contour line.
        
        Returns:
            - PeakIndex: Indices of the detected peaks in the wavelength array.
            - PeakHeight: The intensity values at these peak positions.
        '''
        from scipy.signal import find_peaks
        peaks = find_peaks(sumInt, height = height, threshold = None, distance = 1, prominence=prom)
        PeakIndex = peaks[0]
        # Check if 'peak_heights' exists in the properties dictionary
        if 'peak_heights' in peaks[1]:
            PeakHeight = peaks[1]['peak_heights']
        else:
            # If height=None, we extract values manually from the input data
            PeakHeight = sumInt[PeakIndex]
        return PeakIndex,PeakHeight

    def _FindShoulders(self, data, tP):
        '''
        ###not working
        Detects shoulders using the second derivative.
        A shoulder appears as a peak in the negative second derivative.
        
        Note on scales:
        - If ylog is True: data should be log10(sumInt) and tP should be log10(tP).
          The second derivative on log data is much more sensitive to subtle 
          inflection points in weak transitions (like n -> pi*).
        - If ylog is False: data is linear sumInt and tP is linear.
        
        Returns:
            - shoulder_idx (ndarray): Array of indices where shoulders were found.
            - shoulder_heights (ndarray): The intensity values at these positions 
              extracted from the input data.
        '''
        import numpy as np
        # Calculate the second derivative (rate of change of the slope)
        d2 = np.gradient(np.gradient(data))
        
        # We search for peaks in the opposite of the second derivative (-d2).
        # A local maximum in -d2 corresponds to a point of maximum curvature 
        # (inflection), which identifies a shoulder.
        # We use a very low prominence threshold to capture subtle inflections.
        shoulder_idx, _ = self._FindPeaks(-d2, height=None, prom=0.0001)
        shoulder_heights = data[shoulder_idx]
        print(shoulder_idx, shoulder_heights )
        
        return shoulder_idx, shoulder_heights 
    
    def _pickPeak(self,wavel,peaksIndex,peaksH,color,\
                  shift=500,height=500,posAnnotation=200, ylog=False):
        '''
        Annotates peaks with a small vertical tick and the wavelength value.
        Adjusts offsets based on whether the plot is in log10 scale or linear.
        In log mode, peaksH must already be log10 values.
        '''
        s=shift
        h=height
        a=posAnnotation
        

        for i in range(len(peaksIndex)):
            x = wavel[peaksIndex[i]]
            y = peaksH[i]
            if ylog:
                # In log scale, we use multipliers to keep the same visual distance
                # 1.1 means "10% above the peak"
                # Adjust these factors based on your preference
                y_s = y * 1.1
                y_h = y * 1.3
                y_a = y * 1.5
                self.graph.vlines(x, y_s, y_h, colors=color, linestyles='solid')
                self.graph.annotate(f"{x:.1f}",xy=(x,y),xytext=(x,y_a),rotation=90,size=self.fontsize_peaks,ha='center',va='bottom', color=color)
            else:
                # Classic linear offsets
                self.graph.vlines(x, y+s, y+s+h, colors=color, linestyles='solid')
                self.graph.annotate(f"{x:.1f}",xy=(x,y),xytext=(x,y+s+h+a),rotation=90,size=self.fontsize_peaks,ha='center',va='bottom',color=color)
        return

    def _setup_axes(self, lambdamin, lambdamax, ymax, ylabel="Absorbance"):
            self.graph.set_xlabel('wavelength / nm', size=self.fontSize_axisLabels, fontweight='bold', color='#2f6b91')
            self.graph.set_ylabel(ylabel, size=self.fontSize_axisLabels, fontweight='bold', color='#2f6b91')
            self.graph.set_xlim(lambdamin, lambdamax)
            self.graph.set_ylim(0, ymax)
            self.graph.tick_params(axis='both', labelsize=self.fontSize_axisText,labelcolor='black')
            for tick in self.graph.xaxis.get_majorticklabels(): tick.set_fontweight('bold') #it is both powerful
                                                # (you can specify the type of a specific tick) and annoying
            for tick in self.graph.yaxis.get_majorticklabels(): tick.set_fontweight('bold')
    
    def plotTDDFTSpectrum(self,wavel,sumInt,wavelTAB,feTAB,tP,ylog,labelSpectrum,colorS='#0000ff',colorT='#0000cf'):
        
        '''
        Called by plotEps_lambda_TDDFT. Plots a single simulated UV-Vis spectrum, i.e. after
        gaussian broadening, together with the TDDFT vertical transitions (i.e. plotted as lines)
        
        input:
            - wavel = array of gaussian-broadened wavelengths, in nm
            - sumInt = corresponding molar absorptiopn coefficients, in L. mol-1 cm-1
            - wavelTAB = wavelength of TDDFT, e.g. discretized, transitions
            - ylog = log plot of epsilon
            - tP: threshold for finding the peaks
            - feTAB = TDDFT oscillator strength for each transition of wavelTAB
            - labelSpectrum = title for the spectrum
        '''

        # # --- DEBUG START ---
        # if ylog:
        #     print(f"\n--- DEBUG LOG MODE ---")
        #     print(f"Max sumInt (linear): {np.max(sumInt):.2f}")
        #     print(f"Max sumInt (log10):  {np.log10(max(np.max(sumInt), 1e-5)):.2f}")
        # # --- DEBUG END ---
        if ylog:
            # Apply safety floor to the entire array
            self.graph.set_yscale('log')
            ymin_val = 1.0 # Epsilon = 1
        else:
            self.graph.set_yscale('linear')
            ymin_val = 0
            
        # vertical lines
        for i in range(len(wavelTAB)):
            val_eps = self._calc_epsiG(wavelTAB[i],wavelTAB[i],feTAB[i])
            self.graph.vlines(x=wavelTAB[i], ymin=ymin_val, ymax=max(val_eps, ymin_val), colors=colorT)

        self.graph.plot(wavel,sumInt,linewidth=3,linestyle='-',color=colorS,label=labelSpectrum)

        self.graph.legend(fontsize=self.fontSize_legends)
        if ylog:
            # Use log-transformed data and log-transformed threshold
            # Clipping tP to 1e-5 ensures we don't take log of 0 or negative
            tPlog = np.log10(max(tP, 1e-5))
            # prom=0.05 allows detection of peaks that are close in log-magnitude
            peaks, peaksH_log = self._FindPeaks(np.log10(np.clip(sumInt, 1e-5, None)), tPlog, prom=0.05)
            peaksH = 10**peaksH_log
            # shoulders, shouldersH_log = self._FindShoulders(np.log10(np.clip(sumInt, 1e-5, None)), tPlog)
            # all_idx = np.concatenate((peaks, shoulders))
            # allH_log = np.concatenate((peaksH_log, shouldersH_log))
            # allH = 10**allH_log
        else:
            peaks, peaksH = self._FindPeaks(sumInt,tP)
            # shoulders, shouldersH = self._FindShoulders(wavel, sumInt, tP)
            # all_idx = np.concatenate((peaks, shoulders))
            # allH = np.concatenate((peaksH, shouldersH))
        self._pickPeak(wavel,peaks,peaksH,colorS,500,500,200,ylog)
        
    
    def plotEps_lambda_TDDFT(self,datFile,lambdamin=200,lambdamax=800,\
                             epsMax=None, titles=None, tP = 10, \
                             ylog=False,\
                             filename=None):
        '''
        Plots a TDDFT VUV simulated spectrum (vertical transitions and transitions summed with gaussian functions)
        between lambdamin and lambdamax (sum of states done in the range [lambdamin-50, lambdamlax+50] nm)
        input:
            - datFile: list of pathway/names to "XXX_ExcStab.dat" files generated by 'GParser Gaussian.log -S'
            - lambdamin, lambdamax: plot range
            - epsMax: y axis graph limit
            - titles: list of titles (1 per spectrum plot)
            - tP: threshold for finding the peaks (default = 10 L. mol-1 cm-1)
            - ylog: y logarithmic axis (default: False).
            - save: saves in a png file (300 dpi) if True (default = False)
            - filename: saves figure in a 300 dpi png file if not None (default), with filename=full pathway
        '''
        if self.fig is not None:
            graph = self.graph
            fig = self.fig
            lambdamin = self.lambdamin
            lambdamax = self.lambdamax
            epsMax = self.epsMax
        else:
            fig, graph = self._initializePlot()

        graph.set_prop_cycle(None)

        if self.fig is None:
            self.fig = fig
            self.graph = graph
            self.lambdamin = lambdamin
            self.lambdamax = lambdamax
            self.epsMax = epsMax
            
            graph.set_xlabel('wavelength / nm',size=self.fontSize_axisLabels,fontweight='bold',color='#2f6b91')

            graph.set_xlim(lambdamin,lambdamax)

            import matplotlib.ticker as ticker
            graph.xaxis.set_major_locator(ticker.MultipleLocator(50)) # sets a tick for every integer multiple of the base (here 250) within the view interval
        
        istate,state,wavel,fe,SSq = np.genfromtxt(datFile,skip_header=1,dtype="<U20,<U20,float,float,<U20",unpack=True)
        wavel = np.array(wavel)
        fe = np.array(fe)
        if wavel.size == 1:
            wavel = np.array([wavel])
            fe = np.array([fe])
        wvl = np.arange(lambdamin-50,lambdamax+50,1)
        sumInt = self._sumStatesWithGf(wvl,wavel,fe)
        self.plotTDDFTSpectrum(wvl,sumInt,wavel,fe,tP,ylog,titles,self.colorS,self.colorVT)
        if ylog:
            graph.set_ylabel('log(molar absorption coefficient / L mol$^{-1}$ cm$^{-1})$',size=self.fontSize_axisLabels,fontweight='bold',color='#2f6b91')
            graph.set_ylim(1, epsMax * 5 if epsMax else None)
        else:
            graph.set_yscale('linear')
            graph.set_ylabel('molar absorption coefficient / L mol$^{-1}$ cm$^{-1}$',size=self.fontSize_axisLabels,fontweight='bold',color='#2f6b91')
            graph.set_ylim(0, epsMax if epsMax else np.max(sumInt)*1.18)
        if filename is not None: self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

        peaksI, peaksH = self._FindPeaks(sumInt,tP)
        print(f"{bg.LIGHTREDB}{titles}{bg.OFF}")
        for i in range(len(peaksI)):
            print(f"peak {i:3}. {wvl[peaksI[i]]:4} nm. epsilon_max = {peaksH[i]:.1f} L mol-1 cm-1")
        if ylog:
            print()
            # prom=0.05 allows detection of peaks that are close in log-magnitude
            peaksI, peaksH = self._FindPeaks(np.log10(np.clip(sumInt, 1e-5, None)), np.log10(max(tP, 1e-5)), prom=0.05)
            for i in range(len(peaksI)):
                print(f"peak {i:3}. {wvl[peaksI[i]]:4} nm. log10(epsilon_max) = {peaksH[i]:.1f}")

    def plotAbs_lambda_TDDFT(self, datFiles=None, C0=1e-5, lambdamin=200, lambdamax=800, Amax=2.0,\
                             titles=None, linestyles=[], annotateP=[], tP = 0.1,\
                             resetColors=False,\
                             filename=None):
        '''
        Plots a simulated TDDFT VUV absorbance spectrum (transitions summed with gaussian functions)
        between lambdamin and lambdamax (sum of states done in the range [lambdamin-50, lambdamlax+50] nm)
        input:
            - datFiles: list of pathway/name to files generated by 'GParser Gaussian.log -S'
            - C0: list of concentrations needed to calculate A = epsilon x l x c (in mol.L-1)
            - lambdamin, lambdamax: plot range (x axis)
            - Amax: y axis graph limit
            - titles: list of titles (1 per spectrum plot)
            - linestyles: list of line styles(default = "-", i.e. a continuous line)
            - annotateP: list of Boolean (annotate lambda max True or False. Default = True)
            - tP: threshold for finding the peaks (default = 0.1)
            - resetColors (bool): If True, resets the matplotlib color cycle 
                                 to the first color. This allows different series 
                                 (e.g., gas phase vs. solvent) to share the same 
                                 color coding for each molecule across multiple calls. Default: False
            - save: saves in a png file (300 dpi) if True (default = False)
            - filename: saves figure in a 300 dpi png file if not None (default), with filename=full pathway
        '''

        if self.fig is None:
            fig, graph = self._initializePlot()
            self.fig = fig
            self.graph = graph
            self.lambdamin = lambdamin
            self.lambdamax = lambdamax
            self.Amax = Amax
        else:
            graph = self.graph
            fig = self.fig
            lambdamin = self.lambdamin
            lambdamax = self.lambdamax
            Amax = self.Amax
            if resetColors: graph.set_prop_cycle(None)
            
        if linestyles == []: linestyles = len(datFiles)*['-']
        if annotateP == []: annotateP = len(datFiles)*[True]

        self._setup_axes(lambdamin, lambdamax, self.Amax, ylabel="Absorbance")
        
        wvl = np.arange(lambdamin-50,lambdamax+50,1)
        for f in range(len(datFiles)):
            istate,state,wavel,fe,SSq = np.genfromtxt(datFiles[f],skip_header=1,dtype="<U20,<U20,float,float,<U20",unpack=True)
            sumInt = self._sumStatesWithGf(wvl,wavel,fe)
            Abs = self._Absorbance(sumInt,1,C0[f])
            plot=self.graph.plot(wvl,Abs,linewidth=3,linestyle=linestyles[f],label=f"{titles[f]}. TDDFT ($C_0$={C0[f]} mol/L)")
            peaksI, peaksH = self._FindPeaks(Abs,tP,0.01)
            if (annotateP[f]): self._pickPeak(wvl,peaksI,peaksH,plot[0].get_color(),0.01,0.04,0.02)
            print(f"{bg.LIGHTREDB}TDDFT. {titles[f]}{bg.OFF}")
            for i in range(len(peaksI)):
                print(f"peak {i:3}. {wvl[peaksI[i]]:4} nm. A = {peaksH[i]:.2f}")
                
        self.graph.legend(fontsize=self.fontSize_legends)

        if filename is not None: self.fig.savefig(filename, dpi=300, bbox_inches='tight')

        return
    
    def plotAbs_lambda_exp(self, csvFiles, C0, lambdamin=200, lambdamax=800,\
                             Amax=2.0, titles=None, linestyles=[], annotateP=[], tP = 0.1,\
                             filename=None):
        '''
        Plots an experimental VUV absorbance spectrum read from a csv file between lambdamin and lambdamax
        input:
            - superpose: False = plots a new graph, otherwise the plot is superposed to a previously created one
                         (probably with plotAbs_lambda_TDDFT())
            - csvfiles: list of pathway/name to experimental csvFiles (see examples for the format)
            - C0: list of experimental concentrations, i.e. for each sample
            - lambdamin, lambdamax: plot range (x axis)
            - Amax: graph limit (y axis)
            - titles: list of titles (1 per spectrum plot)
            - linestyles: list of line styles(default = "--", i.e. a dashed line)
            - annotateP: list of Boolean (annotate lambda max True or False. Default = True)
            - tP: threshold for finding the peaks (default = 0.1)
            - save: saves in a png file (300 dpi) if True (default = False)
            - filename: saves figure in a 300 dpi png file if not None (default), with filename=full pathway
        '''
        if linestyles == []: linestyles = len(csvFiles)*['--']
        if annotateP == []: annotateP = len(csvFiles)*[True]

        if self.fig is not None:
            graph = self.graph
            fig = self.fig
            lambdamin = self.lambdamin
            lambdamax = self.lambdamax
            Amax = self.Amax
        else:
            fig, graph = self._initializePlot()
            
        graph.set_prop_cycle(None)
        
        if self.fig is None:
            self.graph = graph
            self.fig = fig
            self.lambdamin = lambdamin
            self.lambdamax = lambdamax
            self.Amax = Amax
            
        self._setup_axes(lambdamin, lambdamax, self.Amax, ylabel="Absorbance")
                
        for f in range(len(csvFiles)):
            wavel,Abs = np.genfromtxt(csvFiles[f],skip_header=1,unpack=True,delimiter=";")
            wavel *= 1e9
            plot=graph.plot(wavel,Abs,linewidth=3,linestyle=linestyles[f],label=f"{titles[f]}. exp ($C_0$={C0[f]} mol/L)")
            peaksI, peaksH = self._FindPeaks(Abs,tP,0.01)
            if (annotateP[f]): self._pickPeak(wavel,peaksI,peaksH,plot[0].get_color(),0.01,0.04,0.02)
            print(f"{bg.LIGHTREDB}exp. {titles[f]}{bg.OFF}")
            for i in range(len(peaksI)):
                print(f"peak {i:3}. {wavel[peaksI[i]]:4} nm. A = {peaksH[i]:.2f}")

        graph.legend(fontsize=self.fontSize_legends)

        if filename is not None: self.fig.savefig(filename, dpi=300, bbox_inches='tight')
    
        return
        
