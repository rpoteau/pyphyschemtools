import csv
import random
import threading
import time
import requests
import difflib 
from rdkit import Chem
from rdkit.Chem import AllChem 
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import rdBase
from IPython.display import display, SVG, clear_output, HTML
import ipywidgets as widgets
from pathlib import Path

rdBase.DisableLog('rdApp.error') 
rdBase.DisableLog('rdApp.warning')

try:
    import py3Dmol
    HAS_3D = True
except ImportError:
    HAS_3D = False


TRADUCTIONS = {
    'fr': {
        'micro': 'Micro', 'macro': 'Macro',
        'iupac_btn': 'Nom IUPAC', 'iupac_result': 'Nom officiel (PubChem) :',
        'nomenclature': 'Nomenclature', 'notes': 'Notes',
        'smiles_label': 'SMILES :',
        'find_function': 'Quelle est la famille chimique en <span style="color:#ff6b6b">ROSE</span> ?',
        'score': 'Score', 'streak': 'Série', 'timer': 'Temps',
        'verify': 'Vérifier', 'next': 'Suivant',
        'bravo': 'Excellent !', 'wrong': 'Dommage !',
        'structural_motif': 'Motif Structural :',
        'name': 'Nom', 'family': 'Famille',
        'prefix': 'Préfixe', 'suffix': 'Suffixe', 'note_label': 'Note',
        'expected': 'Attendu :',
        'explorer_tab': '🔍 Explorateur',
        'quiz_struct_tab': '🔬 Quiz Structure',
        'quiz_nom_tab': '📝 Quiz Nomenclature',
        'searching': 'Recherche...', 'not_found': 'Inconnu',
        'time_up': '⏰ TEMPS ÉCOULÉ !',
        'correct_answer_was': 'La réponse était :',
        'game_over': 'PERDU ! Plus de vies.',
        'no_match': 'Aucune fonction détectée.',
        'timer_opt': '⏱️ Chrono',
        'survival_opt': '❤️ Survie',
        'none_label': '(Aucun)',
        '3d_btn': 'Vue 3D',
        '3d_warning': '⚠️ <b>Note :</b> Structure optimisée par champ de force (UFF).<br>Ceci est une approximation théorique, pas une structure expérimentale.',
        'compare_err': 'Comparaison :',
        'you_thought': 'Vous avez dit :',
        'it_was': 'C\'était :',
        'select_placeholder': '--- Tapez ou Choisissez ---' 
    },
    'en': {
        'micro': 'Micro', 'macro': 'Macro',
        'iupac_btn': 'IUPAC Name', 'iupac_result': 'Official Name (PubChem):',
        'nomenclature': 'Nomenclature', 'notes': 'Notes',
        'smiles_label': 'SMILES:',
        'find_function': 'What is the chemical family in <span style="color:#ff6b6b">PINK</span>?',
        'score': 'Score', 'streak': 'Streak', 'timer': 'Time',
        'verify': 'Check', 'next': 'Next',
        'bravo': 'Excellent!', 'wrong': 'Too bad!',
        'structural_motif': 'Structural Motif:',
        'name': 'Name', 'family': 'Family',
        'prefix': 'Prefix', 'suffix': 'Suffix', 'note_label': 'Note',
        'expected': 'Expected:',
        'explorer_tab': '🔍 Explorer',
        'quiz_struct_tab': '🔬 Structure Quiz',
        'quiz_nom_tab': '📝 Nomenclature Quiz',
        'searching': 'Searching...', 'not_found': 'Unknown',
        'time_up': '⏰ TIME IS UP!',
        'correct_answer_was': 'The answer was:',
        'game_over': 'GAME OVER! No lives left.',
        'no_match': 'No function detected.',
        'timer_opt': '⏱️ Timer',
        'survival_opt': '❤️ Survival',
        'none_label': '(None)',
        '3d_btn': '3D View',
        '3d_warning': '⚠️ <b>Note:</b> Structure optimized by force field (UFF).<br>This is a theoretical approximation, not an experimental structure.',
        'compare_err': 'Comparison:',
        'you_thought': 'You said:',
        'it_was': 'It was:',
        'select_placeholder': '--- Type or Select ---'
    }
}


class FunctionalGroupExplorer:
    """
    A comprehensive educational tool for organic chemistry within the pyphyschemtools suite.
    
    This class provides an interactive interface for:
    - Exploring functional groups using SMARTS patterns.
    - Retrieving IUPAC nomenclature via PubChem API.
    - Visualizing molecules in 2D (with highlighting) and 3D.
    - Training via a Quiz system with fuzzy logic validation and survival mechanics.
    """
    def __init__(self):
        """
        Initialize the application with language support and data loading.
        
        Args:
            lang (str): 'FR' for French or 'EN' for English. Defaults to 'FR'.
        """
        
        self.lang = 'fr'
        
        self.data_funcs = []
        self.data_mols = []
        self.cache_iupac = {}
        self.map_nom_id = {} 
        
        self.last_category = 'Micro'

        self.base_path = Path(__file__).parent
        self.data_dir = self.base_path / "resources" / "orgachem"
        
        self.score = 0
        self.streak = 0
        self.lives = 3 
        self.stop_timer_flag = False
        
        self.current_q_struct = None
        self.current_mol_struct_data = None 
        self.current_svg_struct = None    
        
        self.current_q_nom = None
        self.current_svg_nom = None
        
        self._charger_donnees()
        self.setup_ui()

    def T(self, key):
        return TRADUCTIONS[self.lang].get(key, key)

    def detecter_delimiteur(self, fichier):
        try:
            with open(fichier, 'r', encoding='utf-8-sig') as f:
                return ';' if ';' in f.readline() else ','
        except: return ';'

    def _charger_donnees(self):
        exemples = {}
        
        path_examples = self.data_dir / "Functions_Examples.csv"
        path_funcs = self.data_dir / "Functions.csv"
        path_mols = self.data_dir / "Molecules.csv"
        
        try:
            delim = self.detecter_delimiteur(path_examples)
            with open(path_examples, mode='r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f, delimiter=delim)
                if reader.fieldnames: reader.fieldnames = [n.strip() for n in reader.fieldnames]
                for row in reader:
                    if row.get('id'): exemples[row['id']] = row.get('smiles')
        except: pass

        self.data_funcs = []
        try:
            delim = self.detecter_delimiteur(path_funcs)
            with open(path_funcs, mode='r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f, delimiter=delim)
                if reader.fieldnames: reader.fieldnames = [n.strip() for n in reader.fieldnames]
                for row in reader:
                    if not row.get('smarts'): continue
                    smarts = Chem.MolFromSmarts(row['smarts'])
                    if smarts:
                        try:
                            prio = int(row.get('Priorite', 999))
                        except:
                            prio = 999
                            
                        item = {
                            'id': row.get('id', 'Inconnu'),
                            'nom_fr': row.get('nom_fr', '').strip() or row.get('id'),
                            'smarts': smarts, 
                            'macro_micro': row.get('Macro-Micro', 'Micro'),
                            'priorite': prio, 
                            'prefixe': row.get('Prefixe', '-'),
                            'suffixe': row.get('Suffixe', '-'),
                            'famille': row.get('Famille', 'Inconnue'),
                            'commentaires_fr': row.get('commentaires_fr', ''),
                            'commentaires_en': row.get('commentaires_en', ''),
                            'smiles_exemple': exemples.get(row.get('id', ''))
                        }
                        self.data_funcs.append(item)
            self.data_funcs.sort(key=lambda x: x['priorite'])
            
        except Exception as e: 
            print(f"⚠️ Note: Fichiers CSV non trouvés ou erreur. Mode démo activé.")
        
        if not self.data_funcs:
            self.data_funcs = [
                {'id': 'AcideCarbo', 'nom_fr': 'Acide Carboxylique', 'smarts': Chem.MolFromSmarts('[CX3](=O)[OX2H1]'), 'macro_micro': 'Micro', 'priorite': 1, 'prefixe': 'carboxy', 'suffixe': 'oïque', 'famille': 'Acides', 'commentaires_fr': 'COOH', 'commentaires_en': 'COOH'},
                {'id': 'Cetone', 'nom_fr': 'Cétone', 'smarts': Chem.MolFromSmarts('[#6][CX3](=O)[#6]'), 'macro_micro': 'Micro', 'priorite': 7, 'prefixe': 'oxo', 'suffixe': 'one', 'famille': 'Composés carbonylés', 'commentaires_fr': 'C=O lié à 2 C', 'commentaires_en': 'C=O bonded to 2 C'},
                {'id': 'Alcool', 'nom_fr': 'Alcool', 'smarts': Chem.MolFromSmarts('[#6][OX2H]'), 'macro_micro': 'Micro', 'priorite': 9, 'prefixe': 'hydroxy', 'suffixe': 'ol', 'famille': 'Composés oxygénés', 'commentaires_fr': 'Groupe -OH', 'commentaires_en': '-OH group'},
            ]
            self.data_funcs.sort(key=lambda x: x['priorite'])

        self.data_mols = []
        try:
            delim = self.detecter_delimiteur(path_mols)
            with open(path_mols, mode='r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f, delimiter=delim)
                if reader.fieldnames: reader.fieldnames = [n.strip() for n in reader.fieldnames]
                for row in reader:
                    if row.get('smiles'):
                        n_fr = row.get('nom_fr', row.get('nom', 'Inconnu'))
                        self.data_mols.append({
                            'fr': n_fr,
                            'en': row.get('nom_en', n_fr),
                            'smiles': row['smiles']
                        })
        except: pass
        
        if not self.data_mols:
            self.data_mols = [
                {'fr':'Paracétamol', 'en':'Paracetamol', 'smiles':'CC(=O)NC1=CC=C(O)C=C1'},
                {'fr':'Acétone', 'en':'Acetone', 'smiles':'CC(=O)C'},
                {'fr':'Acide Acétique', 'en':'Acetic Acid', 'smiles':'CC(=O)O'},
                {'fr':'Éthanol', 'en':'Ethanol', 'smiles':'CCO'}
            ]

    def fuzzy_match(self, user_input, expected):
        """Retourne Vrai si la réponse est ressemblante à > 80%"""
        if expected in ['-', '']: return user_input == expected
        seq = difflib.SequenceMatcher(None, user_input, expected)
        return seq.ratio() > 0.8

    def mol_to_svg(self, mol, highlightAtoms=None, highlightBonds=None, size=(300, 200)):
        d2d = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
        opts = d2d.drawOptions()
        opts.addStereoAnnotation = True
        
        try:
            mol.UpdatePropertyCache(strict=False)
        except: pass
        
        try:
            if mol.GetNumConformers() == 0: Chem.Compute2DCoords(mol)
        except: pass

        if highlightAtoms:
            col = (1.0, 0.65, 0.65) # Rose Saumon
            hc_a = {x: col for x in highlightAtoms}
            hc_b = {x: col for x in highlightBonds} if highlightBonds else {}
            d2d.DrawMolecule(mol, highlightAtoms=highlightAtoms, highlightBonds=highlightBonds,
                             highlightAtomColors=hc_a, highlightBondColors=hc_b)
        else:
            d2d.DrawMolecule(mol)
        d2d.FinishDrawing()
        return d2d.GetDrawingText()

    def show_3d_mol(self, smiles):
        if not HAS_3D: 
            return widgets.HTML("<b>Py3Dmol not installed</b>")
        
        mol = Chem.MolFromSmiles(smiles)
        if not mol: 
            return widgets.HTML("Invalid SMILES")

        mol = Chem.AddHs(mol)

        try:
            res = AllChem.EmbedMolecule(mol, randomSeed=42)
            if res == -1:
                res = AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)

            if res != -1:
                AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        except Exception as e:
            print(f"Erreur d'optimisation 3D : {e}")
        
        display(widgets.HTML(
            f"<div style='background-color:#fff3cd; color:#856404; padding:10px; "
            f"border:1px solid #ffeeba; border-radius:5px; margin-bottom:10px; font-size:0.9em;'>"
            f"{self.T('3d_warning')}</div>"
        ))

        mblock = Chem.MolToMolBlock(mol)
        view = py3Dmol.view(width=400, height=300)
        view.addModel(mblock, 'mol')
        view.setStyle({'stick': {'radius': 0.15}, 'sphere': {'scale': 0.25}}) # Style amélioré
        view.zoomTo()
        
        return view

    def get_match_svg(self, smiles, query):
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return None
        matches = mol.GetSubstructMatches(query['smarts'])
        if not matches: return None
        match = random.choice(matches)
        atoms = list(match)
        bonds = []
        for b in query['smarts'].GetBonds():
            try:
                aid1, aid2 = match[b.GetBeginAtomIdx()], match[b.GetEndAtomIdx()]
                mb = mol.GetBondBetweenAtoms(aid1, aid2)
                if mb: bonds.append(mb.GetIdx())
            except: pass
        return self.mol_to_svg(mol, atoms, bonds)

    def setup_ui(self):
        self.w_lang = widgets.ToggleButtons(
            options=[('🇫🇷 Français', 'fr'), ('🇬🇧 English', 'en')],
            value='fr', button_style='', style={'button_width': '150px'}
        )
        self.w_lang.observe(self.on_lang_change, names='value')

        self.ex_smiles = widgets.Text(value='NC(CC1=CC=C(O)C=C1)C(=O)O', description='SMILES:', layout={'width':'400px'})
        self.ex_out = widgets.Output()
        
        self.ex_btn_micro = widgets.Button(description='Micro', button_style='danger')
        self.ex_btn_macro = widgets.Button(description='Macro', button_style='success')
        self.ex_btn_iupac = widgets.Button(description='IUPAC', button_style='info', icon='search')
        self.ex_btn_3d = widgets.Button(description='3D', button_style='warning', icon='cube')
        
        self.ex_toggle_nom = widgets.ToggleButton(value=True, description='Nomenclature', button_style='primary', icon='tags')
        self.ex_toggle_notes = widgets.ToggleButton(value=True, description='Notes', button_style='warning', icon='edit')
        
        self.ex_btn_micro.on_click(lambda b: self.update_explorer('Micro'))
        self.ex_btn_macro.on_click(lambda b: self.update_explorer('Macro'))
        self.ex_btn_iupac.on_click(self.run_iupac)
        self.ex_btn_3d.on_click(self.run_3d)
        self.ex_toggle_nom.observe(lambda c: self.update_explorer(self.last_category), names='value')
        self.ex_toggle_notes.observe(lambda c: self.update_explorer(self.last_category), names='value')

        ui_explorateur = widgets.VBox([
            widgets.HBox([self.ex_smiles, self.ex_btn_micro, self.ex_btn_macro, self.ex_btn_iupac, self.ex_btn_3d]),
            widgets.HBox([self.ex_toggle_nom, self.ex_toggle_notes]),
            self.ex_out
        ])

        self.qs_out = widgets.Output()
        
        self.qs_drop = widgets.Combobox(
            placeholder=self.T('select_placeholder'),
            options=[],
            ensure_option=True, 
            layout={'width': '300px'}
        )
        self.qs_drop.on_submit(self.check_struct)
        
        self.qs_btn_check = widgets.Button(description='Vérifier', button_style='primary', icon='check')
        self.qs_btn_next = widgets.Button(description='Suivant', icon='arrow-right')
        self.qs_check_timer = widgets.Checkbox(value=False, description='⏱️ Chrono', indent=False)
        self.qs_check_survival = widgets.Checkbox(value=False, description='❤️ Survie', indent=False)
        
        self.qs_lbl_lives = widgets.HTML(value=self.get_lives_html())
        
        self.qs_lbl_total_score = widgets.HTML(value="<b>🏆 Score : 0</b>", layout={'margin': '0 0 0 20px'})
        
        self.qs_bar_score = widgets.IntProgress(value=0, min=0, max=100, description='', bar_style='success', layout={'width':'150px'})
        self.qs_bar_timer = widgets.IntProgress(value=100, min=0, max=100, description='⏱️', bar_style='info')

        self.qs_btn_check.on_click(self.check_struct)
        self.qs_btn_next.on_click(self.next_struct)
        self.qs_check_survival.observe(self.on_survival_change, names='value')
        
        ui_struct = widgets.VBox([
            widgets.HBox([self.qs_bar_score, self.qs_lbl_total_score, self.qs_lbl_lives], layout={'justify_content': 'center', 'align_items': 'center'}),
            widgets.HBox([self.qs_bar_timer, self.qs_check_timer, self.qs_check_survival], layout={'justify_content': 'center'}),
            self.qs_out,
            widgets.HTML("<hr>"),
            widgets.HBox([self.qs_drop, self.qs_btn_check, self.qs_btn_next], layout={'justify_content': 'center'})
        ])

        self.qn_out = widgets.Output()
        self.qn_in_pref = widgets.Text()
        self.qn_label_pref = widgets.Label(value='Préfixe :', layout={'width': '80px'})
        self.qn_in_suff = widgets.Text()
        self.qn_label_suff = widgets.Label(value='Suffixe :', layout={'width': '80px'})
        
        self.qn_in_pref.on_submit(self.check_nom)
        self.qn_in_suff.on_submit(self.check_nom)
        
        self.qn_box_pref = widgets.HBox([self.qn_label_pref, self.qn_in_pref], layout={'justify_content': 'center'})
        self.qn_box_suff = widgets.HBox([self.qn_label_suff, self.qn_in_suff], layout={'justify_content': 'center'})

        self.qn_btn_check = widgets.Button(description='Vérifier', button_style='primary', icon='check')
        self.qn_btn_next = widgets.Button(description='Suivant', icon='arrow-right')
        self.qn_bar_score = widgets.IntProgress(value=0, min=0, max=100, description='Score:', bar_style='success')

        self.qn_btn_check.on_click(self.check_nom)
        self.qn_btn_next.on_click(self.next_nom)

        ui_nom = widgets.VBox([
            self.qn_bar_score,
            self.qn_out,
            widgets.HTML("<hr>"),
            self.qn_box_pref,
            self.qn_box_suff,
            widgets.HBox([self.qn_btn_check, self.qn_btn_next], layout={'justify_content': 'center', 'margin_top': '10px'})
        ])

        self.tabs = widgets.Tab(children=[ui_explorateur, ui_struct, ui_nom])
        self.update_labels()
        display(widgets.HTML("<style>.widget-label { font-weight: bold; }</style>"))
        display(widgets.VBox([self.w_lang, self.tabs]))
        
        self.next_struct()
        self.next_nom()

    def update_labels(self):
        # Titres
        self.tabs.set_title(0, self.T('explorer_tab'))
        self.tabs.set_title(1, self.T('quiz_struct_tab'))
        self.tabs.set_title(2, self.T('quiz_nom_tab'))
        
        self.ex_btn_micro.description = self.T('micro')
        self.ex_btn_macro.description = self.T('macro')
        self.ex_btn_iupac.description = self.T('iupac_btn')
        self.ex_btn_3d.description = self.T('3d_btn')
        self.ex_smiles.description = self.T('smiles_label')
        self.ex_toggle_nom.description = self.T('nomenclature')
        self.ex_toggle_notes.description = self.T('notes')
        
        self.qs_btn_check.description = self.T('verify')
        self.qs_btn_next.description = self.T('next')
        self.qs_drop.placeholder = self.T('select_placeholder')
        self.qs_check_timer.description = self.T('timer_opt')
        self.qs_check_survival.description = self.T('survival_opt')
        
        self.qn_label_pref.value = self.T('prefix') + " :"
        self.qn_label_suff.value = self.T('suffix') + " :"
        self.qn_btn_check.description = self.T('verify')
        self.qn_btn_next.description = self.T('next')
        self.qn_bar_score.description = self.T('score')
        
        if self.qn_in_pref.disabled: self.qn_in_pref.placeholder = self.T('none_label')
        if self.qn_in_suff.disabled: self.qn_in_suff.placeholder = self.T('none_label')

    def on_lang_change(self, change):
        self.lang = change['new']
        self.update_labels()
        self.update_explorer(self.last_category)
        if self.current_q_struct:
            self.update_combobox_options() 
        self.afficher_quiz_struct()
        self.afficher_quiz_nom()

    def get_lives_html(self):
        if not self.qs_check_survival.value: return ""
        hearts = "❤️" * self.lives + "🖤" * (3 - self.lives)
        return f"<span style='font-size:1.5em; vertical-align:middle; margin-left:10px'>{hearts}</span>"

    def on_survival_change(self, change):
        if change['new']: 
            self.lives = 3
            self.score = 0
            self.qs_bar_score.value = 0
            self.qs_lbl_total_score.value = f"<b>🏆 {self.T('score')} : 0</b>"
        self.qs_lbl_lives.value = self.get_lives_html()

    def start_timer_thread(self, progress_bar, callback_timeout):
        self.stop_timer_flag = True
        time.sleep(0.1)
        self.stop_timer_flag = False
        
        def _timer_run():
            progress_bar.value = 100
            progress_bar.bar_style = 'info'
            while progress_bar.value > 0:
                if self.stop_timer_flag: return
                time.sleep(0.15)
                progress_bar.value -= 1
                if progress_bar.value < 30: progress_bar.bar_style = 'warning'
                if progress_bar.value < 10: progress_bar.bar_style = 'danger'
            if not self.stop_timer_flag:
                callback_timeout()
        
        t = threading.Thread(target=_timer_run)
        t.start()

    def run_iupac(self, b):
        smi = self.ex_smiles.value
        with self.ex_out:
            clear_output(wait=True)
            if smi in self.cache_iupac:
                nom = self.cache_iupac[smi]
            else:
                print(self.T('searching'))
                try:
                    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smi}/property/IUPACName/TXT"
                    r = requests.get(url, timeout=3)
                    nom = r.text.strip() if r.status_code == 200 else self.T('not_found')
                    self.cache_iupac[smi] = nom
                except: nom = self.T('not_found')
          
            display(widgets.HTML(f"<div style='background:#e3f2fd;padding:10px;border-radius:8px;margin-bottom:10px;'><b>{self.T('iupac_result')}</b><br>{nom}</div>"))
            mol = Chem.MolFromSmiles(smi)
            if mol: display(SVG(self.mol_to_svg(mol, size=(300, 200))))

    def run_3d(self, b):
        smi = self.ex_smiles.value
        with self.ex_out:
            clear_output(wait=True)
            print(self.T('searching'))
            view = self.show_3d_mol(smi)
            if isinstance(view, widgets.HTML):
                display(view)
            elif view:
                view.show()

    def update_explorer(self, cat):
        self.last_category = cat
        with self.ex_out:
            clear_output(wait=True)
            smi = self.ex_smiles.value
            mol = Chem.MolFromSmiles(smi)
            if not mol: print("SMILES Invalide"); return
            
            found = False
            atomes_utilises = set() 
            
            for q in self.data_funcs:
                if q['macro_micro'] != cat: continue
                
                matches = mol.GetSubstructMatches(q['smarts'])
                valid_matches = []
                
                for m in matches:
                    if not any(idx in atomes_utilises for idx in m):
                        valid_matches.append(m)
                        atomes_utilises.update(m)
                
                if valid_matches:
                    found = True
                    atoms_all = set()
                    bonds_all = set()
                    for m in valid_matches:
                        atoms_all.update(m)
                        for b in q['smarts'].GetBonds():
                            try:
                                a1, a2 = m[b.GetBeginAtomIdx()], m[b.GetEndAtomIdx()]
                                mb = mol.GetBondBetweenAtoms(a1, a2)
                                if mb: bonds_all.add(mb.GetIdx())
                            except: pass
                  
                    nom_aff = q['nom_fr'] if self.lang == 'fr' else q['id']
                    display(widgets.HTML(f"<h4>🔹 {nom_aff} </h4>"))
                    display(SVG(self.mol_to_svg(mol, list(atoms_all), list(bonds_all), size=(300, 200))))
                  
                    info_html = ""
                    if self.ex_toggle_nom.value:
                         info_html += f"<div style='background:#f0f4f8;padding:5px;margin:5px;border-left:4px solid #2196F3;font-size:0.9em'><b>{self.T('prefix')}:</b> {q['prefixe']} | <b>{self.T('suffix')}:</b> {q['suffixe']}</div>"
                    if self.ex_toggle_notes.value:
                        comm = q['commentaires_fr'] if self.lang == 'fr' else (q['commentaires_en'] or q['commentaires_fr'])
                        info_html += f"<div style='background:#fff3e0;padding:5px;margin:5px;border-left:4px solid #ff9800;font-size:0.9em'><b>{self.T('note_label')}:</b> {comm}</div>"
                    if info_html: display(widgets.HTML(info_html))
            if not found: print(self.T('no_match'))

    def update_combobox_options(self):
        """Mise à jour des options de la Combobox et du dictionnaire de correspondance"""
        self.map_nom_id = {}
        opts = []
        for q in self.data_funcs:
            nom = f"{q['nom_fr']} ({q['id']})" if self.lang == 'fr' else q['id']
            self.map_nom_id[nom] = q['id']
            opts.append(nom)
        
        self.qs_drop.options = sorted(opts)

    def afficher_quiz_struct(self):
        if not self.current_q_struct: return
        
        with self.qs_out:
            clear_output(wait=True)
            nom_mol = self.current_mol_struct_data['fr'] if self.lang == 'fr' else self.current_mol_struct_data['en']
            display(widgets.HTML(f"""
            <div style='background:#f1f8e9; padding:15px; border-radius:10px; border:1px solid #8bc34a; text-align:center;'>
                <h3 style='margin:0'>{self.T('find_function')}</h3>
                <small>{nom_mol}</small>
            </div>"""))
            if self.current_svg_struct:
                display(SVG(self.current_svg_struct))

    def next_struct(self, b=None):
        if self.qs_check_survival.value and self.lives <= 0:
            with self.qs_out:
                clear_output(wait=True)
                # Affichage du score final en grand
                display(widgets.HTML(f"<h1 style='color:red;text-align:center'>☠️ {self.T('game_over')} ☠️</h1>"))
                display(widgets.HTML(f"<h3 style='text-align:center'>Score Final : {self.score}</h3>"))
            self.qs_btn_check.disabled = True
            return

        self.stop_timer_flag = True
        self.qs_btn_check.disabled = False
        self.qs_btn_check.button_style = 'primary'
        self.qs_btn_check.description = self.T('verify')
        
        self.qs_drop.value = '' 
        
        if not self.data_mols: return
        mol_data = random.choice(self.data_mols)
        mol = Chem.MolFromSmiles(mol_data['smiles'])
        
        atomes_utilises = set()
        candidates = []
        
        for q in self.data_funcs:
            matches = mol.GetSubstructMatches(q['smarts'])
            for m in matches:
                if not any(idx in atomes_utilises for idx in m):
                    candidates.append((q, m))
                    atomes_utilises.update(m)
        
        if not candidates: 
            self.next_struct()
            return
        
        q, match = random.choice(candidates)
        
        bonds = []
        for b in q['smarts'].GetBonds():
            try:
                aid1, aid2 = match[b.GetBeginAtomIdx()], match[b.GetEndAtomIdx()]
                mb = mol.GetBondBetweenAtoms(aid1, aid2)
                if mb: bonds.append(mb.GetIdx())
            except: pass
            
        svg = self.mol_to_svg(mol, list(match), bonds)
            
        self.current_q_struct = q
        self.current_mol_struct_data = mol_data
        self.current_svg_struct = svg
        
        self.update_combobox_options()
        self.afficher_quiz_struct()
        
        if self.qs_check_timer.value:
            self.start_timer_thread(self.qs_bar_timer, self.timeout_struct)
        else:
            self.qs_bar_timer.value = 100
            self.qs_bar_timer.bar_style = 'success'

    def timeout_struct(self):
        self.qs_btn_check.disabled = True
        if self.qs_check_survival.value:
            self.lives -= 1
            self.qs_lbl_lives.value = self.get_lives_html()
          
        with self.qs_out:
            display(widgets.HTML(f"<div style='color:white;background:#d32f2f;padding:10px;text-align:center;border-radius:5px;'><b>{self.T('time_up')}</b><br>{self.T('game_over')}</div>"))

    def check_struct(self, b):
        if self.qs_btn_check.disabled: return
        self.qs_btn_check.disabled = True
        self.stop_timer_flag = True
        
        user_text = self.qs_drop.value
        
        user_id = self.map_nom_id.get(user_text)
        
        if not user_id:
            self.qs_btn_check.disabled = False 
            return

        correct_id = self.current_q_struct['id']
        is_correct = (user_id == correct_id)
        
        with self.qs_out:
            if is_correct:
                self.score += 10
                self.qs_btn_check.button_style = 'success'
                self.qs_btn_check.description = self.T('bravo')
                display(widgets.HTML(f"<div style='font-size:24px;color:#2e7d32;text-align:center;margin-top:10px;'>✅ {self.T('bravo')}</div>"))
            else:
                self.qs_btn_check.button_style = 'danger'
                self.qs_btn_check.description = self.T('wrong')
               
                if self.qs_check_survival.value:
                    self.lives -= 1
                    self.qs_lbl_lives.value = self.get_lives_html()
               
                vrai_nom = self.current_q_struct['nom_fr'] if self.lang == 'fr' else self.current_q_struct['id']
               
                wrong_item = next((item for item in self.data_funcs if item['id'] == user_id), None)
                wrong_svg = None
                wrong_name = "Inconnu"
               
                if wrong_item:
                    wrong_name = wrong_item['nom_fr'] if self.lang == 'fr' else wrong_item['id']
                    if wrong_item.get('smarts'):
                        try:
                            mol_wrong = Chem.Mol(wrong_item['smarts'])
                            mol_wrong.UpdatePropertyCache(strict=False)
                            wrong_svg = self.mol_to_svg(mol_wrong, size=(150, 150))
                        except Exception as e:
                            pass
               
                html_compare = f"""
                <div style='font-size:18px;color:#c62828;text-align:center;margin-top:10px;background:#ffcdd2;padding:10px;border-radius:5px;'>
                    ❌ {self.T('wrong')}
                </div>
                <br>
                <div style="display: flex; justify-content: center; align-items: center; gap: 20px;">
                    <div style="text-align:center; border: 2px solid #e57373; padding: 5px; border-radius: 8px;">
                        <div style="font-weight:bold; color: #c62828;">{self.T('you_thought')}</div>
                        <div>{wrong_name}</div>
                        {wrong_svg if wrong_svg else ""}
                    </div>
                    <div style="font-size: 2em;">👉</div>
                    <div style="text-align:center; border: 2px solid #81c784; padding: 5px; border-radius: 8px; min-width: 150px;">
                        <div style="font-weight:bold; color: #2e7d32;">{self.T('it_was')}</div>
                        <div style="font-size: 1.2em; margin: 10px;">{vrai_nom}</div>
                    </div>
                </div>
                """
                display(widgets.HTML(html_compare))
        
        self.qs_bar_score.value = min(self.score, 100)
        self.qs_lbl_total_score.value = f"<b>🏆 {self.T('score')} : {self.score}</b>"

    def afficher_quiz_nom(self):
        if not self.current_q_nom: return
        with self.qn_out:
            clear_output(wait=True)
            nom_f = self.current_q_nom['nom_fr'] if self.lang == 'fr' else self.current_q_nom['id']
            display(widgets.HTML(f"""
            <div style='background:#e3f2fd; padding:15px; border-radius:10px; border:2px solid #2196F3; text-align:center; width: 60%; margin: auto;'>
                <h3 style='margin:0; color:#1565c0'>{self.T('name')} : <span style='color:#e91e63'>{nom_f}</span></h3>
                <small>{self.T('family')} : {self.current_q_nom['famille']}</small>
            </div>
            <br>"""))
            if self.current_svg_nom:
                display(SVG(self.current_svg_nom))
   
    def next_nom(self, b=None):
        self.qn_btn_check.disabled = False
        self.qn_btn_check.button_style = 'primary'
        self.qn_btn_check.description = self.T('verify')
        
        self.qn_in_pref.value = ''
        self.qn_in_suff.value = ''
        self.qn_in_pref.placeholder = ''
        self.qn_in_suff.placeholder = ''
        self.qn_in_pref.disabled = False
        self.qn_in_suff.disabled = False
        
        self.current_q_nom = random.choice(self.data_funcs)
        
        p_val = self.current_q_nom['prefixe'].strip()
        s_val = self.current_q_nom['suffixe'].strip()
        
        if p_val in ['-', '']: self.qn_box_pref.layout.display = 'none'
        else: self.qn_box_pref.layout.display = 'flex'
          
        if s_val in ['-', '']: self.qn_box_suff.layout.display = 'none'
        else: self.qn_box_suff.layout.display = 'flex'
        
        svg = None
        if self.current_q_nom.get('smiles_exemple'):
            svg = self.get_match_svg(self.current_q_nom['smiles_exemple'], self.current_q_nom)
        if not svg:
            svg = self.mol_to_svg(Chem.Mol(self.current_q_nom['smarts']), size=(200,150))
          
        self.current_svg_nom = svg
        self.afficher_quiz_nom()

    def check_nom(self, b):
        if self.qn_btn_check.disabled: return
        self.qn_btn_check.disabled = True
        
        p_att = self.current_q_nom['prefixe'].strip().lower()
        s_att = self.current_q_nom['suffixe'].strip().lower()
        p_usr = self.qn_in_pref.value.strip().lower()
        s_usr = self.qn_in_suff.value.strip().lower()
        
        is_pref_hidden = (self.qn_box_pref.layout.display == 'none')
        is_suff_hidden = (self.qn_box_suff.layout.display == 'none')
        
        ok_p = True
        if not is_pref_hidden:
            ok_p = self.fuzzy_match(p_usr, p_att)
          
        ok_s = True
        if not is_suff_hidden:
            ok_s = self.fuzzy_match(s_usr, s_att)
        
        res_html = "<div style='background:#f5f5f5;padding:15px;border-radius:10px;margin-top:15px;border:1px solid #ccc;'>"
        
        if not is_pref_hidden:
            if ok_p: 
                msg = f"✅ <b>{self.T('prefix')}:</b> Correct (+5)<br>"
                if p_usr != p_att: msg = f"✅ <b>{self.T('prefix')}:</b> Correct (<i>{p_att}</i>) (+5)<br>"
                res_html += msg
            else: res_html += f"❌ <b>{self.T('prefix')}:</b> {self.T('expected')} <b>{p_att}</b><br>"
        
        if not is_suff_hidden:
            if ok_s: 
                msg = f"✅ <b>{self.T('suffix')}:</b> Correct (+5)<br>"
                if s_usr != s_att: msg = f"✅ <b>{self.T('suffix')}:</b> Correct (<i>{s_att}</i>) (+5)<br>"
                res_html += msg
            else: res_html += f"❌ <b>{self.T('suffix')}:</b> {self.T('expected')} <b>{s_att}</b>"
        
        res_html += "</div>"
        
        pts = 0
        if not is_pref_hidden and ok_p: pts += 5
        if not is_suff_hidden and ok_s: pts += 5
        
        self.qn_btn_check.button_style = 'success' if (ok_p and ok_s) else 'warning'
          
        with self.qn_out: display(widgets.HTML(res_html))
        self.qn_bar_score.value = min(self.qn_bar_score.value + pts, 100)
