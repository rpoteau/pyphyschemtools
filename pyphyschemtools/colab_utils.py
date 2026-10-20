"""
FR — Petits utilitaires génériques pour Google Colab, partagés par
plusieurs modules de pyphyschemtools (kinetics.py, et potentiellement
nano.py, PeriodicTable.py, etc.) : factorise le code répété derrière
chaque paramètre `colab=True` (kbasic.read, KORD.load_from_spreadsheet,
...) au lieu de le dupliquer dans chaque méthode.

EN — Small, generic Google Colab utilities shared across several
pyphyschemtools modules (kinetics.py, and potentially nano.py,
PeriodicTable.py, etc.): factors out the code repeated behind every
`colab=True` parameter (kbasic.read, KORD.load_from_spreadsheet, ...)
instead of duplicating it in every method.
"""

def colab_upload(single=True, verbose=True):
    """
    FR — Ouvre le bouton d'upload de fichier de Google Colab et retourne
    le(s) nom(s) du/des fichier(s) choisi(s) -- prêt à être passé à
    pandas.read_excel(), pandas.read_csv(), etc.

    Lève une ImportError explicite si appelé hors de Google Colab (ce
    n'est pas un détecteur d'environnement -- juste un raccourci pour ce
    que `from google.colab import files; files.upload()` fait déjà, avec
    un message d'erreur clair si le module n'est pas disponible, plutôt
    que l'ImportError générique de Python).

    Args:
        single: si True (défaut), retourne un seul chemin -- celui du
            premier fichier choisi, pour écrire directement
            `pd.read_excel(colab_upload())`. Si False, retourne la liste
            de tous les fichiers uploadés (pour un upload multiple).
        verbose: si True (défaut), affiche le nom du/des fichier(s) reçu(s).

    Returns:
        str (single=True) ou list[str] (single=False).

    EN — Opens Google Colab's file upload button and returns the
    uploaded file's name(s) -- ready to pass to pandas.read_excel(),
    pandas.read_csv(), etc.

    Raises a clear ImportError if called outside Google Colab (this is
    not an environment detector -- just a shortcut for what
    `from google.colab import files; files.upload()` already does, with
    a clear error message if the module isn't available, rather than
    Python's generic ImportError).

    Args:
        single: if True (default), returns a single path -- the first
            file chosen, so you can write `pd.read_excel(colab_upload())`
            directly. If False, returns the list of every uploaded file
            (for a multi-file upload).
        verbose: if True (default), prints the name(s) of the received
            file(s).

    Returns:
        str (single=True) or list[str] (single=False).
    """
    try:
        from google.colab import files   # only available on Colab
    except ImportError:
        raise ImportError(
            "colab_upload() only works inside Google Colab (it relies on "
            "google.colab.files, which isn't installed/available here). "
            "Pass a local file path directly instead."
        )
    uploaded = files.upload()
    names = list(uploaded.keys())
    if verbose:
        for name in names:
            print(f"Fichier reçu / File received: {name}")
    return names[0] if single else names
