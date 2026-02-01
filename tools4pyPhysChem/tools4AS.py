__author__ = "Romuald POTEAU"
__maintainer__ =  "Romuald POTEAU"
__email__ = "romuald.poteau@univ-tlse3.fr"
__status__ = "Development"

####################################################################################################################################
#                    F E U I L L E S   D E   S T Y L E
####################################################################################################################################

from .visualID_Eng import fg, hl, bg


from IPython.display import HTML

def css_styling():
    styles = open("./tools4AS.css", "r").read()
    return HTML(styles)


####################################################################################################################################
#                    F O N C T I O N S    M A I S O N
####################################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
# importation de la libairie pandas
import pandas as pd
import dataframe_image as dfim
import seaborn as sns

def verifNotes(dfCC,labelNoteCC,nomCC,NI,absents='aabs'):
    """
    entrée :
        - dfCC = dataframe dont 1 colonne contient les notes de CC
        - labelNoteCC = label de la colonne qui contient les notes
        - nomCC = label de l'épreuve de CC (ex CC1), utilisé pour l'affichage
        - NI = nombre d'inscrits dans le module
        - absents = 'aabs' : analyser s'il y a des étudiants qui n'ont pas été pointés au CC (défaut)
                  = 'noabs' : ne pas analyser s'il y a des étudiants qui n'ont pas été pointés au CC
                              (ça n'a plus de sens une fois les fichiers concaténés)
        
    sortie :
        - la moyenne et la déviation standard de liste de notes contenues dans le dataframe dfCC
        - le nombre d'étudiants qui n'ont pas composé au CC
    
    affichages :
        - nombre d'étudiants avec label 'ABS'|'Abs'|'abs'
        - nombre d'étudiants sans note ni label ABS. En général c'est problématique. Vérifier le PV 
    """
    print()
    #pd.set_option("display.max_rows", len(dfCC))
    #display(dfCC[labelNoteCC])
    if (absents == 'aabs'):
        nABS = ((dfCC[labelNoteCC] == "ABS") | (dfCC[labelNoteCC] == "abs") | (dfCC[labelNoteCC] == "Abs") ).sum()
        print(f"{hl.BOLD}{fg.BLUE}Etudiants de {nomCC} avec label 'ABS' = {nABS}{fg.OFF}")
    nEMPTY = (dfCC[labelNoteCC].isna()).sum()
    print(f"{hl.BOLD}{fg.BLUE}Etudiants de {nomCC} sans label ni note = {nEMPTY}{fg.OFF}")
    if ((nEMPTY != 0) & (absents == 'aabs')):
        print(f"{fg.RED}{hl.BOLD}Attention !!!  Ça n'est pas normal. Vérifier la liste d'appel{fg.OFF}")
#pandas.to_numeric(arg, errors='raise', downcast=None)
#Convert argument to a numeric type
#errors{‘ignore’, ‘raise’, ‘coerce’}, default ‘raise’
#   If ‘raise’, then invalid parsing will raise an exception.
#   If ‘coerce’, then invalid parsing will be set as NaN.
#   If ‘ignore’, then invalid parsing will return the input.
    nCC_Absents = pd.to_numeric(dfCC[labelNoteCC], errors='coerce').isna().values.sum()
    nCC_Notes = (~pd.to_numeric(dfCC[labelNoteCC], errors='coerce').isna()).values.sum()
    av = pd.to_numeric(dfCC[labelNoteCC], errors='coerce').mean()
    std = pd.to_numeric(dfCC[labelNoteCC], errors='coerce').std()
    print(f"{fg.BLUE}Nombre d'étudiants sur les listes du {nomCC} = {len(dfCC)}. Nombre de notes vs. nombre d'absents = {nCC_Notes} vs. {nCC_Absents}{fg.OFF}")
    print(f"Somme des copies notées & des absents = {nCC_Notes + nCC_Absents}")
    print(f"{NI - nCC_Notes}/{NI} étudiants n'ont pas composé au {nomCC}")
    print(f"{hl.BOLD}Moyenne = {av:.1f}, Écart-type = {std:.1f}{fg.OFF}")
    return av, std, (NI - nCC_Notes)


def RenameDfHeader(dfCC,dfCCname,labelNoteCC,nomCC):
    """
    entrée :
        - dfCC = dataframe dont 1 colonne contient les notes de CC
        - dfCCname = nom (string) du dataframe. Il est recommandé d'utiliser f'{dfCC=}'.split('=')[0]
        - labelNoteCC = label de la colonne qui contient les notes
        - nomCC = label de l'épreuve de CC (ex CC1), utilisé pour l'affichage
    sortie : aucune

    la fonction change le nom 'labelNoteCC' en 'nomCC'
    """
    print(f"{hl.BOLD}{fg.BLUE}Normalisation du nom des colonnes de notes{fg.OFF}")
    print(f"{hl.BOLD}Dataframe = {dfCCname}.{fg.OFF} {labelNoteCC} --> {nomCC}")
    dfCC.rename(columns = {labelNoteCC:nomCC}, inplace=True)
    
def mentionD(row, Mention):
    rowV = row[Mention]
    CHIMIE1 = "L1 CHI"
    CHIMIE2 = "L2 CHI"
    PHYSIQUE1 = "L1 PHY"
    PHYSIQUE2 = "L2 PHY"
    PHYSIQUE3 = "L3 PHY"
    MATHS1 = "L1 MAT"
    MATHS2 = "L2 MAT"
    MATHS3 = "L2 MAT"
    MECA = "L1 MECA"
    MIASHS = "L1 MIASHS"
    EEA = "L1 EEA"
    INFO1 = "L1 INFO"
    INFO2 = "L2 INFO"
    PC1 = "L1 PHYSIQUE CHIMIE"
    PC2 = "L1 PC"
    GC1 = "L1 GC"
    GC2 = "L1 GENIE CIVIL"
    MOBINT = "MOBILITE INTERNAT"
    DUMMY = "DUMMY"
    if (CHIMIE1 in rowV) | (CHIMIE2 in rowV):
        return "CHIMIE"
    elif ((PHYSIQUE1 in rowV) | (PHYSIQUE2 in rowV) | (PHYSIQUE3 in rowV)) & ~(PC1 in rowV):
        return "PHYSIQUE"
    elif (MATHS1 in rowV) | (MATHS2 in rowV) | (MATHS3 in rowV):
        return "MATHS"
    elif MIASHS in rowV:
        return "MIASHS"
    elif MECA in rowV:
        return "MECA"
    elif (INFO1 in rowV) | (INFO2 in rowV):
        return "INFO"
    elif EEA in rowV:
        return "EEA"
    elif (GC1 in rowV) | (GC2 in rowV):
        return "GC"
    elif (PC1 in rowV) | (PC2 in rowV):
        return "PC"
    elif MOBINT in rowV:
        return "MobInt"
    elif DUMMY in rowV:
        return "DUMMY"
    else:
        print(f"Quelle est cette Mention ? {rowV}")
    return

def parcours(row, Parcours):
    rowV = row[Parcours]
    CUPGE = "CUPGE"
    SANTE = "OPT° SANTE"
    ACCOMP = "ACCOMP"
    DUMMY = "DUMMY"
    if CUPGE in rowV:
        return "CUPGE"
    elif SANTE in rowV:
        return "SANTE"
    elif ACCOMP in rowV:
        return "ACCOMPAGNEMENT"
    elif DUMMY in rowV:
        return "DUMMY"
    else:
#        print(f"Pas de parcours dans la mention {rowV}")
        return "Standard"
    return

def MentionAuModule(note, Seuil):
    """
    entrée :
        - note = valeur numérique ou NaN
        - Seuil = seuil de réussite
    sortie :
        - m = mention au module (AJ, P, AB, B, TB ou PB!! dans le cas où la colonne contiendrait une valeur numérique non comprise entre 0 et 20, ou bien toute autre contenu (caractères etc)
    """
    if (note >=0) and (note < Seuil):
        m = "AJ"
    elif (note >=10) and (note < 12):
        m = "P"
    elif (note>=12) and (note < 14):
        m = "AB"
    elif (note >= 14) and (note <16):
        m = "B"
    elif (note >=16) and (note <= 20) :
        m = "TB"
    elif (np.isnan(note)):
        m = np.NaN
    else:
        print(note,"PB")
        m = 'PB!!'
    return m

def concat2ApoG(df2Complete, ID_ApoG, dfCC, Col2SaveInCC, IDCC, nomCC):
    """
    entrée :
        - df2Complete = dataframe à compléter (merge = 'left')
            - au premier appel, ce soit être le fichier de Référence
            - ensuite c'est le fichier de notes lui-même, en cours d'update
        - ID_ApoG = label de la colonne qui contient les numéros étudiants dans le fichier de référence
        - dfCC = dataframe dont 1 colonne contient les notes de CC
        - Col2SaveInCC = liste avec les en-têtes de colonnes qui contiennent les colonnes de dfCC à reporter dans dfNotes
        - IDCC = label de la colonne qui contient les numéros étudiants dans le fichier de notes
        - nomCC = à ce stade, c'est aussi bien le label de la colonne qui contient les notes que le label de l'épreuve de CC (ex CC1), utilisé pour l'affichage
        
    sortie :
        - le dataframe dfNotes. Contient la version concaténée du dataframe d'entrée df2complete et certaines colonnes du dataframe des notes dfCC
          ([Col2SaveInCC + IDCC + nomCC])
        - un dataframe dfnotFoundInRef qui contient la liste des étudiants qui sont dans le fichier de notes et pas dans le fichier de référence
    
    affichages :
        - liste des étudiants qui sont dans le fichier de notes et pas dans le fichier de référence
    """

    #display(df2Complete)
    dfNotes = df2Complete.copy()
        
    pd.set_option('display.max_rows', 12)
    #print(f"{hl.BOLD}{fg.BLUE}Fichier de notes copié de l'export Apogée{fg.OFF}")
    #display(dfNotes)
    Col2SaveInCCExt = Col2SaveInCC.copy()
    Col2SaveInCCExt.extend([IDCC])
    Col2SaveInCCExt.extend([nomCC])
    dfNotes = dfNotes.merge(dfCC[Col2SaveInCCExt], left_on=ID_ApoG, right_on=IDCC, how='left')
    print(f"{hl.BOLD}{fg.BLUE}{nomCC} ajouté dans fichier de notes{fg.OFF}")
    print(f"Les colonnes qui ont été ajoutées sont : {Col2SaveInCCExt}")
    #display(dfNotes)

    dfNotesTmp = df2Complete.copy()
    dfNotesOuter = dfNotesTmp.merge(dfCC[Col2SaveInCCExt], left_on=ID_ApoG, right_on=IDCC, how='outer')
    #display(dfNotesOuter)

    dfnotFoundInReftmp = dfNotesOuter[dfNotesOuter[ID_ApoG].isna()]
    if dfnotFoundInReftmp.shape[0] != 0:
        print(f"{hl.BOLD}{fg.RED}Problème !! Ces étudiants du {nomCC} ne sont pas dans le dataframe de Référence{fg.OFF}")
        print(f"{hl.BOLD}{fg.RED}Ils ne sont pas rajoutés dans ce dataframe, mais dans un dataframe dfnotFoundInRef{fg.OFF}")
        display(dfnotFoundInReftmp)
    else:
        print(f"{hl.BOLD}{fg.GREEN}Tous les étudiants sont bien dans le dataframe de Référence{fg.OFF}")
        
    return dfNotes, dfnotFoundInReftmp

def checkNoID_DuplicateID(df, dfname, ID, nom, NomPrenom):
    import numpy as np
    """
    entrée : 
        - df = dataframe à analyser
        - dfname = nom (string) du dataframe à analyser. Il est recommandé d'utiliser f'{dfCC=}'.split('=')[0]
        - ID = label de la colonne qui contient les numéros étudiants dans df
        - nom = label du dataframe, utilisé pour l'affichage
        - NomPrenom = liste avec les en-têtes de colonnes qui contiennent les noms et les prénoms dans df

    affichage : diagnostic et éventuellement la liste des étudiants sans numéros d'étudiant
    """
    print(f"{hl.BOLD}{fg.BLUE}Dataframe {dfname} (alias {nom}){fg.OFF}")
    noID = df[df[ID].isnull()]
    if (noID.shape[0] != 0):
        print(f"{hl.BOLD}{fg.RED}Etudiants sans ID !{fg.OFF}")
        display(noID.sort_values(by=NomPrenom[0], ascending = True))
    else:
        print(f"{hl.BOLD}{fg.GREEN}Etudiants sans ID ? Pas de problème{fg.OFF}")

    values, counts = np.unique(df[ID].to_numpy(), return_counts=True)
    
    duplicateID = False
    for c in counts:
        if c != 1: duplicateID = True
    if (not duplicateID):
        print(f"{hl.BOLD}{fg.GREEN}Doublon sur les ID ? Pas de problème{fg.OFF}")
    else:
        print(f"{hl.BOLD}{fg.RED}ID en double!{fg.OFF}")
        for i, c in enumerate(counts):
            if c != 1: print(f"{values[i]} x {c}")
    return

def read_excel(xlFile,decimal,name):
    """
    entrée :
        - xlFile = nom du ficher excel
        - decimal = "." ou "," selon le cas
        - name = label du dataframe, utilisé pour l'affichage
    sortie :
        - le dataframe qui contient l'intégralité du fichier excel
        - le nombre de lignes de ce tableau (à l'exclusion de l'en-tête des colonnes)
    affichage : 
        - statistiques descriptives (describe) de toutes les colonnes, y compris celles qui ne contiennent pas de valeurs numériques 
    """
    print(f"{hl.BOLD}{fg.BLUE}{name}{fg.OFF}")
    print(f"Reading... {xlFile}")
    df=pd.read_excel(xlFile,decimal=decimal)

    #pd.set_option('display.max_rows', 12)
    #display(dfCC1)
    display(df.describe(include='all'))
    return df, df.shape[0]

def ReplaceABSByNan(df,nomCC):
    """
    entrée : 
        - dataframe qui contient les notes
        - nom des colonnes qui contiennent les notes
    sortie
        - nouveau dataframe où toutes les notes des colonnes nomCC = ABS sont remplacées par des nan
    """
    dfcopy = df.copy()
    for nom in nomCC:
        # correction introduite le 24/01/2026
        # dfcopy[nom].mask((dfcopy[nom] == "ABS") | (dfcopy[nom] == "abs") | (dfcopy[nom] == "Abs"), np.nan ,inplace=True)
        # On convertit en chaîne, on met en majuscules, et on compare à "ABS"
        # L'assignation directe (df[nom] = ...) évite le ChainedAssignmentError
        dfcopy[nom] = dfcopy[nom].mask(dfcopy[nom].astype(str).str.upper() == "ABS", np.nan)
    return dfcopy

# deprecated à cause de la nouvelle version de pandas (26/01/2026)
def ReplaceNanBy0OLD(df,nomCC):
    """
    entrée :
        - df = dataframe avec les notes
        - nomCC = liste des en-têtes de colonnes qui contiennt les notes
    sortie :
        - le dataframe avec Nan remplacé par 0 pour chaque étudiant qui a au moins une note de CC 
    """
    dfCopy = df.copy()
    for nom in nomCC:
        nomCCred = nomCC.copy()
        nomCCred.remove(nom)
        mask = pd.DataFrame([False]*dfCopy.shape[0],index=dfCopy.index,columns=["mask"])
        for nomred in nomCCred:
            mask["mask"] = (mask["mask"] | dfCopy[nomred].notnull())
        mask["mask"] = (mask["mask"] & dfCopy[nom].isna())
        # correction le 24/01/2026
        # dfCopy[nom].mask(mask["mask"],0.0,inplace=True)
        dfCopy[nom] = dfCopy[nom].mask(mask["mask"], 0.0)
    return dfCopy
    
def ReplaceNanBy0(df, nomCC):
    dfCopy = df.copy()
    for nom in nomCC:
        # 1. On identifie les autres colonnes
        autres_cols = [c for c in nomCC if c != nom]
        
        # 2. On crée le masque : 
        # (Au moins une autre note n'est pas nulle) ET (La note actuelle est nulle)
        condition = dfCopy[autres_cols].notnull().any(axis=1) & dfCopy[nom].isna()
        
        # 3. Application directe (On s'assure que la colonne accepte les flottants)
        dfCopy[nom] = dfCopy[nom].astype(float) 
        dfCopy[nom] = dfCopy[nom].mask(condition, 0.0)
        
    return dfCopy

def dropColumnsByIndex(df,idropC):
    """
    entrée :
        - df = dataframe dont on veut supprimer des colonnes
        - idropC = indices des colonnes dont on veut se débarasser
    sortie :
        - dfCleaned = dataframe originel dont les colonnes indexées par idropC ont été supprimées
    """
    listC = list(df.columns)
    dropC = [listC[idropC[i]] for i in range(len(idropC))]
    print(f"On va se débarrasser des colonnes {dropC}")
    dfCleaned = df.drop(dropC,axis=1)
    return dfCleaned

def CreationDfADMAJGH(df,note,Seuil,prefix,MD,Parc,IDApoG):
    """
    entrée : 
        - df = dataframe avec les mentions/parcours/moyennes
        - note = nom de la colonne qui contient la moyenne globale
        - Seuil = seuil de réussite pour départager ADM & AJ
        - prefix = préfixe du nom de fichier temporaire, incluant ou nom le nom d'un sous-répertoire de sauvegarde
        - MD = nom de la colonne qui contient la dénomination simplifiée de la mention de diplôme ("MentionD")
        - Parc = nom de la colonne qui contient le parcours
        - IDApoG = nom de la colonne qui contient l'ID des étudiants

    sortie : 
        - dfADM = dataframe des admis
        - dfAJ = dataframe des ajournés
        - dfGhosts = dataframe des fantômes (aka Ghosts ; i.e. n'ont passé aucun des CC)
        
    affichage : stats rapides (describe & sum) de chacun des sous-ensembles ADM, AJ, Ghosts
    
    sauvegarde : fichier excel tmp{Note}.xlsx avec 3 onglets (ADM, AJ, Ghosts)
    
    """
    print(f"{hl.BOLD}{bg.LIGHTRED}Construction des dataframes avec les ADM & les AJ sur la base de la {note} >= ou < à {Seuil}{fg.OFF}")
    print(f"{hl.BOLD}{bg.LIGHTRED}Construction également d'un dataframe 'Ghosts', c'est-à-dire avec les fantômes i.e. n'ont passé aucun des CC{fg.OFF}")
    #dataframe reçus
    dfADM = df[df[note]>=Seuil]
    #dataframe ajournés
    dfAJ = df[df[note]<Seuil]
    #dataframe fantômes
    dfGhosts = df[df[note].isnull()]
    
    Ftmp = prefix+'tmp'+note+'.xlsx'

    exceltest = pd.ExcelWriter(Ftmp, engine='xlsxwriter')
    dfADM.to_excel(exceltest, sheet_name='ADM')
    dfAJ.to_excel(exceltest, sheet_name='AJ')
    dfGhosts.to_excel(exceltest, sheet_name='Fantomes')
    exceltest.close()

    print(f"{hl.BOLD}{fg.BLUE}Total{fg.OFF}")
    display(df.groupby(MD)[note].describe().style.format("{0:.1f}"))
    print(f"{hl.BOLD}{fg.BLUE}Admis{fg.OFF}")
    display(dfADM.groupby(MD)[note].describe().style.format("{0:.1f}"))
    print(f"{hl.BOLD}{fg.BLUE}Ajournés{fg.OFF}")
    display(dfAJ.groupby(MD)[note].describe().style.format("{0:.1f}"))

    print(f"{hl.BOLD}{fg.BLUE}Total{fg.OFF}")
    display(df.groupby(MD)[note].describe().sum())
    print(f"{hl.BOLD}{fg.BLUE}Admis{fg.OFF}")
    display(dfADM.groupby(MD)[note].describe().sum())
    print(f"{hl.BOLD}{fg.BLUE}Ajournés{fg.OFF}")
    display(dfAJ.groupby(MD)[note].describe().sum())
    print(f"{hl.BOLD}{fg.BLUE}Fantômes{fg.OFF}")
    display(dfGhosts.groupby(MD)[IDApoG].count())
    return dfADM, dfAJ, dfGhosts

def StatsRéussiteParMentionOuParcours(dfT, dfADM, dfAJ, dfGhosts, Category, note):
    """
    entrée :
        -   dfT = dataframe qui contient toutes les notes, ainsi qu'au moins une catégorisation (exemple : Mention ou Parcours ou Section etc...)
        - dfADM = dataframe qui contient uniquement les étudiants admis
        -  dfAJ = dataframe qui contient uniquement les étudiants ajournés (sans les fantômes)
        - dfGhosts = dataframe qui contient la liste des fantômes
        - Category = nom de la colonne sur laquelle on veut faire des analyses statistiques
        - note = nom de la colonne qui contient la note qu'on veut analyser par catégorie (généralement la moyenne finale)

    sortie :
        - dfStats

    affichage :    
    """
    def incrementer(row,Category,Catunique,note,NN,Presents="Presents"):
        """
        conçu pour pour une analyse ligne par ligne
        si Category = une des catégories de Catunique NN[de cette catégorie] +=1
        nécessite au préalable
            - de fabriquer la liste des catégories uniques (df[Category].unique())
            - d'initialiser à 0 un tableau qui a la même dimension que Catunique

        entrée :
            - row = ligne d'un dataframe à analyser
            - Category = nom de la colonne qui contient les catégories à comptabiliser
            - Catunique = liste exhaustive des catégories du dataframe analysé
            - note = nom de la colonne qui contient les notes
            - NN = tableau dont la colonne qui correspond à l'une des catégories uniques pour cet étudiant est incrémenté de 1
            - Presents (valeur par défaut = "Present") : compte les présents uniquement
              sinon ce sont les fantômes qui sont comptés

        """
        i = 0
        for C in Catunique:
            if (((row[Category] == C) & (not np.isnan(row[note]))) & (Presents=="Presents")) |\
               (((row[Category] == C) & (np.isnan(row[note]))) & (Presents!="Presents")):
                NN[i] +=1
                #print(i,row[CategoryInRow],C,NN[i],row[note],np.isnan(row[note]))
            i += 1
        return
    
    print(f"{hl.BOLD}-- Statistiques uniquement sur les présents --{fg.OFF}")
    CatUnique = dfT[Category].unique()
    print(f"{Category} = {CatUnique}")
    # création des tableaux pour comptabiliser les reçus (NRP), ajournés (NAJP) par parcours
    # NTP contient le nombre total d'étudiants par parcours
    NT = np.zeros(len(CatUnique))
    NADM = np.zeros(len(CatUnique))
    NAJ = np.zeros(len(CatUnique))
    NGH = np.zeros(len(CatUnique))
    dfT.apply(lambda row: incrementer(row, Category, CatUnique, note, NT), axis=1)
    dfADM.apply(lambda row: incrementer(row, Category, CatUnique, note, NADM), axis=1)
    dfAJ.apply(lambda row: incrementer(row, Category, CatUnique, note, NAJ), axis=1)
    dfGhosts.apply(lambda row: incrementer(row, Category, CatUnique, note, NGH, "Fantomes"), axis=1)
    dfADMAJ = pd.concat([dfADM,dfAJ])
#    display(dfADMAJ.describe(include='all'))
#    print(dfADMAJ[Moyenne].mean())

    print(f"  NT/Cat = {NT}")
    print(f"{fg.GREEN}NADM/Cat = {NADM}{fg.OFF}")
    print(f"{fg.RED} NAJ/Cat = {NAJ}{fg.OFF}")
    print(f"{fg.LIGHTGRAY} NGH/Cat = {NGH}{fg.OFF}")

    print(f"{hl.BOLD}{fg.GREEN}Reçus par {Category}{fg.OFF}")
    i = 0
    nADM = 0
    nAJ = 0
    nTOT = 0
    nGH = 0
    Moy = np.zeros(len(CatUnique))
    StD = np.zeros(len(CatUnique))
    Med = np.zeros(len(CatUnique))
    MoyGlob = np.zeros(len(CatUnique))
    MoyGlobT = 0
    for C in CatUnique:
        print(f"{hl.BOLD}{C:>20}{fg.OFF} = {hl.BOLD}{fg.GREEN}{100*NADM[i]/NT[i]:.1f} %{fg.OFF} ({fg.RED}AJ : {NAJ[i]:3.0f}{fg.OFF},"\
              f" {fg.GREEN}ADM : {NADM[i]:3.0f}{fg.OFF}, Tot : {NT[i]:3.0f}      {fg.LIGHTGRAY}+Fantômes : {NGH[i]:3.0f}){fg.OFF}")
#        display(dfADM.loc[dfADM[Category] == C][note].sum())
        Moy[i] = (dfADM.loc[dfADM[Category] == C][note].sum()+dfAJ.loc[dfAJ[Category] == C][note].sum())
        MoyGlob[i] = Moy[i]
        MoyGlobT += Moy[i]
        Moy[i] = np.round(Moy[i] / (NADM[i]+NAJ[i]),2)
        MoyGlob[i] = np.round(MoyGlob[i] / (NADM[i]+NAJ[i]+NGH[i]),2)
        StD[i] = np.round(dfADMAJ.loc[dfADMAJ[Category] == C][note].std(),1)
        Med[i] = np.round(dfADMAJ.loc[dfADMAJ[Category] == C][note].median(),1)
#        print("Moyennes ",Moy[i],MoyGlob[i])
        nADM += NADM[i]
        nAJ += NAJ[i]
        nGH += NGH[i]
        nTOT += NT[i]
        i+=1
    print(f"{hl.BOLD}{fg.GREEN}       ADM = {nADM:3.0f}{fg.OFF}")
    print(f"{hl.BOLD}{fg.RED}        AJ = {nAJ:3.0f}{fg.OFF}")
    print(f"{hl.BOLD}{fg.BLACK}       TOT = {nTOT:3.0f}{fg.OFF}")
    print(f"{hl.BOLD}{fg.LIGHTGRAY}(+Fantomes = {nGH:3.0f}){fg.OFF}")
    MoyT = np.round(dfADMAJ[note].mean(),2)
    StDT = np.round(dfADMAJ[note].std(),1)
    MedT = np.round(dfADMAJ[note].median(),1)
    MoyGlobT = np.round(MoyGlobT/(nTOT+nGH),2)
    rowTotal = [NT.sum(),NADM.sum(),NAJ.sum(),np.round(100*NADM.sum()/NT.sum(),1),MoyT,StDT,MedT,NGH.sum(),NT.sum()+NGH.sum(),np.round(100*NADM.sum()/(NT.sum()+NGH.sum()),1),MoyGlobT]
    defCol = ["Présents","ADM","AJ","Taux ADM (présents)","Moy.","StDev","Med.","Fantômes","Total","Taux ADM (tot.)","Moy."]
    dfStats=pd.DataFrame(zip(NT,NADM,NAJ,np.round(100*NADM/NT,1),Moy,StD,Med,NGH,NT+NGH,np.round(100*NADM/(NT+NGH),1),MoyGlob),index=CatUnique,columns=defCol)
    rowTotal = pd.DataFrame([rowTotal],index=["Total"],columns=dfStats.columns)
    dfStats=pd.concat([dfStats,rowTotal])
    dfStats.style.set_caption(f"Statistiques par {Category}")
    return dfStats

def StatsRéussiteParMentionOuParcoursWithAb(dfT, dfADM, dfAJ, dfGhosts, dfAb, Category, note):
    """
    entrée :
        -   dfT = dataframe qui contient toutes les notes, ainsi qu'au moins une catégorisation (exemple : Mention ou Parcours ou Section etc...)
        - dfADM = dataframe qui contient uniquement les étudiants admis
        -  dfAJ = dataframe qui contient uniquement les étudiants ajournés (sans les fantômes)
        - dfGhosts = dataframe qui contient la liste des fantômes
        - dfAb = dataframe qui contient la liste des étudiants qui ont abandonné
        - Category = nom de la colonne sur laquelle on veut faire des analyses statistiques
        - note = nom de la colonne qui contient la note qu'on veut analyser par catégorie (généralement la moyenne finale)

    sortie :
        - dfStats

    affichage :    
    """
    def incrementer(row,Category,Catunique,note,NN,Presents="Presents"):
        """
        conçu pour pour une analyse ligne par ligne
        si Category = une des catégories de Catunique NN[de cette catégorie] +=1
        nécessite au préalable
            - de fabriquer la liste des catégories uniques (df[Category].unique())
            - d'initialiser à 0 un tableau qui a la même dimension que Catunique

        entrée :
            - row = ligne d'un dataframe à analyser
            - Category = nom de la colonne qui contient les catégories à comptabiliser
            - Catunique = liste exhaustive des catégories du dataframe analysé
            - note = nom de la colonne qui contient les notes
            - NN = tableau dont la colonne qui correspond à l'une des catégories uniques pour cet étudiant est incrémenté de 1
            - Presents (valeur par défaut = "Present") : compte les présents uniquement
              sinon ce sont les fantômes qui sont comptés

        """
        i = 0
        for C in Catunique:
            if (((row[Category] == C) & (not np.isnan(row[note]))) & (Presents=="Presents")) |\
               (((row[Category] == C) & (np.isnan(row[note]))) & (Presents!="Presents")):
                NN[i] +=1
                #print(i,row[CategoryInRow],C,NN[i],row[note],np.isnan(row[note]))
            i += 1
        return
    def incrementerAbandons(row,Category,Catunique,NN):
        i = 0
        for C in Catunique:
            if (row[Category] == C):
                NN[i] +=1
            i += 1
        return
    
    print(f"{hl.BOLD}-- Statistiques uniquement sur les présents --{fg.OFF}")
    CatUnique = dfT[Category].unique()
    print(f"{Category} = {CatUnique}")
    # création des tableaux pour comptabiliser les reçus (NRP), ajournés (NAJP) par parcours
    # NTP contient le nombre total d'étudiants par parcours
    NT = np.zeros(len(CatUnique))
    NADM = np.zeros(len(CatUnique))
    NAJ = np.zeros(len(CatUnique))
    NAb = np.zeros(len(CatUnique))
    NGH = np.zeros(len(CatUnique))
    dfT.apply(lambda row: incrementer(row, Category, CatUnique, note, NT), axis=1)
    dfADM.apply(lambda row: incrementer(row, Category, CatUnique, note, NADM), axis=1)
    dfAJ.apply(lambda row: incrementer(row, Category, CatUnique, note, NAJ), axis=1)
    dfAb.apply(lambda row: incrementerAbandons(row, Category, CatUnique, NAb), axis=1)
    dfGhosts.apply(lambda row: incrementer(row, Category, CatUnique, note, NGH, "Fantomes"), axis=1)
    dfADMAJ = pd.concat([dfADM,dfAJ])
#    display(dfADMAJ.describe(include='all'))
#    print(dfADMAJ[Moyenne].mean())

    print(f"  NT/Cat = {NT}")
    print(f"{fg.GREEN}NADM/Cat = {NADM}{fg.OFF}")
    print(f"{fg.RED} NAJ/Cat = {NAJ}{fg.OFF}")
    print(f"{fg.RED} NAb/Cat = {NAb}{fg.OFF}")
    print(f"{fg.LIGHTGRAY} NGH/Cat = {NGH}{fg.OFF}")

    print(f"{hl.BOLD}{fg.GREEN}Reçus par {Category}{fg.OFF}")
    i = 0
    nADM = 0
    nAJ = 0
    nAb = 0
    nTOT = 0
    nGH = 0
    Moy = np.zeros(len(CatUnique))
    StD = np.zeros(len(CatUnique))
    Med = np.zeros(len(CatUnique))
    MoyGlob = np.zeros(len(CatUnique))
    MoyGlobT = 0
    for C in CatUnique:
        print(f"{hl.BOLD}{C:>20}{fg.OFF} = {hl.BOLD}{fg.GREEN}{100*NADM[i]/NT[i]:.1f} %{fg.OFF} ({fg.RED}AJ : {NAJ[i]:3.0f} dont {NAb[i]:3.0f} Ab{fg.OFF},"\
              f" {fg.GREEN}ADM : {NADM[i]:3.0f}{fg.OFF}, Tot : {NT[i]:3.0f}      {fg.LIGHTGRAY}+Fantômes : {NGH[i]:3.0f}){fg.OFF}")
#        display(dfADM.loc[dfADM[Category] == C][note].sum())
        Moy[i] = (dfADM.loc[dfADM[Category] == C][note].sum()+dfAJ.loc[dfAJ[Category] == C][note].sum())
        MoyGlob[i] = Moy[i]
        MoyGlobT += Moy[i]
        Moy[i] = np.round(Moy[i] / (NADM[i]+NAJ[i]),2)
        MoyGlob[i] = np.round(MoyGlob[i] / (NADM[i]+NAJ[i]+NGH[i]),2)
        StD[i] = np.round(dfADMAJ.loc[dfADMAJ[Category] == C][note].std(),1)
        Med[i] = np.round(dfADMAJ.loc[dfADMAJ[Category] == C][note].median(),1)
#        print("Moyennes ",Moy[i],MoyGlob[i])
        nADM += NADM[i]
        nAJ += NAJ[i]
        nAb += NAb[i]
        nGH += NGH[i]
        nTOT += NT[i]
        i+=1
    print(f"{hl.BOLD}{fg.GREEN}       ADM = {nADM:3.0f}{fg.OFF}")
    print(f"{hl.BOLD}{fg.RED}        AJ = {nAJ:3.0f}{fg.OFF}")
    print(f"{hl.BOLD}{fg.RED}   dont Ab = {nAb:3.0f} (= abandons){fg.OFF}")
    print(f"{hl.BOLD}{fg.BLACK}       TOT = {nTOT:3.0f}{fg.OFF}")
    print(f"{hl.BOLD}{fg.LIGHTGRAY}(+Fantomes = {nGH:3.0f}){fg.OFF}")
    MoyT = np.round(dfADMAJ[note].mean(),2)
    StDT = np.round(dfADMAJ[note].std(),1)
    MedT = np.round(dfADMAJ[note].median(),1)
    MoyGlobT = np.round(MoyGlobT/(nTOT+nGH),2)
    rowTotal = [NT.sum(),NADM.sum(),NAJ.sum(),NAb.sum(),np.round(100*NADM.sum()/NT.sum(),1),MoyT,StDT,MedT,NGH.sum(),NT.sum()+NGH.sum(),np.round(100*NADM.sum()/(NT.sum()+NGH.sum()),1),MoyGlobT]
    defCol = ["Présents","ADM","AJ","dont Ab","Taux ADM (présents)","Moy.","StDev","Med.","Fantômes","Total","Taux ADM (tot.)","Moy."]
    dfStats=pd.DataFrame(zip(NT,NADM,NAJ,NAb,np.round(100*NADM/NT,1),Moy,StD,Med,NGH,NT+NGH,np.round(100*NADM/(NT+NGH),1),MoyGlob),index=CatUnique,columns=defCol)
    rowTotal = pd.DataFrame([rowTotal],index=["Total"],columns=dfStats.columns)
    dfStats=pd.concat([dfStats,rowTotal])
    dfStats.style.set_caption(f"Statistiques par {Category}")
    return dfStats

def ApplySecondeChance(df,nomCC,nomCCSC,nomCCdelaSC):
    """
    modification du dataframe df
    
    entrée :
        - df = dataframe auquel on va ajouter une nouvelle colonne nomCCSC qui va contenir la note de la colonne nomCC soit celle de la colonne nomCCdelaSC, si cell-ci est supérieure à la notede nomCC
        - nomCC = nom de la colonne à laquelle on applique la seconde chance
        - nomCCSC = nom de la nouvelle colonne qui contient la note du CC après application de la seconde chance
        - nomCCdelaSC = nom de la colonne qui contient la note "seconde chance
    sortie :
        - moyenne après application de la seconde chance
        - écart-type  après application de la seconde chance
    """
    df[nomCCSC] = df[[nomCC,nomCCdelaSC]].max(axis=1)
    moySC = pd.to_numeric(df[nomCCSC], errors='coerce').mean()
    stdSC = pd.to_numeric(df[nomCCSC], errors='coerce').std()
    return moySC, stdSC

def ComparaisonMoyennesDMCC(df,nomCC1,nomCC2,SeuilBadDMCC1,SeuilGoodDMCC1):
    """
    entrée :
        - df = dataframe avec uniquement les étudiants AJ ou ADM, c'est-à-dire qu'il n'y a aucun fantôme
        - nomCC1 = en-tête de la colonne de df qui contient la note du premier CC
        - nomCC2 = en-tête de la colonne de df qui contient la note du 2nd CC
        - SeuilBadDMCC1 = Seuil en-dessous duquel la note au nomCC1 est considérée comme médiocre
        - SeuilGoodDMCC1 = Seuil au-deçà duquel la note au nomCC1 est considérée comme bonne
    affichage :
        - moyenne au nomCC2 de la cohorte d'étudiant(e)s
            - en-dessous du SeuilBadDMCC1 au nomCC1
            - au-dessus du SeuilGoodDMCC1 au nomCC1
            - entre les deux seuils au nomCC1
        - jointplot entre nomCC1 et nomCC2 uniquement pour les étudiant(e)s dont la note au CC1 est considérée comme médiocre
    """
    print(f"{hl.BOLD}Corrélation entre {fg.BLUE}{nomCC1}{fg.BLACK} et {fg.BLUE}{nomCC2}{fg.BLACK} ?{hl.OFF}")
    dfBadCC1 = df[df[nomCC1] < SeuilBadDMCC1]
    dfAvCC1 = df[(df[nomCC1] >= SeuilBadDMCC1) & (df[nomCC1] <= SeuilGoodDMCC1)]
    dfGCC1 = df[df[nomCC1] > SeuilGoodDMCC1]
    MoyenneAuCC2desBadCC1 = np.round(dfBadCC1[nomCC2].mean(),1)
    MoyenneAuCC2desAvCC1 = np.round(dfAvCC1[nomCC2].mean(),1)
    MoyenneAuCC2desGoodCC1 = np.round(dfGCC1[nomCC2].mean(),1)
    MoyenneAuCC1 = np.round(df[nomCC1].mean(),1)
    MoyenneAuCC2 = np.round(df[nomCC2].mean(),1)
    print(f"Les étudiant(e)s qui ont eu moins de {SeuilBadDMCC1}/20 au {nomCC1} ont en moyenne {MoyenneAuCC2desBadCC1}/20 au {nomCC2}")
    print(f"Les étudiant(e)s qui ont eu entre {SeuilBadDMCC1}/20 et {SeuilGoodDMCC1}/20 au {nomCC1} ont en moyenne {MoyenneAuCC2desAvCC1}/20 au {nomCC2}")
    print(f"Les étudiant(e)s qui ont eu plus de {SeuilGoodDMCC1}/20 au {nomCC1} ont en moyenne {MoyenneAuCC2desGoodCC1}/20 au {nomCC2}")
    print(f"Pour rappel, la moyenne au {nomCC1} = {MoyenneAuCC1}/20 et celle au {nomCC2} = {MoyenneAuCC2}/20")
    sns.jointplot(x = nomCC1, y = nomCC2, data = dfBadCC1)
    plt.show()
    return

####################################################################################################################################
#                    G R A P H E S
####################################################################################################################################

def Histogrammes(df,nomCC,Moyenne,NomGraphique,w,moy,std,moyT,stdT,legende):
    """
    entrée :
        - df = dataframe qui contient ID, Noms, Prénoms, notes de CC, et Moyenne pondérée
        - nomCC = liste qui contient noms d'en-têtes des colonnes qui contiennent les notes de CC
        - Moyenne = nom de l'en-tête de la colonne qui contient la moyenne
        - NomGraphique = nom du fichier png qui va contenir la figure
        - w = liste qui contient les coeffs des CC
        - moy = liste qui contient la moyenne de chaque CC
        - std = liste qui contient l'écart-type calculé pour chaque CC
        - moyT = moyenne des 4 CC
        - stdT = écart-type calculé pour la note globale
        - legende = titre qui sera affiché sur l'histogramme principal
    affichage :
        - 1 petit histogramme /CC
        - 1 grand histogramme avec la moyenne
    sauvegarde :
        - fichier graphique 'NomGraphique' avec 1 petit histogramme par CC + 1 grand histogramme avec la moyenne
    """
    
    import matplotlib.gridspec as gridspec
    
    wt = sum(w)
    
    fig = plt.figure(figsize=(18, 12))
    plt.rcParams["font.size"] = (14) #font size
    gs = gridspec.GridSpec(2, 4, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1])

    fontpropertiesT = {'family':'sans-serif', 'weight' : 'bold', 'size' : 16}
    fontpropertiesA = {'family':'sans-serif', 'weight' : 'bold', 'size' : 18}

    sns.set_style("whitegrid")
    ax00=plt.subplot(gs[0,0])
    fig00=sns.histplot(data=df, x=nomCC[0], discrete=True, kde=True, color="#3e95ff", alpha=1.0)
    ax00.set_xlabel("note / 20",fontsize=16)
    ax00.set_ylabel("Count",fontsize=16)
    ax00.set_title(f"{nomCC[0]} ({w[0]}%). <N> = {moy[0]:.1f}, $\sigma$ = {std[0]:.1f}",color = "red", font=fontpropertiesT)

    ax01=plt.subplot(gs[0,1])
    fig01=sns.histplot(data=df, x=nomCC[1], discrete=True, kde=True, color="#3e95ff", alpha=1.0)
    ax01.set_xlabel("note / 20",fontsize=16)
    ax01.set_ylabel("Count",fontsize=16)
    ax01.set_title(f"{nomCC[1]} ({w[1]}%) <N> = {moy[1]:.1f}, $\sigma$ = {std[1]:.1f}",color = "red", font=fontpropertiesT)

    ax02=plt.subplot(gs[0,2])
    fig02=sns.histplot(data=df, x=nomCC[2], discrete=True, kde=True, color="#3e95ff", alpha=1.0)
    ax02.set_xlabel("note / 20",fontsize=16)
    ax02.set_ylabel("Count",fontsize=16)
    ax02.set_title(f"{nomCC[2]} ({w[2]}%) <N> = {moy[2]:.1f}, $\sigma$ = {std[2]:.1f}",color = "red", font=fontpropertiesT)

    ax03=plt.subplot(gs[0,3])
    fig03=sns.histplot(data=df, x=nomCC[3], discrete=True, kde=True, color="#3e95ff", alpha=1.0)
    ax03.set_xlabel("note / 20",fontsize=16)
    ax03.set_ylabel("Count",fontsize=16)
    ax03.set_title(f"{nomCC[3]} ({w[3]}%) <N> = {moy[3]:.1f}, $\sigma$ = {std[3]:.1f}",color = "red", font=fontpropertiesT)

    axTot=plt.subplot(gs[1,0:4])
    figTot=sns.histplot(data=df, x=Moyenne, discrete=True, kde=True, color="#737cff", alpha=1.0, stat='count', label = legende)
    axTot.set_xlabel("note / 20",fontsize=16)
    axTot.set_ylabel("Count",fontsize=16)
    axTot.set_xticks(np.arange(0,19,2))
    axTot.set_title(f"Moyenne ({wt}%).  <N> = {moyT:.1f}, $\sigma$ = {stdT:.1f}",color = "red", font=fontpropertiesT)
    axTot.legend()

    fig.savefig(NomGraphique, dpi=300)
    plt.show()
    return

def kdePlotByMentionEtParcours(df,Moyenne,Mention,Parcours,NomGraphique):
    """
    entrée :
        - df = dataframe qui contient ID, Noms, Prénoms, notes de CC, Moyenne pondérée, Mention et Parcours
        - Mention = nom de l'en-tête de la colonne qui contient la Mention de diplôme simplifiée d'un étudiant
        - Parcours = nom de l'en-tête de la colonne qui contient le parcours suivi par un étudiant
    affichage :
        - 1 graphe avec des plots kde par Mention
        - 1 graphe avec des plots kde par Parcours
    sauvegarde :
        - fichier graphique 'NomGraphique' avec les 2 graphes
    """
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(18, 12))
    plt.rcParams["font.size"] = (14) #font size
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], width_ratios=[1])

    fontpropertiesT = {'family':'sans-serif', 'weight' : 'bold', 'size' : 16}
    fontpropertiesA = {'family':'sans-serif', 'weight' : 'bold', 'size' : 18}

    sns.set_style("whitegrid")
    axP=plt.subplot(gs[0,0])
    hue_order = np.sort(df[Mention].unique())
    figP=sns.kdeplot(data=df.sort_values(by = Mention), x=Moyenne, alpha=0.4, hue=Mention, bw_adjust=1, cut=0, fill=True, linewidth=3, hue_order=hue_order)
    axP.set_xlabel("note / 20",fontsize=16)
    axP.set_ylabel("Count",fontsize=16)
    #axP.set_title(f"Total/Mention. <N> = {moy1:.1f}, $\sigma$ = {std1:.1f}",color = "red", font=fontpropertiesT)

    axO=plt.subplot(gs[1,0])
    hue_order = np.sort(df[Parcours].unique())
    figO=sns.kdeplot(data=df.sort_values(by = Parcours), x=Moyenne, alpha=0.4, hue=Parcours, bw_adjust=1, cut=0, fill=True, linewidth=3, hue_order=hue_order)
    axO.set_xlabel("note / 20",fontsize=16)
    axO.set_ylabel("Count",fontsize=16)
    axO.set_xticks(np.arange(0,19,2))
    #axO.set_title(f"Total/Parcours.  <N> = {moyT:.1f}, $\sigma$ = {stdT:.1f}",color = "red", font=fontpropertiesT)

    fig.savefig(NomGraphique, dpi=300)
    plt.show()
    return

def BoiteAMoustachesByMentionEtParcours(df, Moyenne, Mention, Parcours, NomGraphique):
    import matplotlib.gridspec as gridspec
    
    plt.style.use('seaborn-v0_8-white')
    fig = plt.figure(figsize=(16, 12))
    plt.rcParams["font.size"] = (14) #font size
    nP = (df[Mention].unique().shape[0])
    nO = (df[Parcours].unique().shape[0])
    gs = gridspec.GridSpec(2, 2, height_ratios=[1,2*nO/nP], width_ratios=[1,1])

    fontpropertiesT = {'family':'sans-serif', 'weight' : 'bold', 'size' : 16}
    fontpropertiesA = {'family':'sans-serif', 'weight' : 'bold', 'size' : 18}
    
    sns.set_style("whitegrid")
    axP=plt.subplot(gs[:,0])
    figP=sns.boxplot(data=df.sort_values(by = Mention), x=Moyenne, y=Mention, palette='tab20c')
    axP.set_xlabel("note / 20",fontsize=16)
    axP.set_ylabel("",fontsize=16)
    axP.set_xticks(np.arange(0,20,2))
    #axP.set_title(f"Total/Mention. <N> = {moy1:.1f}, $\sigma$ = {std1:.1f}",color = "red", font=fontpropertiesT)
    plt.xticks(fontweight='bold',fontsize=16)
    plt.yticks(fontweight='bold',fontsize=16)
    
    axO=plt.subplot(gs[-1,-1])
    figO=sns.boxplot(data=df.sort_values(by = Parcours), x=Moyenne, y=Parcours, palette='tab20c')
    axO.set_xlabel("note / 20",fontsize=16)
    axO.set_ylabel("",fontsize=16)
    axO.yaxis.tick_right()
    axO.set_xticks(np.arange(0,20,2))
    plt.xticks(fontweight='bold',fontsize=16)
    plt.yticks(fontweight='bold',fontsize=16)

    plt.savefig(NomGraphique, dpi=300)
    plt.show()
    return


def BoiteAMoustachesByMentionEtParcoursHue(df, Moyenne, Mention, Parcours, NomGraphique):
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(18, 12))
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams["font.size"] = (14) #font size

    fontpropertiesT = {'family':'sans-serif', 'weight' : 'bold', 'size' : 16}
    fontpropertiesA = {'family':'sans-serif', 'weight' : 'bold', 'size' : 18}

    sns.set_style("whitegrid")
    hue_order = np.sort(df[Parcours].unique())
    sns.boxplot(data=df.sort_values(by = Mention), x=Moyenne, y=Mention, hue=Parcours, palette='hsv', hue_order=hue_order)
    plt.xlabel("note / 20",fontsize=16)
    plt.ylabel("",fontsize=16)
    plt.xticks(np.arange(0,20,2))
    plt.xticks(fontweight='bold',fontsize=16)
    plt.yticks(fontweight='bold',fontsize=16)
    plt.legend(loc='lower left')
    fig.savefig(NomGraphique, dpi=300)
    plt.show()
    return

def plotTauxADMasBars(dfwoSC,dfwSC,xCol,hueCol,NomGraphique):
    """
    entrée :
        - dfwoSC = dataframe contenant les stats ADM/AJ/Ghosts avant l'application de la seconde chance
        - dfwSC = dataframe contenant les stats ADM/AJ/Ghosts après l'application de la seconde chance 
        - xCol = nom de la colonne qui contient les taux de réussite
        - hueCol = nom de la colonne qui contient les paramètres avant ou après seconde chance
        - NomGraphique = nom du fichier png qui va être sauvé 
    affichage :
        - bar plot de seaborn
    sauvegarde :
        fichier graphique 'NomGraphique' au format png
    """
    from matplotlib.colors import LinearSegmentedColormap
    plt.style.use('seaborn-v0_8-whitegrid')
    my_colors = ['#ff696b','#00aa7f']
    my_cmap = LinearSegmentedColormap.from_list("mycolors",my_colors)
    df = pd.concat([dfwoSC, dfwSC])
    nbars=df.shape[0]
    fig = plt.figure(figsize=(18, nbars*0.6))
    plt.rcParams["font.size"] = (16) #font size
    ax = sns.barplot(data=df,y=df.index.values,x = xCol,hue=hueCol,palette=my_colors)
    ax.bar_label(ax.containers[0],fontweight='bold',fontsize=18)
    ax.bar_label(ax.containers[1],fontweight='bold',fontsize=18)
    plt.xticks(fontweight='bold',fontsize=14)
    plt.yticks(fontweight='bold',fontsize=16)
    plt.xlim(0,100)
    fig.savefig(NomGraphique, dpi=300)
    plt.show()
    return

def StackedBarPop(df,ListCols,ylabel,NomGraphique):
    """
    entrée :
        - df : dataframe contenant l'analyse statistique globale, dont les moyennes, effectifs, etc. i.e. le dataframe renvoyé par StatsRéussiteParMentionOuParcours()
        - ListCols = liste avecles labels des colonnes contenant les valeurs numériques qu'on veut tracer comme un histogramme empilé
        - ylabel = label de l'axe des y
        - NomGraphique = nom du fichier png qui va être sauvé
    affichage :
        - bar plot empilé de pandas
    sauvegarde :
        fichier graphique 'NomGraphique' au format png
    """
    from matplotlib.colors import LinearSegmentedColormap
    plt.style.use('seaborn-v0_8-whitegrid')
    my_colors = ['#b95651','#01bec3']
    my_cmap = LinearSegmentedColormap.from_list("mycolors",my_colors)
    nbars=df.shape[0]
    bplot = df[ListCols].plot(kind='bar',stacked=True, figsize=(nbars*1.5,10), fontsize=16, width=0.85, colormap=my_cmap, edgecolor='black')
    bplot.set_ylabel(ylabel,fontdict={'fontsize':18})
    bplot.bar_label(bplot.containers[0],label_type='center',color='w',fontweight='bold',fontsize=18)
    bplot.bar_label(bplot.containers[1],padding=5,fontweight='bold',fontsize=18)
    bplot.legend(fontsize=20)
    plt.xticks(fontweight='bold')
    plt.savefig(NomGraphique, dpi=300)
    plt.show()
    return

def StackedBarPopPO(dfRef,Mention,Parcours,NomGraphique):
    """
    entrée :
        - dfRef : dataframe contenant la liste des étudiants avec leur Mention et leur otpion, tel que généré par mentionD()
        - Mention = nom de la colonne qui contient le nom simplifié d'une Mention
        - Parcours = nom de la colonne qui contient la version simplifiée d'un Parcours
        - NomGraphique = nom du fichier png qui va être sauvé
    affichage :
        - histplot "hue" de seaborn, càd les effectifs par Parcours empilés pour chaque Mention
    sauvegarde :
        fichier graphique 'NomGraphique' au format png
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    nP = (dfRef[Mention].unique().shape[0])
    fig = plt.figure(figsize=(nP*2,8))
    hue_order = np.sort(dfRef[Parcours].unique())
    hplot = sns.histplot(data=dfRef, x=Mention, hue=Parcours, multiple="stack", stat='count', palette='terrain', hue_order=hue_order)
#    print(len(hplot.containers[0]))
    ntot = np.zeros(len(hplot.containers[0]))
    xtot = np.zeros(len(hplot.containers[0]))
#    print(ntot)
    for c in hplot.containers:
#        print(c[0].get_height())
        for i in range(nP):
#            print(c[i])
            x = c[i].get_x()
            y = c[i].get_y()
            n = c[i].get_height()
            ntot[i] += n
            xtot[i] = x
            if (n!=0) : hplot.annotate(n, (x+0.5, y+n/2), size = 16, weight="bold",ha="center",va="center",color="black")
#    print(ntot,xtot)
#    print(ntot.max())
    [hplot.annotate(ntot[i], (xtot[i]+0.5, ntot[i]+4), size = 20, weight="bold",ha="center",va="center",color="#407ba6") for i in range(nP)]
    plt.rcParams["font.size"] = (16) #font size
    plt.xticks(fontweight='bold',fontsize=16)
    plt.yticks(fontweight='bold',fontsize=16)
    plt.savefig(NomGraphique, dpi=300)
    plt.show()
    return

