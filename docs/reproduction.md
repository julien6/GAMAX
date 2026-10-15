# Reproduire les expériences GAMAX depuis un clone neuf

Ce tutoriel permet de **recompiler les six sources officielles, recréer les
références GAMA et mesurer de nouvelles comparaisons CPU**, sans aucun fichier de
l’ancien répertoire `results/`. Il ne lance rien automatiquement et ne nécessite
ni compte Grid’5000 ni GPU. Le nouveau parcours a été contrôlé par tests de son
orchestration et de ses gates ; aucune nouvelle campagne GAMA/JAX n’a été exécutée
pendant sa préparation. Le [bilan des contrôles](reproduction_preparation_checks.md)
précise ce qui a été vérifié et ce qui reste à exécuter. Les durées et gains sur votre matériel restent à mesurer.

## 1. Prérequis et installation

Utiliser **Linux x86_64 et Python 3.12**, comme l’évaluation CPU. Prévoir Internet
pour les paquets et la distribution GAMA, plusieurs Go libres pour GAMA/JDK et
environnement, puis davantage pour les traces. La durée complète peut être
importante ; commencer par la petite commande ci-dessous. Les ports WebSocket
locaux doivent être autorisés. Les mesures utilisent un CPU logique autorisé et
ne nécessitent pas de serveur GAMA déjà démarré.

Depuis la racine du clone :

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pip check
.venv/bin/python scripts/install_gama.py
```

L’installateur télécharge **GAMA 2025.06.4 Linux avec JDK**, contrôle le SHA-256
épinglé et l’installe dans `.tools/gama`. Il ne modifie pas Java système. Ne pas
substituer une autre version sans traiter cette modification comme une autre
expérience. Les versions Python/JAX/NumPy/SciPy du verrou sont celles de
l’environnement CPU utilisé ; ne pas mettre à jour silencieusement si un paquet
est indisponible pour votre plateforme.

Vérifier l’environnement :

```bash
.venv/bin/python -c 'import jax; print(jax.__version__, jax.devices())'
.venv/bin/python -m pytest -q
```

La suite pytest vérifie l’implémentation ; les comparaisons de trajectoires avec
le vrai moteur GAMA sont les étapes suivantes. Pour les seuls tests du nouveau
parcours, utiliser `pytest -q tests/test_reproduction.py`.

## 2. Voir le plan sans lancer de simulation

```bash
.venv/bin/python scripts/reproduce.py \
  --cases traffic --seeds 1 --repeats 1 --regime cold \
  -o results/reproduction-traffic
```

Sans `--execute`, la commande affiche les étapes et n’écrit aucun résultat,
ne démarre aucun moteur et ne contacte aucun serveur. Les noms de cas sont :
`sir_spatial`, `schelling`, `boids`, `traffic`, `predator_prey`, `ants`.
Le fichier [cpu-v0.json](../configs/reproduction/cpu-v0.json) fixe horizon,
optimisation, capacités et seeds. Le sous-ensemble réservé n’est jamais compilé.

## 3. Premier essai : trafic, une seed, avec et sans rendu

**La commande suivante exécute réellement GAMA et GAMAX.** Choisir un dossier de
sortie neuf : aucun résultat existant n’est écrasé.

```bash
.venv/bin/python scripts/reproduce.py \
  --cases traffic --seeds 1 --repeats 1 --regime cold \
  -o results/reproduction-traffic --execute
```

Elle compile la source officielle inchangée ; vérifie l’initialisation et les
100 cycles contre GAMA ; réémet le même module pour vérifier son identité ;
exécute les deux moteurs dans des processus neufs, sans rendu puis avec le même
rendu Pillow ; compare états, identités, pixels et fichiers sauvegardés ; produit
l’analyse. Le test fonctionnel sur une seed est volontairement petit : il ne
constitue pas une validation sur dix seeds.

Les flottants doivent satisfaire `abs(J-G) <= 1e-12 + 1e-10*abs(G)` ; les valeurs
discrètes et identités doivent être égales. Une divergence bloque la suite. Aucun
résultat favorable n’est déduit de la seule compilation. Les contrôles fonctionnels
supplémentaires (mutants, invariants, indépendance) restent décrits dans les
rapports techniques et ne sont pas implicitement certifiés par ce premier essai.

## 4. Reproduction CPU complète sur les six modèles

Le régime amorti réutilise équitablement une JVM/expérience GAMA et un exécutable
JAX. Il nécessite au moins deux seeds pour son contrôle de réinitialisation.

```bash
.venv/bin/python scripts/reproduce.py \
  --regime amortized -o results/reproduction-six-amortized --execute
```

Par défaut : six modèles, seeds 1–10, 100 cycles, trois répétitions internes,
deux modes et deux moteurs. Cela produit **60 trajectoires de fidélité**,
**18 trajectoires de contrôle de réutilisation native** (seed 1 → 2 → 1 par
modèle), puis **720 simulations mesurées dans 24 processus batch**, plus leurs
chauffes. Les 360 paires de simulations appartiennent à seulement 12 paires de
processus : ne pas les interpréter comme 360 démarrages indépendants.

Pour mesurer aussi les processus neufs :

```bash
.venv/bin/python scripts/reproduce.py \
  --regime both -o results/reproduction-six-both --execute
```

Cela ajoute 720 exécutions à froid. Pour un contrôle intermédiaire amorti :

```bash
.venv/bin/python scripts/reproduce.py \
  --cases traffic --seeds 1 2 --repeats 1 --regime amortized \
  -o results/reproduction-traffic-amortized --execute
```

Le CPU est choisi dans l’affinité autorisée. Éviter les autres calculs sur la
machine et documenter SMT, fréquence et matériel. Les workers imposent CPU,
float64, threads bornés, synchronisation JAX et timeout de 30 minutes par unité.
Les moteurs sont exécutés successivement en ordre randomisé. Le total à froid
inclut le démarrage et le contrôle serveur ; ce n’est pas un benchmark du seul
noyau Java. La traduction figure dans les configurations ; dans l’analyse à
froid elle est supplémentaire au `total_seconds`. L’analyse amortie fournit un
ratio nommé incluant la traduction. Mémoire RSS/PSS : échantillonnage de l’arbre
de processus, hors traduction ; ne pas confondre avec un pic exact garanti.

## 5. Lire les sorties

Pour chaque cas :

```text
results/NOM/cas/
  validation/protocol.json              versions liées par empreintes et seuils
  validation/summary.json               comparaison de chaque seed
  validation/cas/generated/             module autonome et manifeste
  validation/cas/SEED/                  traces GAMA/JAX et identités
  configuration.json                   configuration du batch, si demandé
  reuse/summary.json                   contrôle natif de réinitialisation
  cold/{summary,measurements,analysis}.json
  amortized/{summary,measurements,analysis}.json
```

Les sous-répertoires de mesure conservent également logs, images, sorties et
traces aux checkpoints. `reproduction_status.json` à la racine n’est publié
qu’après succès de toutes les étapes demandées et de toutes les paires. Une
sortie non nulle ou l’absence de ce fichier interdit d’annoncer une reproduction
complète. Conserver les tentatives en échec ; relancer dans un autre répertoire.
Le parcours actuel ne propose pas de reprise à mi-campagne.

Un ratio **GAMA/GAMAX > 1** favorise GAMAX. Lire chaque modèle et mode séparément.
Le bootstrap amorti est conditionnel à un seul batch par moteur/modèle/mode ;
il ne couvre pas la variance entre démarrages. Les petites commandes ne peuvent
pas démontrer un gain robuste. La GUI GAMA n’est pas reproduite : les deux moteurs
utilisent onze images du même rasteriseur commun.

Les analyses peuvent être reconstruites sans nouvelle simulation :

```bash
.venv/bin/python scripts/analyze_amortized_performance.py \
  results/reproduction-six-amortized/traffic/amortized
.venv/bin/python scripts/analyze_official_performance.py \
  results/reproduction-traffic/traffic/cold
```

## 6. Résultats historiques et périmètre publié

Les références natives nécessaires aux tests de régression sont incluses dans
`tests/fixtures/native/`, avec manifeste d’empreintes et indication des extractions.
Les mesures scalaires v0 dans `benchmarks/published_results/v0-costs.json`
permettent de recalculer les ratios de batch :

```bash
python3 scripts/check_published_results.py
```

Ce contrôle ne revérifie ni les trajectoires brutes ni les intervalles de confiance.
Le parcours reproduit la méthode sur le code du checkout, pas les deux anciennes
versions figées exactes ni les secondes observées sur la machine d’origine.
Les nouvelles mesures ne remplacent pas silencieusement les données historiques.
Les archives brutes complètes ne sont pas distribuées. Les scripts historiques
peuvent demander ces archives ; utiliser `scripts/reproduce.py` pour un clone neuf.

Le papier, les notes de rédaction, la roadmap et le guide/profil Grid’5000 sont
exclus du dépôt anonyme. Ce tutoriel couvre les expériences CPU locales ; il ne
fournit pas d’orchestration distante ni de validation GPU.

## 7. Dépannage

| Problème | Action |
|---|---|
| Paquet Python introuvable | Vérifier Python 3.12/Linux x86_64 et le verrou ; garder le diagnostic, ne pas changer les versions en silence |
| Téléchargement GAMA bloqué | Vérifier réseau et accès GitHub ; aucun binaire GAMA n’est commité |
| SHA-256 GAMA incorrect | Conserver le diagnostic ; retirer uniquement le téléchargement invalide avant nouvelle tentative |
| Dossier de sortie déjà présent | Choisir un nouveau nom ; les données ne sont pas écrasées |
| Comparaison échoue | Lire `validation/summary.json`, les `comparison.json` et logs natifs ; aucune performance ne doit contourner l’échec |
| Module/corpus modifié après validation | Refaire la validation dans un nouveau répertoire avec la même version que les mesures |
| OOM, dépassement de capacité, timeout | Conserver les traces ; ne pas augmenter implicitement les capacités ou supprimer les seeds en échec |
| GPU attendu | Ce tutoriel est CPU uniquement ; évaluation GPU hors périmètre de ce tutoriel |
