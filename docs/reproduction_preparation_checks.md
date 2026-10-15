# Contrôles de préparation du parcours reproductible

15 octobre 2026. Ces contrôles ne sont pas une nouvelle campagne scientifique.

Une copie temporaire a été construite à partir de `git ls-files --cached --others
--exclude-standard`, sans `.git`, `.venv`, `.tools` ni archives `results/`. Les
commandes de contrôle ont utilisé l’interpréteur et les dépendances déjà installés
localement, avec la copie publiée dans `PYTHONPATH`. Ce contrôle ne prouve donc
pas un téléchargement/réinstallation réussis sur une machine vierge.

Résultats :

- Empreintes des 166 fichiers de référence natifs inclus : conformes.
- `--help` de reproduce, validate_official_compilation, benchmark_official et
  benchmark_amortized : succès depuis la copie publiée.
- `pytest -q tests/test_reproduction.py` : **11 passed**, avec métadonnées
  synthétiques et plans sans exécution. Rejets testés : seed manquante, horizon,
  optimisation, module/source/corpus modifiés, tolérances modifiées et preuve
  partielle ; acceptation du cas conforme et ordre du gate de réutilisation.
- Collecte de la suite complète pytest : succès ; la suite complète n’a pas été
  exécutée à cette étape.
- `check_published_results.py` : **14 ratios modèle/mode** recalculés depuis les
  mesures scalaires historiques incluses.
- Audit de publication : aucun chemin personnel de l’équipe ni auteur LaTeX
  non anonyme détecté ; métadonnées tierces SVG conservées et signalées.
- `git diff --check` : succès pour les modifications suivies.

Aucun moteur GAMA, simulation JAX, téléchargement, réservation Grid’5000, commit
ou push n’a été lancé. Le premier essai réel documenté dans le tutoriel reste
nécessaire pour valider l’intégration complète du nouveau parcours. Les données
v0 brutes complètes ne sont pas incluses ; les références de tests et les scalaires
publiés ne les remplacent pas.
